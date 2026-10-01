"""Model artifact: save, hash, round trip, verified load, and ``freeze`` (FR-037 to FR-039,
AC-042 to AC-047, NFR-011, NFR-025; DOC-03 §15, §16.2; DN-15; DOC-05 RC-03).

Layout (DOC-03 §15.1; ``models/`` is gitignored)::

    models/staging/{model.joblib, metadata.json}   written by ``evaluate`` (overwritten)
    models/<x.y.z>/{model.joblib, metadata.json}   written by ``freeze``; never modified

**Trust boundary.** A joblib file can execute code when loaded, so ``load_verified`` checks,
in order, the metadata, the release flag (when required), the SHA-256 of ``model.joblib``
against ``metadata.model_sha256``, and the installed library versions against
``metadata.library_versions``; only then does it call ``joblib.load`` (DOC-04 §5.2 steps
3 to 6 and 10). M11's API and CLI reuse it.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price.data.load import sha256_file
from house_price.persistence.metadata import LIBRARIES, SEMVER, ArtifactMetadata

MODELS_DIR = Path("models")
STAGING_DIR = MODELS_DIR / "staging"
# Additional project requirement (M10): post-selection reference refits of the four tuned
# candidates, one directory each. Never released, never served, never read by ``freeze``.
CANDIDATES_DIR = MODELS_DIR / "candidates"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ArtifactError(RuntimeError):
    """The artifact cannot be trusted or used."""


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries a pickled pipeline depends on (DOC-03 §15.3)."""
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def save_model(model: Any, directory: Path) -> str:
    """``joblib.dump`` the complete fitted pipeline; return the file's SHA-256 (DN-15)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MODEL_FILE
    joblib.dump(model, path)
    return sha256_file(path)


def write_metadata(metadata: ArtifactMetadata, directory: Path) -> Path:
    path = directory / METADATA_FILE
    path.write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_metadata(directory: Path) -> ArtifactMetadata:
    path = directory / METADATA_FILE
    if not path.is_file():
        raise ArtifactError(f"metadata not found: {path}")
    try:
        return ArtifactMetadata.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ArtifactError(f"invalid metadata {path}: {exc}") from exc


@dataclass(frozen=True)
class RoundTrip:
    """FR-039 / AC-046: reloaded predictions equal the in-memory pipeline's exactly."""

    passed: bool
    n_rows: int
    max_abs_difference: float

    def as_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_rows": self.n_rows,
            "max_abs_difference": self.max_abs_difference,
            "comparison": "exact",
        }


def _single_threaded(model: Any) -> Any:
    """A deep copy whose nested ``n_jobs`` settings are 1. Random Forest (``n_jobs=-1``,
    DOC-03 §10.5) sums tree predictions across threads in a varying order, which changes
    the last bits of a prediction between calls; comparing single-threaded predictions
    removes that noise and nothing else (the fitted trees are untouched)."""
    clone = copy.deepcopy(model)
    pending = [clone]
    while pending:  # the fitted objects: regressor_, pipeline steps, the blend's estimators_
        obj = pending.pop()
        if hasattr(obj, "n_jobs"):
            obj.n_jobs = 1
        pending += [getattr(obj, "regressor_", None), *(getattr(obj, "estimators_", None) or [])]
        pending += [step for _, step in getattr(obj, "steps", [])]
        pending = [p for p in pending if p is not None and not isinstance(p, str)]
    return clone


def round_trip(model: Any, path: Path, X: pd.DataFrame) -> RoundTrip:
    """Reload the file just written and compare its predictions for exact equality
    (FR-039, AC-046), both computed single-threaded so the comparison is deterministic."""
    reloaded = joblib.load(path)  # the file this process has just written
    before = np.asarray(_single_threaded(model).predict(X), dtype="float64")
    after = np.asarray(_single_threaded(reloaded).predict(X), dtype="float64")
    diff = float(np.max(np.abs(before - after))) if len(before) else 0.0
    return RoundTrip(bool(np.array_equal(before, after)), len(before), diff)


def verify(directory: Path, *, require_release: bool = False) -> ArtifactMetadata:
    """Every check that must pass before the pickle may be opened. Nothing is unpickled."""
    metadata = read_metadata(directory)
    if require_release and not metadata.is_release:
        raise ArtifactError(f"{directory} is not a released artifact (is_release=false)")
    path = directory / MODEL_FILE
    if not path.is_file():
        raise ArtifactError(f"model file not found: {path}")
    actual = sha256_file(path)
    if actual != metadata.model_sha256:
        raise ArtifactError(
            f"model.joblib SHA-256 {actual} does not match metadata "
            f"{metadata.model_sha256}; the file is not loaded"
        )
    installed = library_versions()
    mismatched = {
        name: (metadata.library_versions.get(name), installed[name])
        for name in LIBRARIES
        if metadata.library_versions.get(name) != installed[name]
    }
    if mismatched:
        raise ArtifactError(
            f"library versions differ from the artifact's (recorded, installed): {mismatched}"
        )
    return metadata


def load_verified(
    directory: Path, *, require_release: bool = False
) -> tuple[Any, ArtifactMetadata]:
    """Verify, then (and only then) ``joblib.load`` the pipeline."""
    metadata = verify(directory, require_release=require_release)
    return joblib.load(directory / MODEL_FILE), metadata


# ------------------------------------------------------------------------- freeze


class FreezeError(RuntimeError):
    """``freeze`` refuses; the message lists every failed check."""


# DOC-05 §6.5: the training side, plus the training configuration (the files the training
# run reads: data, validation, schema, features, models).
PROTECTED_PATHS = (
    "src/house_price/data",
    "src/house_price/features",
    "src/house_price/pipelines",
    "src/house_price/models",
    "src/house_price/evaluation",
    "src/house_price/persistence",
    "src/house_price/tracking.py",
    "configs/data.yaml",
    "configs/validation.yaml",
    "configs/schema.yaml",
    "configs/features.yaml",
    "configs/models.yaml",
)
ALLOWED_DIRTY_PREFIX = "reports/evaluation/"  # RC-03


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreezeError(f"git {' '.join(args)} failed: {exc}") from exc


def _dirty_paths(root: Path) -> list[str]:
    """Changed or untracked paths (``git status --porcelain``), repo-relative."""
    paths = []
    for line in _git(root, "status", "--porcelain", "--untracked-files=all").splitlines():
        entry = line[3:].strip().strip('"')
        paths.append(entry.split(" -> ")[-1])
    return paths


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@dataclass
class FreezeCheck:
    problems: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    def require(self, name: str, ok: bool, problem: str) -> None:
        self.checks[name] = ok
        if not ok:
            self.problems.append(f"{name}: {problem}")


def _training_commit(record: dict[str, Any]) -> tuple[str | None, bool | None]:
    """The commit (and dirty flag) the selection record's training ran at."""
    provenance = record.get("provenance", {})
    git = provenance.get("git") or provenance.get("m8_git") or {}
    return git.get("commit"), git.get("dirty")


def check_release(
    root: Path, version_: str, tracking_uri: str, expected_rows: int
) -> tuple[FreezeCheck, ArtifactMetadata | None]:
    """Every §16.2 check (QG-10 to QG-16, smoke rejection, integrity, RC-03, DOC-05 §6.5)."""
    from mlflow.tracking import MlflowClient

    from house_price.models import selection

    fc = FreezeCheck()
    fc.require(
        "semantic_version",
        bool(SEMVER.match(version_)),
        f"{version_!r} is not a semantic version x.y.z",
    )
    target = root / MODELS_DIR / version_
    fc.require("version_is_new", not target.exists(), f"{target} already exists (immutable)")
    staging = root / STAGING_DIR
    try:
        metadata: ArtifactMetadata | None = verify(staging)
    except ArtifactError as exc:
        fc.require("staging_artifact_verified", False, str(exc))
        return fc, None
    assert metadata is not None
    fc.checks["staging_artifact_verified"] = True
    empty = metadata.empty_fields()
    fc.require("metadata_complete", not empty, f"empty or missing fields {empty}")
    fc.require("staging_not_released", not metadata.is_release, "staging is already a release")
    fc.require(
        "production_role",
        metadata.artifact_role == "production",
        f"staging holds a {metadata.artifact_role} artifact; only the production refit "
        "of the selected model can be released",
    )
    fc.require(
        "git_dirty_false", not metadata.git_dirty, "the artifact was produced from a dirty tree"
    )

    # smoke rejection (DOC-03 §16.2 step 3): the refit run's experiment (DN-17: hpp-smoke)
    client = MlflowClient(tracking_uri)
    try:
        refit = client.get_run(metadata.mlflow.refit)
        experiment = client.get_experiment(refit.info.experiment_id).name
        smoke = experiment == "hpp-smoke" or refit.data.tags.get("smoke") == "true"
        fc.require(
            "not_smoke",
            not smoke,
            f"the refit run {metadata.mlflow.refit} is a smoke run ({experiment})",
        )
        fc.require(
            "refit_run_kind",
            refit.data.tags.get("run_kind") == "production_refit",
            "the refit run is not tagged run_kind=production_refit",
        )
    except Exception as exc:  # noqa: BLE001 - any lookup failure means unverifiable provenance
        fc.require("not_smoke", False, f"refit run {metadata.mlflow.refit} not found: {exc}")

    # QG-10 to QG-12: quality_gates.json of this selection record
    gates = _read_json(root / "reports/evaluation/quality_gates.json")
    fc.require(
        "quality_gates_recorded", gates is not None, "reports/evaluation/quality_gates.json missing"
    )
    if gates is not None:
        fc.require(
            "quality_gates_same_record",
            gates.get("selection_record_id") == metadata.selection_record_id,
            "quality_gates.json is for another selection record",
        )
        fc.require(
            "quality_gates_enforced",
            gates.get("enforced") is True,
            "quality gates were not enforced (smoke)",
        )
        failed = [r["rule"] for r in gates.get("rules", []) if not r.get("passed")]
        fc.require(
            "QG-10_to_QG-12",
            bool(gates.get("rules")) and not failed,
            f"failed quality gates {failed}",
        )
    # review of the same record (DN-19; the evidence freeze verifies, DOC-03 §16.2 step 2)
    record = _read_json(root / "reports/selection/selection_record.json") or {}
    fc.require(
        "selection_record_matches",
        record.get("selection_record_id") == metadata.selection_record_id,
        "the artifact's selection record is not the "
        "committed reports/selection/selection_record.json",
    )
    review = selection.review_problems(
        root / "reports/selection/diagnostic_review.yaml", metadata.selection_record_id, smoke=False
    )
    fc.require("diagnostic_review_passes", not review, "; ".join(review))
    # QG-13: refit on all in-scope rows
    fc.require(
        "QG-13",
        metadata.training_rows == expected_rows,
        f"training_rows {metadata.training_rows} != {expected_rows}",
    )
    # QG-14: round trip exact (recorded by evaluate) and metadata complete (above)
    summary = _read_json(root / "reports/evaluation/evaluation_summary.json") or {}
    refit_info = summary.get("production_refit") or {}
    fc.require(
        "QG-14",
        bool(refit_info.get("round_trip", {}).get("passed"))
        and refit_info.get("model_sha256") == metadata.model_sha256,
        "no exact round trip recorded for this artifact",
    )
    # QG-15: reproducibility check passed for this commit. In the Release Run the check runs
    # (R2) before the selection record and review are committed (R3, R4), so "this commit"
    # means the same training code, configuration and requirements as the repro commit.
    head = _git(root, "rev-parse", "HEAD").strip()
    repro = _read_json(root / "reports/reproducibility/repro_check.json")
    repro_git = (repro or {}).get("git") or {}
    repro_ok = repro is not None and repro.get("passed") is True and repro_git.get("dirty") is False
    if repro_ok:
        changed = _git(
            root,
            "diff",
            "--name-only",
            str(repro_git.get("commit")),
            "HEAD",
            "--",
            *PROTECTED_PATHS,
            "requirements.txt",
        ).split()
        fc.require(
            "QG-15",
            not changed,
            "training code, configuration or requirements "
            f"changed since the reproducibility check: {changed[:10]}",
        )
    else:
        fc.require(
            "QG-15",
            False,
            "no passing reproducibility check (AC-033) on a clean commit "
            "(reports/reproducibility/repro_check.json)",
        )
    # QG-16: clean tree except reports/evaluation/ (RC-03); HEAD = metadata git_commit
    dirty = [p for p in _dirty_paths(root) if not p.startswith(ALLOWED_DIRTY_PREFIX)]
    fc.require(
        "QG-16_clean_tree",
        not dirty,
        f"uncommitted changes outside {ALLOWED_DIRTY_PREFIX}: {dirty[:10]}",
    )
    fc.require(
        "QG-16_head",
        head == metadata.git_commit,
        f"HEAD {head} differs from the artifact's git_commit {metadata.git_commit}",
    )
    # DOC-05 §6.5 training-code freeze: no protected change since the training run. (The
    # training run may be recorded dirty: in the Release Run the repro report of R2 is still
    # uncommitted when R3 trains; protected changes are caught by the diff and QG-16.)
    commit, _ = _training_commit(record)
    if commit is None:
        fc.require(
            "training_code_frozen",
            False,
            "the selection record has no training commit; re-run train (DOC-05 §6.5)",
        )
    else:
        changed = _git(root, "diff", "--name-only", commit, "HEAD", "--", *PROTECTED_PATHS).split()
        fc.require(
            "training_code_frozen",
            not changed,
            "training code/configuration changed "
            f"since the training run {commit[:12]}: {changed[:10]}; re-run train and the "
            "review (DOC-05 §6.5)",
        )
    return fc, metadata


@dataclass(frozen=True)
class FreezeResult:
    version: str
    directory: Path
    model_sha256: str
    release_run_id: str
    checks: dict[str, bool]


def freeze(version_: str, root: Path, config_dir: Path, tracking_uri: str) -> FreezeResult:
    """DOC-03 §16.2: verify every gate, copy byte-for-byte, mark released, log hpp-release."""
    from house_price.config import load_project_config
    from house_price.tracking import Lineage, Tracker

    config = load_project_config(config_dir, root)
    fc, metadata = check_release(
        root, version_, tracking_uri, config.data.scope.expected_rows_after
    )
    if fc.problems or metadata is None:
        raise FreezeError("freeze refuses:\n  - " + "\n  - ".join(fc.problems))
    staging, target = root / STAGING_DIR, root / MODELS_DIR / version_
    target.mkdir(parents=True)
    shutil.copyfile(staging / MODEL_FILE, target / MODEL_FILE)  # byte for byte
    if sha256_file(target / MODEL_FILE) != metadata.model_sha256:
        raise FreezeError("the copied model.joblib does not match model_sha256")
    lineage = Lineage(
        pipeline_run_id=metadata.pipeline_run_id,
        git_commit=metadata.git_commit,
        git_dirty=metadata.git_dirty,
        data_sha256=metadata.data_sha256,
        split_manifest_sha256=metadata.split_manifest_sha256,
        config_hash=metadata.config_hash,
        seed=metadata.seed,
    )
    tracker = Tracker(tracking_uri, lineage)
    with tracker.run(
        "release",
        stage="release",
        candidate=metadata.selected_candidate,
        run_name=f"release_{version_}",
        extra_tags={"model_version": version_, "selection_record_id": metadata.selection_record_id},
    ) as run:
        released = metadata.model_copy(
            update={
                "model_version": version_,
                "is_release": True,
                "mlflow": metadata.mlflow.model_copy(update={"release": run.run_id}),
            }
        )
        write_metadata(released, target)
        run.log_params({"model_version": version_, "model_sha256": metadata.model_sha256})
        run.log_artifact(target / MODEL_FILE)
        run.log_artifact(target / METADATA_FILE)
    return FreezeResult(version_, target, metadata.model_sha256, run.run_id, fc.checks)

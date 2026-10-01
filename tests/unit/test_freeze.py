"""``freeze`` (DOC-03 §16.2, QG-10 to QG-16; DOC-05 M10-4, RC-03, §6.5), on a controlled
temporary git repository with a small fixture artifact. No real release is created."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from house_price.cli import main as cli_main
from house_price.data.load import sha256_file
from house_price.models import selection
from house_price.persistence import artifact
from house_price.persistence.artifact import FreezeError, freeze
from house_price.tracking import Lineage, Tracker
from tests.conftest import CONFIG_DIR, artifact_metadata

EXPECTED_ROWS = 2925  # configs/data.yaml scope.expected_rows_after


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD")


@dataclass
class Release:
    root: Path
    uri: str
    training_commit: str

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    def freeze(self, version: str = "1.0.0") -> artifact.FreezeResult:
        return freeze(version, self.root, self.config_dir, self.uri)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _stage(rel: Release, *, smoke: bool = False, **overrides: object) -> None:
    """A staging artifact for HEAD with refit / final-evaluation runs in MLflow."""
    head = git(rel.root, "rev-parse", "HEAD")
    meta0 = artifact_metadata(git_commit=head)
    lineage = Lineage(pipeline_run_id="run-1", git_commit=head, git_dirty=False,
                      data_sha256="c" * 64, split_manifest_sha256="d" * 64,
                      config_hash="e" * 64, seed=42)  # fmt: skip
    tracker = Tracker(rel.uri, lineage, experiment_override="smoke" if smoke else None)
    with tracker.run("evaluation", stage="evaluation", candidate="ridge",
                     run_kind="final_holdout_evaluation") as final:  # fmt: skip
        pass
    with tracker.run("evaluation", stage="evaluation", candidate="ridge",
                     run_kind="production_refit") as refit:  # fmt: skip
        pass
    X = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    model = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    model.fit(X, np.exp(10 + 0.1 * X["x"]))
    staging = rel.root / artifact.STAGING_DIR
    sha = artifact.save_model(model, staging)
    values = {"model_sha256": sha, "training_rows": EXPECTED_ROWS,
              "mlflow": meta0.mlflow.model_copy(  # type: ignore[attr-defined]
                  update={"refit": refit.run_id, "final_holdout_evaluation": final.run_id}),
              **overrides}  # fmt: skip
    artifact.write_metadata(meta0.model_copy(update=values), staging)  # type: ignore[attr-defined]
    _write(
        rel.root / "reports/evaluation/evaluation_summary.json",
        json.dumps({"production_refit": {"model_sha256": sha, "round_trip": {"passed": True}}}),
    )
    _write(
        rel.root / "reports/evaluation/quality_gates.json",
        json.dumps(
            {
                "selection_record_id": "rec-1",
                "enforced": True,
                "rules": [{"rule": "holdout_log_rmse", "passed": True}],
            }
        ),
    )


@pytest.fixture
def release(tmp_path: Path) -> Release:
    root = tmp_path / "repo"
    shutil.copytree(CONFIG_DIR, root / "configs")
    _write(root / "src/house_price/models/placeholder.py", "VALUE = 1\n")
    _write(root / ".gitignore", "/models/\n/mlruns/\n")  # root-anchored, as in the project
    git(root.parent, "init", "-q", str(root))
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "freeze-test")
    training = commit_all(root, "training code")
    record = {"selection_record_id": "rec-1",
              "provenance": {"git": {"commit": training, "dirty": False}}}  # fmt: skip
    review = {"selection_record_id": "rec-1", "status": "final", "reviewer": "freeze-test",
              "reviewed_at": "2026-10-01",
              "gates": {g: {"result": "pass", "reasoning": "fixture"}
                        for g in selection.REVIEW_GATES}}  # fmt: skip
    _write(root / "reports/selection/selection_record.json", json.dumps(record))
    _write(root / "reports/selection/diagnostic_review.yaml", yaml.safe_dump(review))
    _write(
        root / "reports/reproducibility/repro_check.json",
        json.dumps({"passed": True, "git": {"commit": training, "dirty": False}}),
    )
    commit_all(root, "selection record, review, repro report")
    rel = Release(root, (root / "mlruns").as_uri(), training)
    _stage(rel)
    return rel


def _refused(rel: Release, check: str, version: str = "1.0.0") -> str:
    with pytest.raises(FreezeError) as caught:
        rel.freeze(version)
    assert f"{check}:" in str(caught.value), str(caught.value)
    assert not (rel.root / "models" / version).exists() or version == "existing"
    return str(caught.value)


def test_a_complete_release_freezes(release: Release) -> None:
    result = release.freeze("1.0.0")
    target = release.root / "models/1.0.0"
    staged = release.root / artifact.STAGING_DIR / "model.joblib"
    assert (target / "model.joblib").read_bytes() == staged.read_bytes()  # byte for byte
    meta = artifact.read_metadata(target)
    assert (meta.is_release, meta.model_version) == (True, "1.0.0")
    assert meta.mlflow.release == result.release_run_id and meta.empty_fields() == []
    assert meta.model_sha256 == sha256_file(target / "model.joblib") == result.model_sha256
    client = MlflowClient(release.uri)
    run = client.get_run(result.release_run_id)
    assert client.get_experiment(run.info.experiment_id).name == "hpp-release"
    assert {a.path for a in client.list_artifacts(run.info.run_id)} == {
        "model.joblib",
        "metadata.json",
    }
    artifact.load_verified(target, require_release=True)  # the released artifact loads


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "01.0.0", "1.0.0-rc1", "latest", ""])
def test_invalid_versions_are_refused(release: Release, version: str) -> None:
    with pytest.raises(FreezeError, match="semantic_version"):
        release.freeze(version)


def test_an_existing_version_is_refused(release: Release) -> None:
    release.freeze("1.0.0")
    with pytest.raises(FreezeError, match="version_is_new"):
        release.freeze("1.0.0")


def test_a_failed_quality_gate_is_refused(release: Release) -> None:
    _write(release.root / "reports/evaluation/quality_gates.json", json.dumps(
        {"selection_record_id": "rec-1", "enforced": True,
         "rules": [{"rule": "holdout_log_rmse", "passed": False}]}))  # fmt: skip
    _refused(release, "QG-10_to_QG-12")


def test_a_smoke_artifact_is_refused(release: Release) -> None:
    _stage(release, smoke=True, training_rows=200)  # refit run in hpp-smoke
    message = _refused(release, "not_smoke")
    assert "QG-13:" in message  # 200 != 2925 rows as well


def test_smoke_gates_are_refused(release: Release) -> None:
    _write(release.root / "reports/evaluation/quality_gates.json", json.dumps(
        {"selection_record_id": "rec-1", "enforced": False,
         "rules": [{"rule": "holdout_log_rmse", "passed": True}]}))  # fmt: skip
    _refused(release, "quality_gates_enforced")


def test_a_dirty_tree_is_refused(release: Release) -> None:
    _write(release.root / "notes.txt", "uncommitted\n")
    _refused(release, "QG-16_clean_tree")


def test_changes_under_reports_evaluation_are_allowed(release: Release) -> None:
    _write(release.root / "reports/evaluation/extra_report.json", "{}")  # RC-03
    assert release.freeze("1.0.0").version == "1.0.0"


def test_a_head_mismatch_is_refused(release: Release) -> None:
    _write(release.root / "docs/note.md", "later commit\n")
    commit_all(release.root, "docs after the evaluation")
    _refused(release, "QG-16_head")


def test_training_code_changes_after_training_are_refused(release: Release) -> None:
    _write(release.root / "src/house_price/models/placeholder.py", "VALUE = 2\n")
    commit_all(release.root, "change training code")
    _stage(release)  # even a fresh evaluate at the new HEAD is not enough (DOC-05 §6.5)
    message = _refused(release, "training_code_frozen")
    assert "QG-15:" in message  # the repro check no longer covers this code either


def test_training_configuration_changes_are_refused(release: Release) -> None:
    models = release.root / "configs/models.yaml"
    models.write_text(models.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    commit_all(release.root, "change training configuration")
    _stage(release)
    _refused(release, "training_code_frozen")


def test_a_missing_reproducibility_check_is_refused(release: Release) -> None:
    (release.root / "reports/reproducibility/repro_check.json").unlink()
    commit_all(release.root, "remove repro report")
    _stage(release)
    _refused(release, "QG-15")


def test_a_failed_reproducibility_check_is_refused(release: Release) -> None:
    _write(release.root / "reports/reproducibility/repro_check.json",
           json.dumps({"passed": False, "git": {"commit": release.training_commit,
                                                "dirty": False}}))  # fmt: skip
    commit_all(release.root, "failed repro")
    _stage(release)
    _refused(release, "QG-15")


def test_training_rows_must_be_all_in_scope_rows(release: Release) -> None:
    _stage(release, training_rows=2340)
    _refused(release, "QG-13")


def test_a_missing_round_trip_is_refused(release: Release) -> None:
    _write(release.root / "reports/evaluation/evaluation_summary.json", json.dumps({}))
    _refused(release, "QG-14")


def test_a_dirty_artifact_is_refused(release: Release) -> None:
    _stage(release, git_dirty=True)
    _refused(release, "git_dirty_false")


def test_a_tampered_staging_artifact_is_refused(release: Release) -> None:
    with (release.root / artifact.STAGING_DIR / "model.joblib").open("ab") as handle:
        handle.write(b"x")
    _refused(release, "staging_artifact_verified")


def test_an_unsigned_review_is_refused(release: Release) -> None:
    path = release.root / "reports/selection/diagnostic_review.yaml"
    review = yaml.safe_load(path.read_text(encoding="utf-8"))
    review.update(status="draft", reviewer=None)
    path.write_text(yaml.safe_dump(review), encoding="utf-8")
    commit_all(release.root, "draft review")
    _stage(release)
    _refused(release, "diagnostic_review_passes")


def test_cli_freeze_refuses_with_a_message(
    release: Release, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli_main(["freeze", "--version", "not-a-version", "--root", str(release.root),
                     "--tracking-uri", release.uri])  # fmt: skip
    assert code == 1 and "semantic_version" in capsys.readouterr().err
    assert cli_main(["freeze", "--version", "1.0.0", "--root", str(release.root),
                     "--tracking-uri", release.uri]) == 0  # fmt: skip


def test_a_candidate_artifact_copied_into_staging_is_refused(release: Release) -> None:
    """Additional requirement: candidate reference artifacts are never releasable."""
    staging = release.root / artifact.STAGING_DIR
    meta = artifact.read_metadata(staging)
    candidate = meta.model_copy(update={
        "artifact_role": "candidate", "holdout": None, "baseline_reference": None,
        "temporal_diagnostic": None, "quality_gates": None,
        "mlflow": meta.mlflow.model_copy(update={"final_holdout_evaluation": None}),
    })  # fmt: skip
    artifact.write_metadata(candidate, staging)
    _refused(release, "production_role")

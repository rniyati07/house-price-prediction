"""Training orchestration: ``house-price train`` / ``make train`` (FR-023, DOC-03 §3.2, §4.2).

At M6 the run is:

    load + hash check -> validate -> scope rule -> load the persisted split
    -> generate the shared folds (artifacts/cv/folds.json) -> CV of both baselines
    -> log each to MLflow (hpp-baselines) -> stop

Each run also writes an M6 run record (``results/M6/.../result.json``) whose ``run_id`` is
the ``pipeline_run_id`` tagged on the MLflow runs: it summarizes the baselines and links to
their MLflow runs; the fold-level detail stays in MLflow.

Later milestones extend this orchestrator (ablation, tuning, final comparison, selection).
The split is only loaded: if the manifest is missing, training stops instead of creating
one. Only the development set is used; the holdout rows are never read here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    FeatureConfig,
    ModelsConfig,
    ProjectConfig,
    ValidationConfig,
    config_hash,
    load_feature_config,
    load_models_config,
    load_project_config,
)
from house_price.data.errors import DataError
from house_price.data.load import load_raw, sha256_file
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.evaluation.cv import CVResult, FoldPlan, generate_folds, run_cv
from house_price.models.registry import baseline_candidates
from house_price.results import ResultRun, start_run
from house_price.tracking import (
    EXPERIMENTS,
    Lineage,
    Tracker,
    default_tracking_uri,
    git_state,
    new_pipeline_run_id,
)

FOLDS_PATH = Path("artifacts/cv/folds.json")  # DOC-03 §9.2


class TrainError(DataError):
    """Training cannot start on the current project state."""


@dataclass(frozen=True)
class TrainResult:
    pipeline_run_id: str
    folds_path: Path
    results: dict[str, CVResult]
    run_ids: dict[str, str]
    dev_rows: int
    dev_log_price_sd: float
    result_path: Path | None = None  # the M6 run record


@dataclass(frozen=True)
class _Provenance:
    """What the run record needs beyond the results: identity and settings of the run."""

    lineage: Lineage
    tracking_uri: str
    manifest_path: Path
    validation: ValidationConfig


def training_config_hash(
    config: ProjectConfig, features: FeatureConfig, models: ModelsConfig
) -> str:
    """``config_hash`` of every configuration file the training run reads (DOC-03 §5.1)."""
    return config_hash(config.data, config.validation, config.schema, features, models)


def run_train(
    config_dir: Path | None = None,
    root: Path | None = None,
    tracking_uri: str | None = None,
    results_dir: Path | None = None,
    argv: list[str] | None = None,
) -> TrainResult:
    """Run the M6 training flow, record it, and return the baseline results."""
    root = (root or Path.cwd()).resolve()
    pipeline_run_id = new_pipeline_run_id()
    with start_run("M6", root=root, results_dir=results_dir, run_id=pipeline_run_id,
                   entry_point="house-price train", argv=argv) as record:  # fmt: skip
        result, provenance = _train(config_dir or root / DEFAULT_CONFIG_DIR, root,
                                    tracking_uri, pipeline_run_id)  # fmt: skip
        _record(record, result, provenance)
        return replace(result, result_path=record.path)


def _train(
    config_dir: Path, root: Path, tracking_uri: str | None, pipeline_run_id: str
) -> tuple[TrainResult, _Provenance]:
    config = load_project_config(config_dir, root)
    features, models = load_feature_config(config_dir), load_models_config(config_dir)
    if not config.manifest_path.is_file():
        raise TrainError(
            f"split manifest not found at {config.manifest_path}; create the split first "
            "(make split). Training never creates or regenerates it."
        )

    raw = load_raw(config)
    in_scope, scope_record = apply_scope_rule(raw, config.data.scope, config.schema.id_column)
    split = create_or_load_split(in_scope, scope_record, config)
    dev, validation = split.dev, config.validation

    plan = generate_folds(
        dev, id_column=config.schema.id_column, target=config.schema.target,
        n_bins=validation.n_bins, n_splits=validation.cv_folds, n_repeats=validation.cv_repeats,
        seed=validation.seed, dev_sha256=split.manifest.dev_sha256,
    )  # fmt: skip
    folds_path = root / FOLDS_PATH
    plan.save(folds_path)
    plan = FoldPlan.load(folds_path, split.manifest.dev_sha256)  # every CV reads the file

    commit, dirty = git_state()
    lineage = Lineage(
        pipeline_run_id=pipeline_run_id, git_commit=commit, git_dirty=dirty,
        data_sha256=config.data.raw_sha256, split_manifest_sha256=sha256_file(config.manifest_path),
        config_hash=training_config_hash(config, features, models), seed=validation.seed,
    )  # fmt: skip
    tracker = Tracker(tracking_uri or default_tracking_uri(root), lineage)

    results: dict[str, CVResult] = {}
    run_ids: dict[str, str] = {}
    for candidate in baseline_candidates(models):
        result = run_cv(candidate.build(features, config.schema), dev, plan, config.schema)
        with tracker.run("baselines", stage="baselines", candidate=candidate.name,
                         run_name=candidate.name) as run:  # fmt: skip
            run.log_params({
                **candidate.params, "branch": candidate.branch,
                "engineered_features": ",".join(candidate.engineered_features(features)) or "none",
                "n_folds": validation.cv_folds, "n_repeats": validation.cv_repeats,
                "eligible_for_selection": candidate.eligible,
            })  # fmt: skip
            run.log_metrics({
                **{f"fold_{i:02d}": score for i, score in enumerate(result.fold_scores, start=1)},
                "cv_mean": result.mean, "cv_se": result.se, **result.secondary,
                "duration_seconds": result.duration_seconds,
            })  # fmt: skip
        results[candidate.name] = result
        run_ids[candidate.name] = run.run_id

    target = dev[config.schema.target].to_numpy(dtype="float64")
    trained = TrainResult(
        pipeline_run_id=lineage.pipeline_run_id, folds_path=folds_path, results=results,
        run_ids=run_ids, dev_rows=len(dev), dev_log_price_sd=float(np.log1p(target).std(ddof=1)),
    )  # fmt: skip
    return trained, _Provenance(lineage, tracker.tracking_uri, config.manifest_path, validation)


def _record(run: ResultRun, result: TrainResult, provenance: _Provenance) -> None:
    """Summarize the training run and link it to MLflow (no fold-level duplication)."""
    lineage, validation = provenance.lineage, provenance.validation
    run.log_lineage(
        data_sha256=lineage.data_sha256, split_manifest_sha256=lineage.split_manifest_sha256,
        config_hash=lineage.config_hash,
        config_hash_covers=["data", "validation", "schema", "features", "models"],
        seed=lineage.seed,
    )  # fmt: skip
    run.log_params({"candidates": list(result.results), "cv_folds": validation.cv_folds,
                    "cv_repeats": validation.cv_repeats, "n_bins": validation.n_bins})  # fmt: skip
    run.log_metrics({"dev_rows": result.dev_rows, "dev_log_price_sd": result.dev_log_price_sd})
    for name, cv in result.results.items():
        run.log_metrics({f"{name}.cv_mean": cv.mean, f"{name}.cv_se": cv.se,
                         **{f"{name}.{k}": v for k, v in cv.secondary.items()},
                         f"{name}.duration_seconds": cv.duration_seconds})  # fmt: skip
    run.log_value("mlflow", {
        "tracking_uri": provenance.tracking_uri, "experiment": EXPERIMENTS["baselines"],
        "pipeline_run_id": result.pipeline_run_id, "runs": result.run_ids,
        "detail": "per-fold scores, parameters and standard tags are in these MLflow runs",
    })  # fmt: skip
    if {"dummy_median", "linear_2feat"} <= set(result.results):
        run.log_value("plausibility", plausibility(result))
    run.log_artifact(result.folds_path, description="shared CV folds (DN-03)")
    run.log_artifact(provenance.manifest_path, description="persisted split manifest (read only)")


def plausibility(result: TrainResult) -> dict[str, Any]:
    """DOC-05 M6-6, read off this run's results: the dummy scores near the SD of log price
    and the two-feature heuristic beats it by more than one standard error."""
    dummy, heuristic = result.results["dummy_median"], result.results["linear_2feat"]
    return {
        "dummy_cv_mean_over_dev_log_price_sd": dummy.mean / result.dev_log_price_sd,
        "heuristic_minus_dummy_cv_mean": heuristic.mean - dummy.mean,
        "dummy_cv_se": dummy.se,
        "heuristic_beats_dummy_by_more_than_one_se": heuristic.mean < dummy.mean - dummy.se,
    }


def report(result: TrainResult) -> str:
    """Human-readable summary printed by the CLI."""
    lines = [
        f"pipeline_run_id: {result.pipeline_run_id}",
        f"shared folds: {result.folds_path}",
        f"development rows: {result.dev_rows}; SD of log1p(SalePrice): {result.dev_log_price_sd:.4f}",
        "",
        f"{'candidate':<14}{'cv_mean':>10}{'cv_se':>10}{'cv_mae':>12}{'cv_mape':>10}{'cv_r2':>9}",
    ]
    for name, cv in result.results.items():
        lines.append(
            f"{name:<14}{cv.mean:>10.4f}{cv.se:>10.4f}{cv.secondary['cv_mae']:>12,.0f}"
            f"{cv.secondary['cv_mape']:>9.2f}%{cv.secondary['cv_r2']:>9.3f}"
        )
    lines.append("")
    lines.append("Logged to MLflow experiment hpp-baselines (inspect with: mlflow ui).")
    if result.result_path is not None:
        lines.append(f"Run record: {result.result_path}")
    return "\n".join(lines)

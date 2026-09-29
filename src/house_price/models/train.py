"""Training orchestration: ``house-price train`` / ``make train`` (FR-023, DOC-03 §3.2, §4.2).

At M6 the run is:

    load + hash check -> validate -> scope rule -> load the persisted split
    -> generate the shared folds (artifacts/cv/folds.json) -> CV of both baselines
    -> log each to MLflow (hpp-baselines) -> stop

Later milestones extend this orchestrator (ablation, tuning, final comparison, selection).
The split is only loaded: if the manifest is missing, training stops instead of creating
one. Only the development set is used; the holdout rows are never read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    FeatureConfig,
    ModelsConfig,
    ProjectConfig,
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
from house_price.tracking import (
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


def training_config_hash(
    config: ProjectConfig, features: FeatureConfig, models: ModelsConfig
) -> str:
    """``config_hash`` of every configuration file the training run reads (DOC-03 §5.1)."""
    return config_hash(config.data, config.validation, config.schema, features, models)


def run_train(
    config_dir: Path | None = None, root: Path | None = None, tracking_uri: str | None = None
) -> TrainResult:
    """Run the M6 training flow and return the baseline results."""
    root = (root or Path.cwd()).resolve()
    config_dir = config_dir or root / DEFAULT_CONFIG_DIR
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
        pipeline_run_id=new_pipeline_run_id(), git_commit=commit, git_dirty=dirty,
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
    return TrainResult(
        pipeline_run_id=lineage.pipeline_run_id, folds_path=folds_path, results=results,
        run_ids=run_ids, dev_rows=len(dev), dev_log_price_sd=float(np.log1p(target).std(ddof=1)),
    )  # fmt: skip


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
    return "\n".join(lines)

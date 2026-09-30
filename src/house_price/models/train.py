"""Training orchestration: ``house-price train`` / ``make train`` (FR-023, DOC-03 §3.2, §4.2).

At M7 the run is:

    load + hash check -> validate -> scope rule -> load the persisted split
    -> generate the shared folds (artifacts/cv/folds.json)
    -> CV of both baselines, logged to MLflow (hpp-baselines)                       [M6]
    -> per-branch feature ablation, logged to hpp-ablation; table saved as E-30     [M7]
    -> RC-02: compare the ablation outcome with the one committed in features.yaml
         no committed outcome, or a different one -> STOP (nothing is written)
    -> development check of Ridge, Lasso, Random Forest, LightGBM with their reference
       configurations on the committed feature sets, logged to hpp-cv-comparison
       (stage=development_check; never used for selection)                          [M7]
    -> stop (tuning and the final comparison are M8)

Each run writes one run record (``results/M7/.../result.json``) whose ``run_id`` is the
``pipeline_run_id`` tagged on every MLflow run; it summarizes the stages and links to their
MLflow runs. An RC-02 stop is recorded as ``stopped``, not ``failed``.

The split is only loaded: if the manifest is missing, training stops instead of creating
one. Only the development set is used; the holdout rows are never read here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from house_price.models import ablation
from house_price.models.ablation import BranchAblation, OutcomeCheck
from house_price.models.registry import Candidate, baseline_candidates, registry
from house_price.pipelines.branches import BRANCHES
from house_price.results import ResultRun, start_run
from house_price.tracking import (
    EXPERIMENTS,
    Lineage,
    TrackedRun,
    Tracker,
    default_tracking_uri,
    git_state,
    new_pipeline_run_id,
)

FOLDS_PATH = Path("artifacts/cv/folds.json")  # DOC-03 §9.2
ABLATION_DIR = Path("artifacts/ablation")  # grid tables (gitignored, like artifacts/cv)
MILESTONE = "M7"
EXIT_RC02_STOP = 3  # train stopped on purpose by the RC-02 ablation check

Log = Callable[[str], None]


class TrainError(DataError):
    """Training cannot start on the current project state."""


@dataclass(frozen=True)
class TrainResult:
    pipeline_run_id: str
    folds_path: Path
    results: dict[str, CVResult]  # the baselines
    run_ids: dict[str, str]  # baseline MLflow runs
    dev_rows: int
    dev_log_price_sd: float
    result_path: Path | None = None  # the run record
    ablation: dict[str, BranchAblation] = field(default_factory=dict)
    ablation_table_path: Path | None = None  # E-30
    ablation_run_ids: dict[str, str] = field(default_factory=dict)  # parent run per branch
    outcome_check: OutcomeCheck | None = None
    dev_checks: dict[str, CVResult] = field(default_factory=dict)
    dev_check_run_ids: dict[str, str] = field(default_factory=dict)
    dev_check_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def stopped(self) -> bool:
        """RC-02 stopped the run before the development checks."""
        return self.outcome_check is not None and not self.outcome_check.passed


@dataclass(frozen=True)
class _Provenance:
    """What the run record needs beyond the results: identity and settings of the run."""

    lineage: Lineage
    tracking_uri: str
    manifest_path: Path
    validation: ValidationConfig
    grid_paths: dict[str, Path]


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
    log: Log | None = None,
) -> TrainResult:
    """Run the training flow, record it, and return what it produced.

    An RC-02 stop is a normal return with ``result.stopped`` set (and the record
    ``stopped``); errors raise.
    """
    root = (root or Path.cwd()).resolve()
    pipeline_run_id = new_pipeline_run_id()
    with start_run(MILESTONE, root=root, results_dir=results_dir, run_id=pipeline_run_id,
                   entry_point="house-price train", argv=argv) as record:  # fmt: skip
        result, provenance = _train(config_dir or root / DEFAULT_CONFIG_DIR, root,
                                    tracking_uri, pipeline_run_id, log or (lambda _: None))  # fmt: skip
        _record(record, result, provenance)
        if result.stopped and result.outcome_check is not None:
            record.mark_stopped(result.outcome_check.message())
        return replace(result, result_path=record.path)


def _train(
    config_dir: Path, root: Path, tracking_uri: str | None, pipeline_run_id: str, log: Log
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
    dev, validation, schema = split.dev, config.validation, config.schema

    plan = generate_folds(
        dev, id_column=schema.id_column, target=schema.target,
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

    # ---------------------------------------------------------------- M6: baselines
    log("baselines: dummy_median, linear_2feat")
    results: dict[str, CVResult] = {}
    run_ids: dict[str, str] = {}
    for candidate in baseline_candidates(models):
        result = run_cv(candidate.build(features, schema), dev, plan, schema)
        with tracker.run("baselines", stage="baselines", candidate=candidate.name,
                         run_name=candidate.name) as run:  # fmt: skip
            run.log_params({
                **candidate.params, "branch": candidate.branch,
                "engineered_features": ",".join(candidate.engineered_features(features)) or "none",
                "n_folds": validation.cv_folds, "n_repeats": validation.cv_repeats,
                "eligible_for_selection": candidate.eligible,
            })  # fmt: skip
            _log_cv_metrics(run, result)
        results[candidate.name] = result
        run_ids[candidate.name] = run.run_id

    target = dev[schema.target].to_numpy(dtype="float64")
    trained = TrainResult(
        pipeline_run_id=lineage.pipeline_run_id, folds_path=folds_path, results=results,
        run_ids=run_ids, dev_rows=len(dev), dev_log_price_sd=float(np.log1p(target).std(ddof=1)),
    )  # fmt: skip

    # ------------------------------------------------------------------ M7: ablation
    candidates = registry(models, validation.seed)
    references = {"linear": candidates[models.ablation.linear],
                  "tree": candidates[models.ablation.tree]}  # fmt: skip
    branch_results: dict[str, BranchAblation] = {}
    for branch in BRANCHES:
        log(f"ablation ({branch} branch, reference {references[branch].name}): "
            f"full feature set + {len(features.engineered)} leave-one-out runs x 15 folds")  # fmt: skip
        branch_results[branch] = ablation.run_branch_ablation(
            branch, references[branch], dev, plan, features, schema
        )
    table = ablation.ablation_table(branch_results)
    table_path = ablation.save_table(table, root / ablation.E30_PATH)
    grid_paths = {
        branch: ablation.save_table(
            result.grid.table(), root / ABLATION_DIR / f"{branch}_{result.reference}_grid.csv"
        )
        for branch, result in branch_results.items() if result.grid is not None
    }  # fmt: skip
    ablation_run_ids = {
        branch: _log_ablation(tracker, result, table_path, grid_paths.get(branch), validation)
        for branch, result in branch_results.items()
    }
    check = ablation.check_outcome(features, ablation.proposed_outcome(branch_results))
    trained = replace(
        trained, ablation=branch_results, ablation_table_path=table_path,
        ablation_run_ids=ablation_run_ids, outcome_check=check,
    )  # fmt: skip
    provenance = _Provenance(lineage, tracker.tracking_uri, config.manifest_path, validation,
                             grid_paths)  # fmt: skip
    if not check.passed:
        log(check.message())
        return trained, provenance
    log(check.message())

    # ----------------------------------------------------- M7: development checks
    dev_checks: dict[str, CVResult] = {}
    dev_run_ids: dict[str, str] = {}
    dev_params: dict[str, dict[str, Any]] = {}
    for candidate in candidates.values():
        if not candidate.eligible:
            continue  # the baselines ran above
        log(f"development check: {candidate.name}")
        overrides = _reference_overrides(candidate, branch_results)
        template = candidate.build(features, schema, overrides)
        effective = template.regressor.named_steps["model"].get_params()
        result = run_cv(template, dev, plan, schema)
        with tracker.run("cv_comparison", stage="development_check", candidate=candidate.name,
                         run_name=f"{candidate.name}_development_check") as run:  # fmt: skip
            run.log_params({
                **effective, "branch": candidate.branch, "tier": candidate.tier,
                "engineered_features": ",".join(candidate.engineered_features(features)) or "none",
                "dropped_engineered": ",".join(
                    getattr(features, candidate.branch).dropped_engineered or []) or "none",
                "n_folds": validation.cv_folds, "n_repeats": validation.cv_repeats,
                "used_for_selection": False,
            })  # fmt: skip
            _log_cv_metrics(run, result)
        dev_checks[candidate.name] = result
        dev_run_ids[candidate.name] = run.run_id
        dev_params[candidate.name] = {k: _plain(v) for k, v in effective.items()}
    trained = replace(trained, dev_checks=dev_checks, dev_check_run_ids=dev_run_ids,
                      dev_check_params=dev_params)  # fmt: skip
    return trained, provenance


def _reference_overrides(
    candidate: Candidate, branch_results: dict[str, BranchAblation]
) -> dict[str, object]:
    """The reference configuration's missing grid parameter: Ridge's ``alpha`` is the one
    its ablation grid chose (DOC-03 §6.6 step 2). A grid parameter with no value is an
    error rather than a silent scikit-learn default."""
    if candidate.grid is None or candidate.grid.param in candidate.params:
        return {}
    for result in branch_results.values():
        if result.reference == candidate.name and result.grid is not None:
            return {result.grid.param: result.grid.best_value}
    raise TrainError(f"{candidate.name}: no reference value for {candidate.grid.param!r}")


def _plain(value: Any) -> Any:
    return value if value is None or isinstance(value, (bool, int, float, str)) else str(value)


def _log_cv_metrics(run: TrackedRun, result: CVResult) -> None:
    run.log_metrics({
        **{f"fold_{i:02d}": score for i, score in enumerate(result.fold_scores, start=1)},
        "cv_mean": result.mean, "cv_se": result.se, **result.secondary,
        "duration_seconds": result.duration_seconds,
    })  # fmt: skip


def _log_ablation(
    tracker: Tracker,
    result: BranchAblation,
    table_path: Path,
    grid_path: Path | None,
    validation: ValidationConfig,
) -> str:
    """One parent run per branch, one nested run per removed feature (DOC-03 §14.2)."""
    table = result.table()
    with tracker.run("ablation", stage="ablation", candidate=result.reference,
                     run_name=f"ablation_{result.branch}") as parent:  # fmt: skip
        parent.log_params({
            **result.reference_params, "branch": result.branch,
            "reference_model": result.reference, "n_folds": validation.cv_folds,
            "n_repeats": validation.cv_repeats,
            "dropped_engineered": ",".join(result.dropped) or "none",
        })  # fmt: skip
        parent.log_metrics({
            "mean_with": result.full.mean, "se_with": result.full.se,
            "n_dropped": len(result.dropped),
            **{f"fold_{i:02d}": s for i, s in enumerate(result.full.fold_scores, start=1)},
        })  # fmt: skip
        if result.grid is not None:
            parent.log_params({f"grid_best_{result.grid.param}": result.grid.best_value,
                               "grid_best_on_edge": result.grid.on_edge})  # fmt: skip
        parent.log_artifact(table_path)
        if grid_path is not None:
            parent.log_artifact(grid_path)
        for row in table.itertuples():
            removed = result.without[str(row.feature)]
            with tracker.run("ablation", stage="ablation", candidate=result.reference,
                             run_name=f"without_{row.feature}",
                             parent_run_id=parent.run_id) as child:  # fmt: skip
                child.log_params({"branch": result.branch, "feature_removed": row.feature})
                child.log_metrics({
                    "mean_with": float(row.mean_with), "mean_without": float(row.mean_without),
                    "delta": float(row.delta), "delta_se": float(row.delta_se),
                    "retained": float(bool(row.retained)),
                    **{f"fold_{i:02d}": s for i, s in enumerate(removed.fold_scores, start=1)},
                })  # fmt: skip
    return parent.run_id


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
        _record_cv(run, name, cv)
    runs: dict[str, Any] = {EXPERIMENTS["baselines"]: result.run_ids}
    if {"dummy_median", "linear_2feat"} <= set(result.results):
        run.log_value("plausibility", plausibility(result))
    run.log_artifact(result.folds_path, description="shared CV folds (DN-03)")
    run.log_artifact(provenance.manifest_path, description="persisted split manifest (read only)")

    if result.ablation:
        run.log_params(
            {"ablation.references": {b: r.reference for b, r in result.ablation.items()}}
        )
        run.log_value("ablation", {
            branch: {
                "reference_model": r.reference, "reference_params": r.reference_params,
                "mean_with": r.full.mean, "se_with": r.full.se,
                "grid": None if r.grid is None else {
                    "param": r.grid.param, "best_value": r.grid.best_value,
                    "best_on_edge": r.grid.on_edge, "n_values": len(r.grid.values),
                },
                "proposed_dropped": r.dropped,
                "features": {
                    str(row.feature): {"mean_without": row.mean_without, "delta": row.delta,
                                       "delta_se": row.delta_se, "retained": bool(row.retained)}
                    for row in r.table().itertuples()
                },
            }
            for branch, r in result.ablation.items()
        })  # fmt: skip
        runs[EXPERIMENTS["ablation"]] = result.ablation_run_ids
        if result.ablation_table_path is not None:
            run.log_artifact(result.ablation_table_path, description="E-30 ablation table")
        for path in provenance.grid_paths.values():
            run.log_artifact(path, description="ablation reference grid (Ridge alpha)")
    if result.outcome_check is not None:
        check = result.outcome_check
        run.log_value("rc02", {"status": check.status, "proposed": check.proposed,
                               "committed": check.committed,
                               "differences": check.differences})  # fmt: skip
    if result.dev_checks:
        for name, cv in result.dev_checks.items():
            _record_cv(run, f"development_check.{name}", cv)
        run.log_value("development_check", {
            "used_for_selection": False, "effective_params": result.dev_check_params,
        })  # fmt: skip
        runs[EXPERIMENTS["cv_comparison"]] = result.dev_check_run_ids
    run.log_value("mlflow", {
        "tracking_uri": provenance.tracking_uri, "pipeline_run_id": result.pipeline_run_id,
        "runs": runs,
        "detail": "per-fold scores, parameters and standard tags are in these MLflow runs",
    })  # fmt: skip


def _record_cv(run: ResultRun, prefix: str, cv: CVResult) -> None:
    run.log_metrics({f"{prefix}.cv_mean": cv.mean, f"{prefix}.cv_se": cv.se,
                     **{f"{prefix}.{k}": v for k, v in cv.secondary.items()},
                     f"{prefix}.duration_seconds": cv.duration_seconds})  # fmt: skip


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


def _cv_table(results: dict[str, CVResult]) -> list[str]:
    lines = [(f"{'candidate':<16}{'cv_mean':>10}{'cv_se':>10}{'cv_mae':>12}{'cv_mape':>10}"
              f"{'cv_r2':>9}")]  # fmt: skip
    for name, cv in results.items():
        lines.append(
            f"{name:<16}{cv.mean:>10.4f}{cv.se:>10.4f}{cv.secondary['cv_mae']:>12,.0f}"
            f"{cv.secondary['cv_mape']:>9.2f}%{cv.secondary['cv_r2']:>9.3f}"
        )
    return lines


def _ablation_lines(result: BranchAblation) -> list[str]:
    head = f"{result.branch} branch, reference {result.reference}"
    if result.grid is not None:
        edge = " (ON GRID EDGE)" if result.grid.on_edge else ""
        head += f", {result.grid.param} = {result.grid.best_value:.6g} from the grid{edge}"
    lines = [head, f"  full feature set: mean {result.full.mean:.6f} (SE {result.full.se:.6f})",
             f"  {'feature':<14}{'mean_without':>14}{'delta':>12}{'delta_se':>11}  decision"]  # fmt: skip
    for row in result.table().itertuples():
        decision = "retain" if row.retained else "DROP"
        lines.append(f"  {row.feature:<14}{row.mean_without:>14.6f}{row.delta:>+12.6f}"
                     f"{row.delta_se:>11.6f}  {decision}")  # fmt: skip
    return lines


def report(result: TrainResult) -> str:
    """Human-readable summary printed by the CLI."""
    lines = [
        f"pipeline_run_id: {result.pipeline_run_id}",
        f"shared folds: {result.folds_path}",
        f"development rows: {result.dev_rows}; SD of log1p(SalePrice): {result.dev_log_price_sd:.4f}",
        "",
        "Baselines (MLflow experiment hpp-baselines)",
        *_cv_table(result.results),
    ]
    if result.ablation:
        lines += ["", "Feature ablation (hpp-ablation; IN-04: drop only if delta < 0)"]
        for branch in BRANCHES:
            lines += _ablation_lines(result.ablation[branch])
        lines.append(f"Ablation table (E-30): {result.ablation_table_path}")
        for branch in BRANCHES:
            dropped = result.ablation[branch].dropped
            lines.append(f"Proposed {branch}.dropped_engineered: {dropped}")
    if result.outcome_check is not None:
        lines += ["", result.outcome_check.message()]
    if result.dev_checks:
        lines += ["", ("Development checks (hpp-cv-comparison, stage=development_check; "
                       "not used for selection)"), *_cv_table(result.dev_checks)]  # fmt: skip
    if result.result_path is not None:
        lines += ["", f"Run record: {result.result_path}"]
    return "\n".join(lines)

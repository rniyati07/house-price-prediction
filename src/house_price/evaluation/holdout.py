"""Final holdout evaluation, baseline reference scoring, and the ``evaluate`` command up to
the temporal diagnostic (FR-033, FR-035, AC-038 to AC-043; DOC-03 §9.4, §12, §13; IN-12).

This is the **only** module that reads the holdout file (AC-030, DOC-03 §9.4). The real
evaluation runs once, in the M13 Release Run (DOC-05 RC-01); M9 builds and tests this
machinery in smoke mode only (DN-17).

Preconditions (DOC-03 §12.2), all checked before anything is read or fitted:

1. the selection record exists;
2. ``diagnostic_review.yaml`` is a completed human review of that record with all three
   gates ``pass`` (DN-19; the automatic smoke review is accepted in smoke mode only);
3. no ``final_holdout_evaluation`` run exists for that record (AC-038);
4. the git tree is clean (enforced in real mode; recorded, not enforced, in smoke mode).

Then: fit the selected configuration on the development set, predict the holdout once, log
``run_kind=final_holdout_evaluation``; score each baseline once (``baseline_reference``);
check the quality gates (``quality_gates``); run the temporal diagnostic on development rows
(``temporal_diagnostic``); then the production refit (M10, FR-034, FR-038, FR-039): the
selected configuration is refit on all in-scope rows (development + holdout, 2,925) and
written to ``models/staging/`` with ``metadata.json``, the exact round trip is checked on the
development set, and the run is logged as ``run_kind=production_refit``.

Smoke mode refits on the smoke development sample only (the holdout substitute stays
separate) and writes to ``artifacts/smoke/model/``; the refit run is in ``hpp-smoke``, which
is how ``freeze`` recognises and refuses a smoke artifact (DN-17, DOC-03 §16.2 step 3).

Candidate reference artifacts (additional project requirement, M10): after the production
refit, each of the four tuned candidates is refit once, on the same rows, with its tuned
parameters from the run's ``_best.json`` (checked against the selection record's
``pipeline_run_id``), and written to ``models/candidates/<name>/`` (smoke:
``artifacts/smoke/candidates/<name>/``) with ``artifact_role="candidate"`` metadata. They are
never scored on the holdout, never influence selection, and are never released.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    ProjectConfig,
    load_feature_config,
    load_models_config,
    load_project_config,
)
from house_price.data.errors import DataError
from house_price.data.load import load_raw, read_typed_csv, sha256_file
from house_price.data.profile import utc_now
from house_price.data.schema import select_model_input, validate_raw
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.evaluation import gates, temporal
from house_price.evaluation.metrics import all_metrics
from house_price.models import selection
from house_price.models.registry import baseline_candidates, registry
from house_price.persistence import artifact, metadata
from house_price.results import start_run
from house_price.tracking import Lineage, Tracker, default_tracking_uri, git_state

EVALUATION_DIR = Path("reports/evaluation")
SMOKE_DIR = selection.SMOKE_DIR  # every smoke output lives here (DN-17)
SMOKE_SPLIT_NAME = selection.SMOKE_SPLIT_NAME

Log = Callable[[str], None]


class EvaluationError(DataError):
    """Evaluation refuses to run (a precondition failed)."""


@dataclass(frozen=True)
class EvaluationPaths:
    selection_dir: Path
    evaluation_dir: Path
    smoke_split: Path | None
    model_dir: Path
    candidates_dir: Path
    tuning_dir: Path

    @classmethod
    def for_mode(cls, root: Path, smoke: bool) -> EvaluationPaths:
        if smoke:
            base = root / SMOKE_DIR
            return cls(base / "selection", base / "evaluation", base / SMOKE_SPLIT_NAME,
                       base / "model", base / "candidates", base / "tuning")  # fmt: skip
        return cls(root / selection.SELECTION_DIR, root / EVALUATION_DIR, None,
                   root / artifact.STAGING_DIR, root / artifact.CANDIDATES_DIR,
                   root / "artifacts/tuning")  # fmt: skip


@dataclass
class Preconditions:
    checks: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def prior_final_evaluations(tracker: Tracker, record_id: str) -> list[str]:
    """``final_holdout_evaluation`` runs already logged for this selection record."""
    experiment = tracker.client.get_experiment_by_name(tracker.experiment_name("evaluation"))
    if experiment is None:
        return []
    runs = tracker.client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            "tags.run_kind = 'final_holdout_evaluation' and "
            f"tags.selection_record_id = '{record_id}'"
        ),
    )
    return [run.info.run_id for run in runs]


def check_preconditions(
    record: dict[str, Any], review_path: Path, tracker: Tracker, *, smoke: bool, dirty: bool
) -> Preconditions:
    """DOC-03 §12.2 preconditions; every failure is collected (none is skipped)."""
    pre = Preconditions()
    record_id = record["selection_record_id"]
    review = selection.review_problems(review_path, record_id, smoke=smoke)
    pre.checks["review_passes"] = not review
    pre.problems += review
    prior = prior_final_evaluations(tracker, record_id)
    pre.checks["prior_final_evaluations"] = prior
    if prior:
        pre.problems.append(
            f"a final holdout evaluation already exists for selection record "
            f"{record_id} (runs {prior}); the holdout is evaluated once"
        )
    pre.checks["git_dirty"] = dirty
    pre.checks["clean_tree_enforced"] = not smoke
    if dirty and not smoke:
        pre.problems.append("the git working tree is not clean")
    return pre


def load_real_holdout(config: ProjectConfig, in_scope: pd.DataFrame, scope: Any) -> pd.DataFrame:
    """Verify the persisted split (holdout hash included) and read the holdout rows."""
    split = create_or_load_split(in_scope, scope, config)  # verifies the holdout hash
    cast = read_typed_csv(
        config.holdout_path, config.schema, config.data.missing_tokens, source_headers=False
    )
    holdout = validate_raw(cast.frame, config.schema, cast_failures=cast.failures)
    if [int(v) for v in holdout[config.schema.id_column]] != split.manifest.holdout_ids:
        raise EvaluationError("holdout ids differ from the manifest")
    return holdout


def smoke_frames(
    dev: pd.DataFrame, path: Path, id_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The smoke development sample and its holdout substitute (disjoint development rows)."""
    if not path.is_file():
        raise EvaluationError(f"smoke split not found at {path}; run train --smoke first")
    split = json.loads(path.read_text(encoding="utf-8"))
    by_id = dev.set_index(id_column, drop=False)
    sample, substitute = split["sample_ids"], split["holdout_substitute_ids"]
    if set(sample) & set(substitute):
        raise EvaluationError("smoke sample and holdout substitute overlap")
    return by_id.loc[sample].reset_index(drop=True), by_id.loc[substitute].reset_index(drop=True)


def _fit_predict(model: Any, train: pd.DataFrame, test: pd.DataFrame, schema: Any) -> np.ndarray:
    model.fit(select_model_input(train, schema), train[schema.target])
    return np.asarray(model.predict(select_model_input(test, schema)), dtype="float64")


def run_evaluate(
    config_dir: Path | None = None,
    root: Path | None = None,
    tracking_uri: str | None = None,
    *,
    smoke: bool = False,
    argv: Sequence[str] | None = None,
    log: Log | None = None,
) -> dict[str, Any]:
    """``house-price evaluate [--smoke]``: preconditions, final evaluation, baseline
    reference, quality gates, temporal diagnostic. Returns the evaluation summary."""
    log = log or (lambda _: None)
    root = (root or Path.cwd()).resolve()
    config_dir = config_dir or root / DEFAULT_CONFIG_DIR
    paths = EvaluationPaths.for_mode(root, smoke)
    with start_run("M10", root=root, entry_point="house-price evaluate", argv=argv) as run_record:
        config = load_project_config(config_dir, root)
        features, models = load_feature_config(config_dir), load_models_config(config_dir)
        schema, seed = config.schema, config.validation.seed
        record = selection.read_record(paths.selection_dir / selection.RECORD_NAME)
        record_id = record["selection_record_id"]
        commit, dirty = git_state()
        lineage = Lineage(
            pipeline_run_id=record["pipeline_run_id"],
            git_commit=commit,
            git_dirty=dirty,
            data_sha256=config.data.raw_sha256,
            split_manifest_sha256=sha256_file(config.manifest_path),
            config_hash=record["provenance"].get("config_hash")
            or record["provenance"]["m8_lineage"]["config_hash"],
            seed=seed,
        )
        tracker = Tracker(
            tracking_uri or default_tracking_uri(root),
            lineage,
            experiment_override="smoke" if smoke else None,
        )
        pre = check_preconditions(
            record, paths.selection_dir / selection.REVIEW_NAME, tracker, smoke=smoke, dirty=dirty
        )
        run_record.log_params({"smoke": smoke, "selection_record_id": record_id})
        run_record.log_value("preconditions", {"checks": pre.checks, "problems": pre.problems})
        if pre.problems:
            raise EvaluationError("evaluate refuses to run: " + "; ".join(pre.problems))

        raw = load_raw(config)
        in_scope, scope = apply_scope_rule(raw, config.data.scope, schema.id_column)
        if smoke:
            full_dev = create_or_load_split(in_scope, scope, config, verify_holdout_file=False).dev
            dev, holdout = smoke_frames(full_dev, paths.smoke_split, schema.id_column)  # type: ignore[arg-type]
        else:
            dev = create_or_load_split(in_scope, scope, config, verify_holdout_file=False).dev
            holdout = load_real_holdout(config, in_scope, scope)
        log(
            f"final evaluation of {record['selected']['name']} on "
            f"{'the smoke holdout substitute' if smoke else 'the holdout'} ({len(holdout)} rows)"
        )
        tags = {"selection_record_id": record_id, "smoke": str(smoke).lower()}
        build = lambda: selection.build_selected(record, models, seed, features, schema)
        actual = holdout[schema.target].to_numpy(dtype="float64")
        predicted = _fit_predict(build(), dev, holdout, schema)
        final = all_metrics(actual, predicted)
        with tracker.run(
            "evaluation",
            stage="evaluation",
            candidate=record["selected"]["name"],
            run_name="final_holdout_evaluation",
            run_kind="final_holdout_evaluation",
            extra_tags=tags,
        ) as run:
            run.log_params({"n_train": len(dev), "n_holdout": len(holdout), "smoke": smoke})
            run.log_metrics(final)
        final_run_id = run.run_id

        reference: dict[str, dict[str, float]] = {}
        for baseline in baseline_candidates(models):
            scores = _fit_predict(baseline.build(features, schema), dev, holdout, schema)
            reference[baseline.name] = all_metrics(actual, scores)
        with tracker.run(
            "evaluation",
            stage="evaluation",
            candidate="baselines",
            run_name="baseline_reference",
            run_kind="baseline_reference",
            extra_tags=tags,
        ) as run:
            run.log_params({"note": "IN-12: computed once, drives no decision"})
            run.log_metrics({f"{b}.{k}": v for b, m in reference.items() for k, v in m.items()})
        baseline_run_id = run.run_id

        results = gates.check_quality_gates(final, reference, record)
        gate_path = paths.evaluation_dir / gates.GATES_PATH.name
        gate_report = gates.write_gates(results, gate_path, smoke=smoke, record_id=record_id)
        with tracker.run(
            "evaluation",
            stage="evaluation",
            candidate=record["selected"]["name"],
            run_name="quality_gates",
            run_kind="quality_gates",
            extra_tags=tags,
        ) as run:
            run.log_metrics({f"{r.rule}.passed": float(r.passed) for r in results})
            run.log_artifact(gate_path)
        gates_run_id = run.run_id

        tresult = temporal.run_temporal(build, dev, schema)
        temporal_path = paths.evaluation_dir / "temporal_diagnostic.json"
        temporal_path.write_text(json.dumps(tresult.as_record(), indent=2) + "\n", encoding="utf-8")
        with tracker.run(
            "evaluation",
            stage="evaluation",
            candidate=record["selected"]["name"],
            run_name="temporal_diagnostic",
            run_kind="temporal_diagnostic",
            extra_tags={**tags, "label": temporal.LABEL},
        ) as run:
            run.log_params(
                {"n_train": tresult.n_train, "n_test": tresult.n_test, "label": temporal.LABEL}
            )
            run.log_metrics(
                {**tresult.metrics, "mean_signed_log_error": tresult.mean_signed_log_error}
            )
        temporal_run_id = run.run_id

        # ---------------------------------------------------- production refit (M10)
        refit_rows = dev if smoke else pd.concat([dev, holdout], ignore_index=True)
        log(f"production refit of {record['selected']['name']} on {len(refit_rows)} rows "
            f"({'smoke development sample' if smoke else 'all in-scope rows'})")  # fmt: skip
        with tracker.run("evaluation", stage="evaluation", candidate=record["selected"]["name"],
                         run_name="production_refit", run_kind="production_refit",
                         extra_tags=tags) as run:  # fmt: skip
            model = build()
            model.fit(select_model_input(refit_rows, schema), refit_rows[schema.target])
            model_sha = artifact.save_model(model, paths.model_dir)
            trip = artifact.round_trip(model, paths.model_dir / artifact.MODEL_FILE,
                                       select_model_input(dev, schema))  # fmt: skip
            meta = _staging_metadata(
                model, record, config, features, models, lineage, tresult.as_record(),
                final, reference, gate_report, model_sha, len(refit_rows),
                {"refit": run.run_id, "final_holdout_evaluation": final_run_id},
            )  # fmt: skip
            meta_path = artifact.write_metadata(meta, paths.model_dir)
            run.log_params({"training_rows": len(refit_rows), "model_sha256": model_sha,
                            "round_trip_passed": trip.passed})  # fmt: skip
            run.log_metrics({"round_trip_max_abs_difference": trip.max_abs_difference})
            run.log_artifact(meta_path)
            if not trip.passed:
                raise EvaluationError(
                    f"round trip failed: max |difference| {trip.max_abs_difference}"
                )
            # ------------------------- candidate reference refits (additional requirement)
            candidates = _refit_candidates(
                paths, record, config, features, models, meta, refit_rows, dev, schema, log
            )
            for name, info in candidates.items():
                run.log_artifact(Path(info["metadata"]), artifact_path=f"candidates/{name}")
        refit_run_id = run.run_id
        summary = {
            "selection_record_id": record_id,
            "smoke": smoke,
            "selected": record["selected"]["name"],
            "final_holdout": final,
            "baseline_reference": reference,
            "quality_gates": gate_report,
            "temporal_diagnostic": tresult.as_record(),
            "mlflow_run_ids": {
                "final_holdout_evaluation": final_run_id,
                "baseline_reference": baseline_run_id,
                "quality_gates": gates_run_id,
                "temporal_diagnostic": temporal_run_id,
                "production_refit": refit_run_id,
            },
            "experiment": tracker.experiment_name("evaluation"),
            "production_refit": {
                "model_dir": str(paths.model_dir),
                "model_sha256": model_sha,
                "training_rows": len(refit_rows),
                "round_trip": trip.as_record(),
                "is_release": False,
                "model_version": meta.model_version,
            },
            "candidate_refits": candidates,
        }
        summary_path = paths.evaluation_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        run_record.log_value("evaluation", summary)
        for path in (gate_path, temporal_path, summary_path, meta_path):
            run_record.log_artifact(path)
        return summary


TUNED = ("ridge", "lasso", "random_forest", "lightgbm")


def _refit_candidates(
    paths: EvaluationPaths,
    record: dict[str, Any],
    config: ProjectConfig,
    features: Any,
    models: Any,
    production: metadata.ArtifactMetadata,
    rows: pd.DataFrame,
    dev: pd.DataFrame,
    schema: Any,
    log: Log,
) -> dict[str, dict[str, Any]]:
    """Post-selection reference refits of the four tuned candidates (never evaluated on the
    holdout, never released). Provenance is shared with the production refit; everything
    that describes a model describes the candidate itself."""
    candidates = registry(models, config.validation.seed)
    folds = {c["name"]: c for c in record["candidates"]}
    X_rows, X_dev = select_model_input(rows, schema), select_model_input(dev, schema)
    results: dict[str, dict[str, Any]] = {}
    for name in TUNED:
        best_path = paths.tuning_dir / f"{name}_best.json"
        if not best_path.is_file():
            raise EvaluationError(f"tuned configuration not found: {best_path}")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        if best["lineage"]["pipeline_run_id"] != record["pipeline_run_id"]:
            raise EvaluationError(f"{best_path} is from pipeline run "
                                  f"{best['lineage']['pipeline_run_id']}, not the selection's "
                                  f"{record['pipeline_run_id']}")  # fmt: skip
        log(f"candidate reference refit: {name} on {len(rows)} rows")
        model = candidates[name].build(features, schema, best["best_params"])
        model.fit(X_rows, rows[schema.target])
        directory = paths.candidates_dir / name
        sha = artifact.save_model(model, directory)
        trip = artifact.round_trip(model, directory / artifact.MODEL_FILE, X_dev)
        if not trip.passed:
            raise EvaluationError(f"{name}: round trip failed ({trip.max_abs_difference})")
        scores = folds[name]
        meta = metadata.ArtifactMetadata.model_validate({
            **production.model_dump(),
            "artifact_role": "candidate", "model_sha256": sha, "created_at": utc_now(),
            "selected_candidate": name,
            "hyperparameters": {"tuned_params": best["best_params"],
                                "estimator_params": best["estimator_params"]},
            "feature_sets": metadata.feature_sets({name: candidates[name].branch}, features),
            "transformed_feature_names": metadata.transformed_feature_names(model, name),
            "cv": {"mean": scores["mean"], "se": scores["se"],
                   "n_folds": float(len(scores["fold_scores"]))},
            "holdout": None, "baseline_reference": None, "temporal_diagnostic": None,
            "quality_gates": None,
            "mlflow": {**production.mlflow.model_dump(), "final_holdout_evaluation": None},
        })  # fmt: skip
        empty = meta.empty_fields()
        if empty:
            raise EvaluationError(f"{name}: incomplete candidate metadata {empty}")
        meta_path = artifact.write_metadata(meta, directory)
        results[name] = {"dir": str(directory), "model_sha256": sha, "training_rows": len(rows),
                         "round_trip": trip.as_record(), "metadata": str(meta_path)}  # fmt: skip
    return results


def _staging_metadata(
    model: Any,
    record: dict[str, Any],
    config: ProjectConfig,
    features: Any,
    models: Any,
    lineage: Lineage,
    temporal_record: dict[str, Any],
    final: dict[str, float],
    reference: dict[str, dict[str, float]],
    gate_report: dict[str, Any],
    model_sha: str,
    training_rows: int,
    runs: dict[str, str],
) -> metadata.ArtifactMetadata:
    """DOC-03 §15.3 metadata of the staging artifact (``model_version`` "unreleased",
    ``is_release`` false, ``mlflow.release`` null until ``freeze``)."""
    import platform

    selected = record["selected"]
    name, hp = selected["name"], selected["hyperparameters"]
    candidates = registry(models, config.validation.seed)
    components = ({hp["linear"]: candidates[hp["linear"]].branch, hp["tree"]: candidates[hp["tree"]].branch}
                  if name == selection.BLEND_NAME else {name: candidates[name].branch})  # fmt: skip
    folds = next(c for c in record["candidates"] if c["name"] == name)["fold_scores"]
    columns = metadata.input_schema(config.schema)
    return metadata.ArtifactMetadata(
        model_version=metadata.UNRELEASED, is_release=False, model_sha256=model_sha,
        created_at=utc_now(), git_commit=lineage.git_commit, git_dirty=lineage.git_dirty,
        data_sha256=lineage.data_sha256, split_manifest_sha256=lineage.split_manifest_sha256,
        config_hash=lineage.config_hash, pipeline_run_id=record["pipeline_run_id"],
        selection_record_id=record["selection_record_id"],
        mlflow=metadata.MlflowLinks(**runs, release=None),
        python_version=platform.python_version(), library_versions=artifact.library_versions(),
        selected_candidate=name, hyperparameters=hp,
        feature_sets=metadata.feature_sets(components, features),
        transformed_feature_names=metadata.transformed_feature_names(model, name),
        target_transform={"func": model.func.__name__, "inverse_func": model.inverse_func.__name__},
        training_rows=training_rows,
        cv={"mean": selected["mean"], "se": selected["se"], "n_folds": float(len(folds))},
        holdout=final, baseline_reference={b: m["log_rmse"] for b, m in reference.items()},
        temporal_diagnostic=temporal_record, quality_gates=gate_report["rules"],
        input_schema=columns, schema_hash=metadata.schema_hash(columns),
        scope_rule=metadata.ScopeRule(column=config.data.scope.column,
                                      max_in_domain=config.data.scope.threshold),
        seed=config.validation.seed,
        reproducibility_tolerance=config.validation.reproducibility_tolerance,
    )  # fmt: skip

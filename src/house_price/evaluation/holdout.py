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
(``temporal_diagnostic``). Refitting on all 2,925 rows and the artifact are M10.
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
from house_price.data.schema import select_model_input, validate_raw
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.evaluation import gates, temporal
from house_price.evaluation.metrics import all_metrics
from house_price.models import selection
from house_price.models.registry import baseline_candidates
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

    @classmethod
    def for_mode(cls, root: Path, smoke: bool) -> EvaluationPaths:
        if smoke:
            base = root / SMOKE_DIR
            return cls(base / "selection", base / "evaluation", base / SMOKE_SPLIT_NAME)
        return cls(root / selection.SELECTION_DIR, root / EVALUATION_DIR, None)


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
    with start_run("M9", root=root, entry_point="house-price evaluate", argv=argv) as run_record:
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
                "temporal_diagnostic": run.run_id,
            },
            "experiment": tracker.experiment_name("evaluation"),
            "not_in_m9": "refit on all rows and the model artifact are M10",
        }
        summary_path = paths.evaluation_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        run_record.log_value("evaluation", summary)
        for path in (gate_path, temporal_path, summary_path):
            run_record.log_artifact(path)
        return summary

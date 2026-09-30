"""Model selection (ADR-11, FR-030, FR-031, DOC-03 §11.3 to §11.7, DN-08, DN-19).

**Rule.** Among the eligible candidates (the four tuned models and the blend; the baselines
are never eligible, DOC-03 §8.1):

1. Blend admission (§11.5): the blend is admissible only if
   ``mean_blend < m_single - se_single`` for the best single eligible model; otherwise it
   is excluded before step 2.
2. One-standard-error rule (§11.4): the best candidate of the remaining pool has mean
   ``m*`` and SE ``se*``; the threshold is ``t = m* + se*``; the admissible set is every
   candidate with mean <= ``t``.
3. Choose the admissible candidate in the lowest simplicity tier (1 Ridge/Lasso < 2 Random
   Forest < 3 LightGBM < 4 blend); within a tier the lower mean; on an exact tie Ridge
   (DN-08).
4. If the blend was admitted in step 1, it replaces the result of step 3 (§11.5).

The primary metric is the mean CV log-RMSE; secondary metrics never enter the decision.
Nothing here reads the holdout.

**Review (DN-19).** ``check_review`` accepts only a completed human review
(``status: final``, a named reviewer, a date) for the current ``selection_record_id`` with
all three gates ``pass`` and stated reasoning. A generated draft (``status: draft``,
``reviewer: null``) never passes. The automatic smoke review is accepted only in smoke mode.

**Rehearsal (M9).** ``python -m house_price.models.selection --from-m8`` builds the
selection record and the diagnostic report from the persisted M8 outputs (comparison
table, OOF predictions, ``_best.json``) without rerunning any tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from house_price.data.profile import utc_now

BLEND_NAME = "blend"
SELECTION_DIR = Path("reports/selection")
SMOKE_DIR = Path("artifacts/smoke")  # every smoke output lives here (DN-17)
SMOKE_SPLIT_NAME = "smoke_split.json"
RECORD_NAME = "selection_record.json"
REVIEW_NAME = "diagnostic_review.yaml"
REVIEW_GATES = ("residual_pattern", "price_decile_bias", "neighborhood_error")
RULES = {
    "primary_metric": "mean CV log-RMSE (ADR-12)",
    "standard_error": "std(fold scores, ddof=1) / sqrt(n folds) (DN-04)",
    "one_standard_error": "DOC-03 §11.4 (ADR-11)",
    "tiers": "1 ridge/lasso < 2 random_forest < 3 lightgbm < 4 blend (DOC-03 §8.1, DN-08)",
    "tie_break": "within a tier the lower mean; exact tie -> ridge (DN-08)",
    "blend_admission": "mean_blend < m_single - se_single, else excluded (DOC-03 §11.5)",
}


class SelectionError(ValueError):
    """The candidate table cannot be selected from."""


class ReviewError(ValueError):
    """The diagnostic review does not allow evaluation (DN-19)."""


@dataclass(frozen=True)
class CandidateScore:
    """One candidate's final CV result (DOC-03 §11.7 ``candidates``)."""

    name: str
    tier: int | None  # None for the baselines (never eligible)
    mean: float
    se: float
    fold_scores: list[float]
    secondary: dict[str, float]

    @property
    def eligible(self) -> bool:
        return self.tier is not None

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "eligible": self.eligible,
            "mean": self.mean,
            "se": self.se,
            "fold_scores": self.fold_scores,
            "secondary": self.secondary,
        }


@dataclass(frozen=True)
class SelectionResult:
    best_single: CandidateScore
    blend: CandidateScore | None
    blend_bar: float  # m_single - se_single
    blend_admitted: bool
    pool: list[CandidateScore]  # the candidates the 1-SE rule considers
    best: CandidateScore  # m*, se*
    threshold: float  # t = m* + se*
    admissible: list[CandidateScore]
    one_se_choice: CandidateScore  # the §11.4 result
    selected: CandidateScore

    @property
    def reason(self) -> str:
        if self.blend_admitted:
            return (
                f"blend admitted (§11.5): {self.selected.mean:.6f} < {self.blend_bar:.6f} = "
                f"{self.best_single.name} mean {self.best_single.mean:.6f} - SE "
                f"{self.best_single.se:.6f}; it replaces the §11.4 result "
                f"({self.one_se_choice.name})"
            )
        excluded = (
            ""
            if self.blend is None
            else (f"blend excluded (§11.5: {self.blend.mean:.6f} >= {self.blend_bar:.6f}); ")
        )
        return (
            f"{excluded}lowest tier ({self.selected.tier}) among the candidates within 1 SE of "
            f"the best ({self.best.name} {self.best.mean:.6f} + {self.best.se:.6f} = "
            f"{self.threshold:.6f}): {sorted(c.name for c in self.admissible)}"
        )


def _by_mean(c: CandidateScore) -> tuple[float, int, int, str]:
    return (c.mean, c.tier or 0, 0 if c.name == "ridge" else 1, c.name)


def _by_tier(c: CandidateScore) -> tuple[int, float, int, str]:
    return (c.tier or 0, c.mean, 0 if c.name == "ridge" else 1, c.name)


def select(scores: Sequence[CandidateScore]) -> SelectionResult:
    """Apply DOC-03 §11.4 and §11.5 to the final CV results."""
    eligible = [c for c in scores if c.eligible]
    singles = [c for c in eligible if c.name != BLEND_NAME]
    if not singles:
        raise SelectionError("no eligible single model in the candidate table")
    blend = next((c for c in eligible if c.name == BLEND_NAME), None)
    best_single = min(singles, key=_by_mean)
    bar = best_single.mean - best_single.se
    admitted = blend is not None and blend.mean < bar
    pool = [*singles, *([blend] if admitted and blend is not None else [])]
    best = min(pool, key=_by_mean)
    threshold = best.mean + best.se
    admissible = [c for c in pool if c.mean <= threshold]
    choice = min(admissible, key=_by_tier)
    selected = blend if admitted and blend is not None else choice
    return SelectionResult(
        best_single, blend, bar, admitted, pool, best, threshold, admissible, choice, selected
    )


def selection_record(
    result: SelectionResult,
    scores: Sequence[CandidateScore],
    *,
    pipeline_run_id: str,
    hyperparameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The selection record (DOC-03 §11.7)."""
    selected = result.selected
    baselines = [c for c in scores if not c.eligible]
    return {
        "selection_record_id": str(uuid.uuid4()),
        "pipeline_run_id": pipeline_run_id,
        "created_at": utc_now(),
        "rules": RULES,
        "candidates": [c.as_record() for c in scores],
        "eligible": [c.name for c in scores if c.eligible],
        "best_single": {
            "name": result.best_single.name,
            "mean": result.best_single.mean,
            "se": result.best_single.se,
        },
        "blend_admitted": {
            "admitted": result.blend_admitted,
            "blend_mean": None if result.blend is None else result.blend.mean,
            "bar": result.blend_bar,
            "rule": RULES["blend_admission"],
        },
        "threshold": {
            "value": result.threshold,
            "best": result.best.name,
            "best_mean": result.best.mean,
            "best_se": result.best.se,
        },
        "admissible": [c.name for c in result.admissible],
        "one_se_choice": result.one_se_choice.name,
        "selected": {
            "name": selected.name,
            "tier": selected.tier,
            "mean": selected.mean,
            "se": selected.se,
            "hyperparameters": dict(hyperparameters),
            "reason": result.reason,
        },
        "baseline_margins": {
            b.name: {
                "selected_minus_baseline": selected.mean - b.mean,
                "baseline_mean": b.mean,
                "baseline_se": b.se,
            }
            for b in baselines
        },
        "provenance": dict(provenance),
    }


def write_json(payload: Mapping[str, Any], path: Path) -> Path:
    from house_price.results import jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path


def read_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"selection record not found: {path}")
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


# --------------------------------------------------------------------- review (DN-19)


def draft_review(record_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    """A generated draft: evidence pointers only, no judgement, never passes."""
    return {
        "selection_record_id": record_id,
        "status": "draft",
        "reviewer": None,
        "reviewed_at": None,
        "note": (
            "Generated draft. A human reviewer must read the diagnostic report, record a "
            "pass/fail judgement with reasoning for each gate, set status: final, and sign "
            "(reviewer, reviewed_at). Evaluation refuses a draft (DN-19)."
        ),
        "evidence": dict(summary),
        "gates": {gate: {"result": None, "reasoning": None} for gate in REVIEW_GATES},
    }


def smoke_review(record_id: str) -> dict[str, Any]:
    """DN-17: the automatic review of a smoke run (accepted only in smoke mode)."""
    return {
        "selection_record_id": record_id,
        "status": "final",
        "smoke": True,
        "reviewer": "smoke-auto (DN-17)",
        "reviewed_at": utc_now(),
        "gates": {
            gate: {
                "result": "pass",
                "reasoning": "auto-generated smoke review; not a judgement (DN-17)",
            }
            for gate in REVIEW_GATES
        },
    }


def write_review(review: Mapping[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(review), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def review_problems(path: Path, record_id: str, *, smoke: bool) -> list[str]:
    """Every reason the review does not allow evaluation (empty list = it does)."""
    if not path.is_file():
        return [f"diagnostic review not found: {path}"]
    try:
        review = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [f"diagnostic review is not valid YAML: {exc}"]
    problems = []
    if review.get("selection_record_id") != record_id:
        problems.append(
            f"review is for selection record {review.get('selection_record_id')!r}, "
            f"not the current {record_id!r}"
        )
    if review.get("status") != "final":
        problems.append(f"review status is {review.get('status')!r}, not 'final'")
    if not review.get("reviewer"):
        problems.append("review is not signed (reviewer is empty)")
    if not review.get("reviewed_at"):
        problems.append("review has no reviewed_at date")
    if review.get("smoke") and not smoke:
        problems.append("an automatic smoke review cannot approve a real evaluation")
    gates = review.get("gates") or {}
    for gate in REVIEW_GATES:
        entry = gates.get(gate) or {}
        if entry.get("result") != "pass":
            problems.append(f"gate {gate}: result is {entry.get('result')!r}, not 'pass'")
        if not str(entry.get("reasoning") or "").strip():
            problems.append(f"gate {gate}: no reasoning recorded")
    return problems


def check_review(path: Path, record_id: str, *, smoke: bool = False) -> dict[str, Any]:
    problems = review_problems(path, record_id, smoke=smoke)
    if problems:
        raise ReviewError("diagnostic review does not allow evaluation: " + "; ".join(problems))
    review: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return review


# ---------------------------------------------------------------- selected model


def selected_overrides(record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """The selected candidate's name and its tuned hyperparameters."""
    selected = record["selected"]
    return selected["name"], dict(selected["hyperparameters"])


def build_selected(
    record: Mapping[str, Any], models: Any, seed: int, features: Any, schema: Any
) -> Any:
    """A fresh, unfitted pipeline of the selected configuration (registry + record)."""
    from house_price.models.registry import blend_candidate, registry

    name, hp = selected_overrides(record)
    candidates = registry(models, seed)
    if name == BLEND_NAME:
        blend = blend_candidate(
            candidates[hp["linear"]],
            hp["linear_params"],
            candidates[hp["tree"]],
            hp["tree_params"],
            models.blend.tier,
        )
        return blend.build(features, schema)
    if name not in candidates or not candidates[name].eligible:
        raise SelectionError(f"selected candidate {name!r} is not an eligible registry entry")
    return candidates[name].build(features, schema, hp["tuned_params"])


# ------------------------------------------------------------------ M9 rehearsal


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def scores_from_comparison(table: Any) -> list[CandidateScore]:
    """Candidate scores from ``comparison.csv`` (M8 final CV comparison)."""
    import math

    folds = sorted(c for c in table.columns if c.startswith("fold_"))
    scores = []
    for row in table.to_dict("records"):
        tier = row["tier"]
        scores.append(
            CandidateScore(
                name=str(row["candidate"]),
                tier=None
                if tier is None or (isinstance(tier, float) and math.isnan(tier))
                else int(tier),
                mean=float(row["cv_mean"]),
                se=float(row["cv_se"]),
                fold_scores=[float(row[f]) for f in folds],
                secondary={k: float(row[k]) for k in ("cv_mae", "cv_mape", "cv_r2")},
            )
        )
    return scores


def hyperparameters_for(
    name: str, best: Mapping[str, Mapping[str, Any]], blend_linear: str | None
) -> dict[str, Any]:
    """Final hyperparameters of ``name``: the tuned values (overrides) and the full estimator
    settings; for the blend, its two components (DN-07)."""
    if name == BLEND_NAME:
        if blend_linear is None:
            raise SelectionError("the blend's linear component is not recorded")
        return {
            "linear": blend_linear,
            "tree": "lightgbm",
            "weights": [0.5, 0.5],
            "linear_params": best[blend_linear]["best_params"],
            "tree_params": best["lightgbm"]["best_params"],
            "linear_estimator_params": best[blend_linear]["estimator_params"],
            "tree_estimator_params": best["lightgbm"]["estimator_params"],
        }
    return {
        "tuned_params": best[name]["best_params"],
        "estimator_params": best[name]["estimator_params"],
    }


def rehearse_from_m8(
    root: Path, config_dir: Path, selection_dir: Path, log: Any = print
) -> dict[str, Any]:
    """Select from the persisted M8 outputs and write the record, diagnostics and draft
    review. No tuning is rerun and the holdout file is never opened."""
    import pandas as pd

    from house_price.config import load_feature_config, load_project_config
    from house_price.data.load import load_raw
    from house_price.data.scope import apply_scope_rule
    from house_price.data.split import create_or_load_split
    from house_price.evaluation import diagnostics
    from house_price.results import read_index, read_result

    comparison_path = root / "artifacts/cv_comparison/comparison.csv"
    table = pd.read_csv(comparison_path)
    best_paths = {
        n: root / f"artifacts/tuning/{n}_best.json"
        for n in ("ridge", "lasso", "random_forest", "lightgbm")
    }
    best = {n: json.loads(p.read_text(encoding="utf-8")) for n, p in best_paths.items()}
    pipeline_ids = {b["lineage"]["pipeline_run_id"] for b in best.values()}
    if len(pipeline_ids) != 1:
        raise SelectionError(f"the _best.json files come from different runs: {pipeline_ids}")
    (pipeline_run_id,) = pipeline_ids
    index = [line for line in read_index(root / "results") if line["run_id"] == pipeline_run_id]
    if len(index) != 1 or index[0]["milestone"] != "M8" or index[0]["status"] != "succeeded":
        raise SelectionError(f"no single succeeded M8 run record for {pipeline_run_id}")
    record_path = root / "results" / index[0]["result"]
    m8 = read_result(record_path)
    m8_runs = m8["findings"]["cv_comparison"]["mlflow_run_ids"]
    if dict(zip(table["candidate"], table["mlflow_run_id"], strict=True)) != m8_runs:
        raise SelectionError("comparison.csv does not match the M8 run record")
    for name, b in best.items():
        if (
            abs(
                b["best_cv_mean"]
                - float(table.set_index("candidate").loc[name, "tuning_best_cv_mean"])
            )
            > 1e-12  # CSV float round-trip; the files must describe the same run
        ):
            raise SelectionError(f"{name}_best.json does not match comparison.csv")

    scores = scores_from_comparison(table)
    result = select(scores)
    blend_linear = m8["findings"]["cv_comparison"]["blend_linear_component"]
    hyper = hyperparameters_for(result.selected.name, best, blend_linear)
    sources = [
        comparison_path,
        *best_paths.values(),
        record_path,
        root / f"artifacts/cv_comparison/{result.selected.name}_oof.csv",
    ]
    provenance = {
        "source": "persisted M8 outputs (python -m house_price.models.selection --from-m8); "
        "no tuning rerun",
        "m8_pipeline_run_id": pipeline_run_id,
        "m8_git": m8["git"],  # recorded exactly as M8 recorded it
        "m8_lineage": m8["lineage"],
        "m8_execution_budget": m8["findings"].get("execution_budget"),
        "blend_linear_component": blend_linear,
        "files": {_rel(p, root): _sha(p) for p in sources},
    }
    record = selection_record(
        result,
        scores,
        pipeline_run_id=pipeline_run_id,
        hyperparameters=hyper,
        provenance=provenance,
    )

    config = load_project_config(config_dir, root)
    load_feature_config(config_dir)  # the committed feature configuration must load
    raw = load_raw(config)
    in_scope, scope = apply_scope_rule(raw, config.data.scope, config.schema.id_column)
    dev = create_or_load_split(in_scope, scope, config, verify_holdout_file=False).dev
    oof = pd.read_csv(root / f"artifacts/cv_comparison/{result.selected.name}_oof.csv")
    report = diagnostics.write_report(
        result.selected.name,
        oof,
        dev,
        selection_dir,
        id_column=config.schema.id_column,
        target=config.schema.target,
    )
    summary = report.summary()
    record["diagnostic_report"] = {k: _rel(v, root) for k, v in report.paths.items()}
    write_json(record, selection_dir / RECORD_NAME)
    write_json(summary, selection_dir / diagnostics.SUMMARY)
    write_review(draft_review(record["selection_record_id"], summary), selection_dir / REVIEW_NAME)
    log(f"selected: {result.selected.name} ({result.reason})")
    return record


def log_selection(
    tracker: Any,
    record: Mapping[str, Any],
    selection_dir: Path,
    extra_tags: Mapping[str, str] | None = None,
) -> str:
    """One run in ``hpp-selection`` (DOC-03 §11.6, §11.7) with the record, the diagnostic
    report and the review file attached."""
    selected = record["selected"]
    tags = {"selection_record_id": record["selection_record_id"], **(extra_tags or {})}
    with tracker.run(
        "selection",
        stage="selection",
        candidate=selected["name"],
        run_name=f"selection_{selected['name']}",
        extra_tags=tags,
    ) as run:
        run.log_params(
            {
                "selected": selected["name"],
                "selected_tier": selected["tier"],
                "best_single": record["best_single"]["name"],
                "blend_admitted": record["blend_admitted"]["admitted"],
                "admissible": ",".join(record["admissible"]),
                "one_se_choice": record["one_se_choice"],
                "selection_record_id": record["selection_record_id"],
            }
        )
        run.log_metrics(
            {
                "selected_cv_mean": selected["mean"],
                "selected_cv_se": selected["se"],
                "threshold": record["threshold"]["value"],
                "blend_bar": record["blend_admitted"]["bar"],
                **{
                    f"margin_vs_{b}": m["selected_minus_baseline"]
                    for b, m in record["baseline_margins"].items()
                },
            }
        )
        for path in sorted(selection_dir.glob("*")):
            if path.is_file():
                run.log_artifact(path)
    return str(run.run_id)


def run_rehearsal(
    root: Path, config_dir: Path, argv: Sequence[str] | None = None
) -> dict[str, Any]:
    """The M9 selection rehearsal: record, diagnostics, draft review, an ``hpp-selection``
    run, and an M9 run record. The review stays a draft until a human signs it."""
    from house_price.config import load_project_config
    from house_price.data.load import sha256_file
    from house_price.results import start_run
    from house_price.tracking import Lineage, Tracker, default_tracking_uri, git_state

    selection_dir = root / SELECTION_DIR
    with start_run(
        "M9", root=root, entry_point="python -m house_price.models.selection", argv=argv
    ) as result_run:
        record = rehearse_from_m8(root, config_dir, selection_dir)
        config = load_project_config(config_dir, root)
        commit, dirty = git_state()
        m8 = record["provenance"]["m8_lineage"]
        lineage = Lineage(
            pipeline_run_id=record["pipeline_run_id"],
            git_commit=commit,
            git_dirty=dirty,
            data_sha256=config.data.raw_sha256,
            split_manifest_sha256=sha256_file(config.manifest_path),
            config_hash=m8["config_hash"],
            seed=config.validation.seed,
        )
        tracker = Tracker(default_tracking_uri(root), lineage)
        run_id = log_selection(tracker, record, selection_dir, {"source": "m8_artifacts"})
        result_run.log_lineage(
            **{k: v for k, v in m8.items()}, m8_git=record["provenance"]["m8_git"]
        )
        result_run.log_params(
            {
                "mode": "rehearsal from persisted M8 outputs",
                "m8_pipeline_run_id": record["pipeline_run_id"],
            }
        )
        result_run.log_value(
            "selection",
            {
                k: record[k]
                for k in (
                    "selection_record_id",
                    "best_single",
                    "blend_admitted",
                    "threshold",
                    "admissible",
                    "one_se_choice",
                    "selected",
                    "baseline_margins",
                )
            },
        )
        result_run.log_value(
            "mlflow",
            {"experiment": "hpp-selection", "run_id": run_id, "tracking_uri": tracker.tracking_uri},
        )
        result_run.log_value(
            "review",
            {"status": "draft", "reviewer": None, "note": "evaluation refuses until a human signs"},
        )
        for path in sorted(selection_dir.glob("*")):
            if path.is_file():
                result_run.log_artifact(path)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    from house_price.config import DEFAULT_CONFIG_DIR, ConfigError
    from house_price.data.errors import DataError

    parser = argparse.ArgumentParser(
        prog="python -m house_price.models.selection",
        description="M9 selection rehearsal from the persisted M8 outputs (no tuning rerun).",
    )
    parser.add_argument("--from-m8", action="store_true", required=True)
    parser.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    root = (args.root or Path.cwd()).resolve()
    try:
        run_rehearsal(root, args.config_dir or root / DEFAULT_CONFIG_DIR, argv=argv)
    except (ConfigError, DataError, SelectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

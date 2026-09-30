"""Per-branch feature ablation and the RC-02 outcome check (ADR-07, FR-015, DOC-03 §6.6,
DN-05, DOC-01 IN-04, DOC-05 RC-02).

**Procedure (DN-05).** Before any tuning, for each branch, with that branch's reference
model from ``models.yaml`` (linear: Ridge with ``alpha`` chosen by its grid on the full
feature set; tree: LightGBM with the fixed reference parameters):

1. the full-feature CV score on the shared folds (every engineered feature active);
2. for each of the 12 engineered features, the CV score with that feature moved to the
   branch's dropped group (one at a time).

The ordinal map and ``MSSubClass`` recoding are encodings, not added features, and are not
ablated.

**Decision (IN-04), per branch.** ``delta = mean_without - mean_with``. A feature is
retained unless its removal *lowers* the mean CV log-RMSE, i.e. it is dropped only when
``delta < 0``; a tie keeps it. ``se_*`` and ``delta_se`` (the paired standard error of the
15 fold differences, DN-04) are reported for context and never used to decide.

**RC-02.** The outcome is never written by code. ``check_outcome`` compares the newly
computed outcome with the one committed in ``features.yaml``: no committed outcome, or a
different one, stops ``train``.

Only the development set and the shared folds are used; the holdout is never read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from house_price.config import FeatureConfig, SchemaConfig
from house_price.evaluation.cv import CVResult, FoldPlan, run_cv, standard_error
from house_price.models.registry import Candidate
from house_price.models.tuning import GridResult, evaluate_grid
from house_price.pipelines.branches import BRANCHES, Branch

E30_PATH = Path("reports/eda/E-30_ablation.csv")  # DOC-02 E-30
TABLE_COLUMNS = [
    "branch", "feature", "reference_model", "reference_params", "mean_with", "se_with",
    "mean_without", "se_without", "delta", "delta_se", "retained",
]  # fmt: skip
FLOAT_FORMAT = "%.10g"  # the EDA tables' format (reports/eda)


def is_retained(mean_with: float, mean_without: float) -> bool:
    """IN-04: kept unless removing the feature lowers the mean CV log-RMSE."""
    return not mean_without < mean_with


def with_dropped(features: FeatureConfig, branch: Branch, dropped: list[str]) -> FeatureConfig:
    """``features`` with ``branch``'s ablation outcome set to ``dropped`` (an ablation
    variant; the committed outcome in ``features.yaml`` is ignored and never changed)."""
    groups = getattr(features, branch).model_copy(update={"dropped_engineered": list(dropped)})
    return features.model_copy(update={branch: groups})


@dataclass(frozen=True)
class BranchAblation:
    """One branch's ablation: the full-feature result and one result per removed feature."""

    branch: Branch
    reference: str
    reference_params: dict[str, object]
    full: CVResult
    without: dict[str, CVResult]
    grid: GridResult | None = field(default=None, repr=False)

    def table(self) -> pd.DataFrame:
        rows = []
        for feature, result in self.without.items():
            differences = np.subtract(result.fold_scores, self.full.fold_scores)
            rows.append({
                "branch": self.branch, "feature": feature, "reference_model": self.reference,
                "reference_params": json.dumps(self.reference_params, sort_keys=True),
                "mean_with": self.full.mean, "se_with": self.full.se,
                "mean_without": result.mean, "se_without": result.se,
                "delta": result.mean - self.full.mean,
                "delta_se": standard_error(differences.tolist()),
                "retained": is_retained(self.full.mean, result.mean),
            })  # fmt: skip
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)

    @property
    def dropped(self) -> list[str]:
        return [f for f, r in self.without.items() if not is_retained(self.full.mean, r.mean)]


def run_branch_ablation(
    branch: Branch,
    reference: Candidate,
    dev: pd.DataFrame,
    plan: FoldPlan,
    features: FeatureConfig,
    schema: SchemaConfig,
) -> BranchAblation:
    """DN-05 for one branch with its reference candidate.

    If the reference has a grid (Ridge), the grid is evaluated on the full feature set, its
    best value becomes the reference parameter, and that grid point is the full-feature
    result (same pipeline, same folds). Otherwise the reference configuration is used.
    """
    if reference.branch != branch:
        raise ValueError(f"{reference.name} is a {reference.branch} model, not {branch}")
    full_features = with_dropped(features, branch, [])
    grid: GridResult | None = None
    overrides: dict[str, object] = {}
    if reference.grid is not None:
        grid = evaluate_grid(reference, dev, plan, full_features, schema)
        overrides = {grid.param: grid.best_value}
        full = grid.best
    else:
        full = run_cv(reference.build(full_features, schema), dev, plan, schema)
    without = {
        feature: run_cv(
            reference.build(with_dropped(features, branch, [feature]), schema, overrides),
            dev, plan, schema,
        )
        for feature in features.engineered
    }  # fmt: skip
    return BranchAblation(
        branch=branch, reference=reference.name,
        reference_params={**reference.params, **overrides}, full=full, without=without,
        grid=grid,
    )  # fmt: skip


def ablation_table(results: Mapping[str, BranchAblation]) -> pd.DataFrame:
    """The E-30 table: one row per branch and engineered feature."""
    return pd.concat([results[b].table() for b in BRANCHES if b in results], ignore_index=True)


def save_table(table: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    return path


def proposed_outcome(results: Mapping[str, BranchAblation]) -> dict[str, list[str]]:
    """Per branch, the engineered features the rule drops (in ``features.yaml`` order)."""
    return {branch: results[branch].dropped for branch in BRANCHES}


# ---------------------------------------------------------------------------- RC-02


@dataclass(frozen=True)
class OutcomeCheck:
    """The RC-02 comparison of a newly computed outcome with the committed one."""

    status: Literal["missing", "match", "differs"]
    proposed: dict[str, list[str]]
    committed: dict[str, list[str] | None]
    differences: dict[str, dict[str, list[str]]]

    @property
    def passed(self) -> bool:
        return self.status == "match"

    def message(self) -> str:
        if self.status == "match":
            return "RC-02: the recomputed ablation outcome matches features.yaml."
        if self.status == "missing":
            absent = [b for b, v in self.committed.items() if v is None]
            return (
                f"RC-02 STOP: features.yaml has no committed ablation outcome ({', '.join(absent)}"
                " dropped_engineered absent). Review the ablation table, then write the outcome "
                "into features.yaml (linear.dropped_engineered, tree.dropped_engineered) in a "
                "reviewed commit and run train again. Nothing was written."
            )
        lines = ["RC-02 STOP: the recomputed ablation outcome differs from features.yaml:"]
        for branch, diff in self.differences.items():
            lines.append(
                f"  {branch}: newly dropped {diff['newly_dropped'] or 'none'}; "
                f"no longer dropped {diff['no_longer_dropped'] or 'none'}"
            )
        lines.append("The committed outcome was not changed. Investigate before continuing.")
        return "\n".join(lines)


def check_outcome(features: FeatureConfig, proposed: Mapping[str, list[str]]) -> OutcomeCheck:
    """Compare ``proposed`` with ``features.yaml``'s committed outcome (order-insensitive)."""
    committed: dict[str, list[str] | None] = {
        branch: getattr(features, branch).dropped_engineered for branch in BRANCHES
    }
    proposed = {branch: list(proposed[branch]) for branch in BRANCHES}
    if any(value is None for value in committed.values()):
        return OutcomeCheck("missing", proposed, committed, {})
    differences: dict[str, dict[str, list[str]]] = {}
    for branch in BRANCHES:
        new, old = set(proposed[branch]), set(committed[branch] or [])
        if new != old:
            differences[branch] = {
                "newly_dropped": sorted(new - old),
                "no_longer_dropped": sorted(old - new),
            }
    return OutcomeCheck("differs" if differences else "match", proposed, committed, differences)

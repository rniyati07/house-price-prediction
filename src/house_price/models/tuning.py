"""Hyperparameter tuning (ADR-13, DOC-03 §10). M7 part: the grid-evaluation helper.

``evaluate_grid`` scores a candidate at every point of its log-spaced grid
(``models.yaml``; Ridge 1e-3 to 1e3, Lasso 1e-5 to 1e-1, 25 values each, DOC-03 §10.3,
§10.4) with the standard CV runner on the shared folds, so each point fits a fresh pipeline
per fold on development rows only. M7 uses it to choose Ridge's ablation reference
``alpha`` (DOC-03 §6.6 step 2); M8 reuses it for the Ridge and Lasso tuning (nested MLflow
runs, ``edge_warning``). The Optuna studies are M8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from house_price.config import FeatureConfig, GridConfig, SchemaConfig
from house_price.evaluation.cv import CVResult, FoldPlan, run_cv
from house_price.models.registry import Candidate


def grid_values(grid: GridConfig) -> list[float]:
    """``grid.n`` values evenly spaced in log10 from ``grid.low`` to ``grid.high``."""
    values = np.logspace(np.log10(grid.low), np.log10(grid.high), num=grid.n)
    return [float(v) for v in values]


@dataclass(frozen=True)
class GridResult:
    """Every grid point's CV result, in ascending parameter order."""

    param: str
    values: list[float]
    results: list[CVResult]

    @property
    def best_index(self) -> int:
        """Lowest mean CV log-RMSE; an exact tie goes to the smaller value (first index)."""
        return int(np.argmin([result.mean for result in self.results]))

    @property
    def best_value(self) -> float:
        return self.values[self.best_index]

    @property
    def best(self) -> CVResult:
        return self.results[self.best_index]

    @property
    def on_edge(self) -> bool:
        """The best value is the first or last grid point (DOC-03 §10.3 edge warning)."""
        return self.best_index in (0, len(self.values) - 1)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            self.param: self.values,
            "cv_mean": [r.mean for r in self.results],
            "cv_se": [r.se for r in self.results],
        })  # fmt: skip


def evaluate_grid(
    candidate: Candidate,
    dev: pd.DataFrame,
    plan: FoldPlan,
    features: FeatureConfig,
    schema: SchemaConfig,
) -> GridResult:
    """Score ``candidate`` at every point of its configured grid on the shared folds."""
    if candidate.grid is None:
        raise ValueError(f"candidate {candidate.name!r} has no grid in models.yaml")
    param, values = candidate.grid.param, grid_values(candidate.grid)
    results = [
        run_cv(candidate.build(features, schema, {param: value}), dev, plan, schema)
        for value in values
    ]
    return GridResult(param=param, values=values, results=results)

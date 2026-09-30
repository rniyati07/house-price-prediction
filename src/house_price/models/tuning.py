"""Hyperparameter tuning (ADR-13, DOC-03 §10).

**Grid (Ridge, Lasso; DOC-03 §10.3, §10.4).** ``evaluate_grid`` scores a candidate at every
point of its log-spaced grid (``models.yaml``: Ridge 1e-3 to 1e3, Lasso 1e-5 to 1e-1, 25
values each) with the standard CV runner on the shared folds. M7 uses it to choose Ridge's
ablation reference ``alpha``; M8's ``run_grid_study`` wraps it as a study. ``edge_warning``
is true when the best value is the first or last grid point; the range is never changed.

**Optuna (Random Forest, LightGBM; DOC-03 §10.2, §10.5, §10.6).** ``run_optuna_study``:
``TPESampler(seed)``, direction minimize, sequential trials (``n_jobs=1``), no pruning,
in-memory storage, ``n_trials`` and the search space from ``models.yaml``. Every trial is
scored on all folds of the shared plan.

**Objective.** The mean CV log-RMSE returned by ``run_cv`` (the project metric); CV MAE,
MAPE and R² are carried along for reporting only. Every point or trial fits a fresh
pipeline per fold on development rows only; the holdout is never read.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd

from house_price.config import (
    FeatureConfig,
    GridConfig,
    ModelsConfig,
    SchemaConfig,
    SearchParam,
    ValidationConfig,
)
from house_price.evaluation.cv import CVResult, FoldPlan, run_cv
from house_price.models.registry import Candidate

SAMPLER = "TPESampler"  # DOC-03 §10.2, ADR-13
DIRECTION: Literal["minimize"] = "minimize"
N_JOBS = 1  # sequential: parallel trials make TPE depend on completion order
PRUNER = "none"  # every trial is scored on every fold

optuna.logging.set_verbosity(optuna.logging.WARNING)


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
    on_point: Callable[[int, dict[str, Any], CVResult], None] | None = None,
) -> GridResult:
    """Score ``candidate`` at every point of its configured grid on the shared folds.

    ``on_point(index, params, result)`` is called after each point (e.g. to log it).
    """
    if candidate.grid is None:
        raise ValueError(f"candidate {candidate.name!r} has no grid in models.yaml")
    param, values = candidate.grid.param, grid_values(candidate.grid)
    results = []
    for index, value in enumerate(values):
        result = run_cv(candidate.build(features, schema, {param: value}), dev, plan, schema)
        results.append(result)
        if on_point is not None:
            on_point(index, {param: value}, result)
    return GridResult(param=param, values=values, results=results)


# -------------------------------------------------------------------------- studies


@dataclass(frozen=True)
class Trial:
    """One grid point or Optuna trial: its index, parameters, and CV result."""

    number: int
    params: dict[str, Any]
    result: CVResult


@dataclass(frozen=True)
class StudyResult:
    """A finished tuning study for one candidate (DOC-03 §10.8)."""

    candidate: str
    method: Literal["grid", "optuna"]
    trials: list[Trial]
    duration_seconds: float
    search_space: dict[str, Any]  # the configured grid or search space, as data
    seed: int | None = None  # Optuna sampler seed
    grid: GridResult | None = field(default=None, repr=False)

    @property
    def best(self) -> Trial:
        """Lowest mean CV log-RMSE; an exact tie goes to the earlier trial / smaller value."""
        return self.trials[int(np.argmin([t.result.mean for t in self.trials]))]

    @property
    def edge_warning(self) -> bool | None:
        """Grid studies only: the best value is the first or last grid point (DOC-03 §10.3).
        ``None`` for Optuna studies, which have no documented edge rule."""
        return None if self.grid is None else self.grid.on_edge

    @property
    def params_at_bounds(self) -> dict[str, str] | None:
        """Optuna studies only, informational: best parameters exactly equal to a configured
        bound (``"low"`` / ``"high"``). Not a warning and not a failure."""
        if self.method != "optuna":
            return None
        at: dict[str, str] = {}
        for name, value in self.best.params.items():
            spec = self.search_space[name]
            if value == spec["low"]:
                at[name] = "low"
            elif value == spec["high"]:
                at[name] = "high"
        return at

    def table(self) -> pd.DataFrame:
        """Trial history: one row per point or trial."""
        rows = [{"trial": t.number, **t.params, "cv_mean": t.result.mean, "cv_se": t.result.se,
                 **t.result.secondary, "duration_seconds": t.result.duration_seconds}
                for t in self.trials]  # fmt: skip
        return pd.DataFrame(rows)


TrialCallback = Callable[[Trial], None]


def run_grid_study(
    candidate: Candidate,
    dev: pd.DataFrame,
    plan: FoldPlan,
    features: FeatureConfig,
    schema: SchemaConfig,
    on_trial: TrialCallback | None = None,
) -> StudyResult:
    """Ridge / Lasso tuning: every point of the configured grid (``evaluate_grid``)."""
    if candidate.grid is None:
        raise ValueError(f"candidate {candidate.name!r} has no grid in models.yaml")
    started = time.perf_counter()

    def point(index: int, params: dict[str, Any], result: CVResult) -> None:
        if on_trial is not None:
            on_trial(Trial(index, params, result))

    grid = evaluate_grid(candidate, dev, plan, features, schema, on_point=point)
    trials = [Trial(i, {grid.param: v}, r)
              for i, (v, r) in enumerate(zip(grid.values, grid.results, strict=True))]  # fmt: skip
    return StudyResult(
        candidate=candidate.name, method="grid", trials=trials,
        duration_seconds=time.perf_counter() - started,
        search_space=candidate.grid.model_dump(), grid=grid,
    )  # fmt: skip


def suggest(trial: optuna.Trial, space: Mapping[str, SearchParam]) -> dict[str, Any]:
    """Draw one configuration from the configured search space (exactly its dimensions)."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
        else:
            params[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    return params


def make_study(seed: int) -> optuna.Study:
    """In-memory study: TPE with ``seed``, minimize, no pruning (DOC-03 §10.2)."""
    return optuna.create_study(
        direction=DIRECTION,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )


def run_optuna_study(
    candidate: Candidate,
    dev: pd.DataFrame,
    plan: FoldPlan,
    features: FeatureConfig,
    schema: SchemaConfig,
    seed: int,
    on_trial: TrialCallback | None = None,
) -> StudyResult:
    """Random Forest / LightGBM tuning: ``n_trials`` sequential TPE trials."""
    if not candidate.search_space or candidate.n_trials is None:
        raise ValueError(f"candidate {candidate.name!r} has no search space in models.yaml")
    space = candidate.search_space
    started = time.perf_counter()
    trials: list[Trial] = []

    def objective(optuna_trial: optuna.Trial) -> float:
        params = suggest(optuna_trial, space)
        result = run_cv(candidate.build(features, schema, params), dev, plan, schema)
        trial = Trial(optuna_trial.number, params, result)
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)
        return result.mean  # the only optimization target

    make_study(seed).optimize(objective, n_trials=candidate.n_trials, n_jobs=N_JOBS,
                              show_progress_bar=False)  # fmt: skip
    return StudyResult(
        candidate=candidate.name, method="optuna", trials=trials,
        duration_seconds=time.perf_counter() - started,
        search_space={name: spec.model_dump() for name, spec in space.items()}, seed=seed,
    )  # fmt: skip


def run_study(
    candidate: Candidate,
    dev: pd.DataFrame,
    plan: FoldPlan,
    features: FeatureConfig,
    schema: SchemaConfig,
    seed: int,
    on_trial: TrialCallback | None = None,
) -> StudyResult:
    """The candidate's configured tuning method (``models.yaml`` ``tuning``)."""
    if candidate.tuning == "grid":
        return run_grid_study(candidate, dev, plan, features, schema, on_trial)
    if candidate.tuning == "optuna":
        return run_optuna_study(candidate, dev, plan, features, schema, seed, on_trial)
    raise ValueError(f"candidate {candidate.name!r} is not tuned")


# --------------------------------------------------------------- execution budget


def execution_budget(models: ModelsConfig, validation: ValidationConfig) -> dict[str, Any]:
    """The tuning effort a configuration asks for: grid sizes, trial budgets, search spaces,
    and the CV folds (recorded so a run with a reduced budget is visibly different)."""
    c = models.candidates
    return {
        "ridge_grid_points": c.ridge.grid.n if c.ridge.grid else None,
        "lasso_grid_points": c.lasso.grid.n if c.lasso.grid else None,
        "random_forest_trials": c.random_forest.n_trials,
        "lightgbm_trials": c.lightgbm.n_trials,
        "cv_folds": validation.cv_folds,
        "cv_repeats": validation.cv_repeats,
        "search_spaces": {
            name: {k: v.model_dump() for k, v in (getattr(c, name).search_space or {}).items()}
            or (getattr(c, name).grid.model_dump() if getattr(c, name).grid else None)
            for name in ("ridge", "lasso", "random_forest", "lightgbm")
        },
    }


def budget_deviations(used: dict[str, Any], canonical: dict[str, Any]) -> list[str]:
    """Human-readable differences between a run's budget and the canonical configuration."""
    return [
        f"{key}: {used[key]} (canonical {canonical[key]})"
        if key != "search_spaces"
        else "search_spaces differ from the canonical configuration"
        for key in canonical
        if used.get(key) != canonical[key]
    ]

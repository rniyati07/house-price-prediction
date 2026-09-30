"""Tuning: the M7 grid-evaluation helper and the M8 studies (DOC-03 §10; DOC-05 M7 task 2,
M8-1, M8-2)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from house_price.config import (
    FeatureConfig,
    GridConfig,
    SchemaConfig,
    SearchParam,
    load_feature_config,
    load_models_config,
)
from house_price.data.load import load_raw, sha256_file
from house_price.data.scope import apply_scope_rule
from house_price.evaluation.cv import CVResult, FoldPlan, generate_folds
from house_price.models import tuning
from house_price.models.registry import Candidate, registry
from house_price.models.tuning import GridResult, evaluate_grid, grid_values
from house_price.pipelines.build import build_pipeline
from tests.conftest import (
    CONFIG_DIR,
    FAST_FOLDS,
    FAST_GRID,
    FAST_LGBM_TRIALS,
    FAST_RF_TRIALS,
    make_env,
    use_fast_models,
)


def test_grid_has_25_log_spaced_values_between_the_bounds() -> None:
    for grid, low, high in (
        (GridConfig(param="alpha", low=1e-3, high=1e3, n=25), 1e-3, 1e3),
        (GridConfig(param="alpha", low=1e-5, high=1e-1, n=25), 1e-5, 1e-1),
    ):
        values = grid_values(grid)
        assert len(values) == 25
        assert values[0] == pytest.approx(low) and values[-1] == pytest.approx(high)
        ratios = np.diff(np.log10(values))
        assert np.allclose(ratios, ratios[0])  # equal steps on the log scale
        assert ratios[0] == pytest.approx((math.log10(high) - math.log10(low)) / 24)
        assert grid_values(grid) == values  # deterministic


def test_configured_ridge_and_lasso_grids_are_the_doc03_grids() -> None:
    candidates = registry(load_models_config(CONFIG_DIR), 42)
    ridge = grid_values(candidates["ridge"].grid)  # type: ignore[arg-type]
    lasso = grid_values(candidates["lasso"].grid)  # type: ignore[arg-type]
    assert ridge[12] == pytest.approx(1.0) and lasso[12] == pytest.approx(1e-3)


def _result(mean: float) -> CVResult:
    return CVResult([mean] * 15, mean, 0.0, pd.DataFrame(), {}, 0.0)


def test_best_value_ties_and_edges() -> None:
    grid = GridResult("alpha", [0.1, 1.0, 10.0], [_result(0.3), _result(0.2), _result(0.2)])
    assert grid.best_index == 1 and grid.best_value == 1.0  # tie -> smaller value
    assert not grid.on_edge
    edge = GridResult("alpha", [0.1, 1.0, 10.0], [_result(0.1), _result(0.2), _result(0.3)])
    assert edge.on_edge and edge.best_value == 0.1
    assert list(edge.table().columns) == ["alpha", "cv_mean", "cv_se"]


@pytest.fixture(scope="module")
def dev_and_plan(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig]:
    config = make_env(tmp_path_factory.mktemp("tuning")).load()
    dev, _ = apply_scope_rule(load_raw(config), config.data.scope, config.schema.id_column)
    # 3 folds x 1 repeat keeps the linear fits (Yeo-Johnson on 48 columns) cheap; the real
    # 5 x 3 plan is covered by test_cv.py.
    plan = generate_folds(dev, id_column="Id", target="SalePrice", n_bins=10, n_splits=3,
                          n_repeats=1, seed=42, dev_sha256="0" * 64)  # fmt: skip
    return dev, plan, load_feature_config(CONFIG_DIR), config.schema


@pytest.mark.parametrize("name", ["ridge", "lasso"])
def test_evaluate_grid_scores_every_point_on_the_shared_folds(
    dev_and_plan: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig], name: str
) -> None:
    dev, plan, features, schema = dev_and_plan
    candidate = registry(load_models_config(CONFIG_DIR), 42)[name]
    assert candidate.grid is not None
    small = candidate.grid.model_copy(update={"n": 3})
    candidate = type(candidate)(**{**candidate.__dict__, "grid": small})
    first = evaluate_grid(candidate, dev, plan, features, schema)
    assert first.values == grid_values(small)
    assert len(first.results) == 3
    assert all(len(r.fold_scores) == len(plan.folds) and math.isfinite(r.mean)
               for r in first.results)  # fmt: skip
    second = evaluate_grid(candidate, dev, plan, features, schema)
    assert [r.fold_scores for r in first.results] == [r.fold_scores for r in second.results]


def test_candidate_without_a_grid_is_refused(
    dev_and_plan: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig],
) -> None:
    dev, plan, features, schema = dev_and_plan
    lightgbm = registry(load_models_config(CONFIG_DIR), 42)["lightgbm"]
    with pytest.raises(ValueError, match="no grid"):
        evaluate_grid(lightgbm, dev, plan, features, schema)


# ============================================================== M8: tuning studies
#
# Expected values are DOC-03's (§10.3 to §10.6, DN-06); the canonical files must hold them.

DOC03_RF_SPACE = {
    "n_estimators": ("int", 100, 600, False), "max_depth": ("int", 6, 30, False),
    "min_samples_split": ("int", 2, 20, False), "min_samples_leaf": ("int", 1, 10, False),
    "max_features": ("float", 0.2, 1.0, False),
}  # fmt: skip
DOC03_LGBM_SPACE = {
    "learning_rate": ("float", 0.01, 0.2, True), "num_leaves": ("int", 8, 64, False),
    "min_child_samples": ("int", 5, 50, False), "n_estimators": ("int", 200, 2000, False),
    "subsample": ("float", 0.5, 1.0, False), "colsample_bytree": ("float", 0.3, 1.0, False),
    "reg_alpha": ("float", 1e-8, 10.0, True), "reg_lambda": ("float", 1e-8, 10.0, True),
}  # fmt: skip


def _space(candidate: Candidate) -> dict[str, tuple[str, float, float, bool]]:
    assert candidate.search_space is not None
    return {n: (p.type, p.low, p.high, p.log) for n, p in candidate.search_space.items()}


def test_canonical_search_spaces_and_budgets_are_doc03() -> None:
    """M8-1 / AC-031: exactly the documented dimensions, ranges, scales and budgets."""
    c = registry(load_models_config(CONFIG_DIR), 42)
    assert (c["ridge"].tuning, c["lasso"].tuning) == ("grid", "grid")
    assert (c["ridge"].grid.low, c["ridge"].grid.high, c["ridge"].grid.n) == (1e-3, 1e3, 25)  # type: ignore[union-attr]
    assert (c["lasso"].grid.low, c["lasso"].grid.high, c["lasso"].grid.n) == (1e-5, 1e-1, 25)  # type: ignore[union-attr]
    assert c["lasso"].params["max_iter"] == 50000
    assert _space(c["random_forest"]) == DOC03_RF_SPACE and c["random_forest"].n_trials == 30
    assert _space(c["lightgbm"]) == DOC03_LGBM_SPACE and c["lightgbm"].n_trials == 100
    rf, lgbm = c["random_forest"].params, c["lightgbm"].params
    assert (rf["random_state"], rf["n_jobs"], rf["bootstrap"]) == (42, -1, True)
    assert lgbm["deterministic"] and lgbm["force_row_wise"] and lgbm["n_jobs"] == 1
    assert lgbm["subsample_freq"] == 1 and lgbm["random_state"] == 42
    assert c["ridge"].search_space is None and c["ridge"].n_trials is None


def test_sampler_is_seeded_tpe_sequential_without_pruning() -> None:
    study = tuning.make_study(42)
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    assert (tuning.SAMPLER, tuning.N_JOBS, tuning.PRUNER) == ("TPESampler", 1, "none")


def _draws(seed: int, space: dict[str, SearchParam], n: int = 8) -> list[dict[str, object]]:
    study, draws = tuning.make_study(seed), []

    def objective(trial: optuna.Trial) -> float:
        draws.append(tuning.suggest(trial, space))
        return float(len(draws))

    study.optimize(objective, n_trials=n, n_jobs=1)
    return draws


def test_same_seed_same_suggestions_exactly_the_configured_dimensions() -> None:
    space = registry(load_models_config(CONFIG_DIR), 42)["lightgbm"].search_space
    assert space is not None
    first, second, other = _draws(42, dict(space)), _draws(42, dict(space)), _draws(7, dict(space))
    assert first == second and first != other
    for draw in first:
        assert set(draw) == set(DOC03_LGBM_SPACE)  # no undocumented hyperparameter
        for name, value in draw.items():
            kind, low, high, _ = DOC03_LGBM_SPACE[name]
            assert low <= value <= high  # type: ignore[operator]
            assert isinstance(value, int) == (kind == "int")


def _fake_cv(means: list[float], maes: list[float]) -> Callable[..., CVResult]:
    """A stand-in for run_cv: trial i gets mean means[i] and MAE maes[i]."""
    calls: list[FoldPlan] = []

    def fake(template: object, dev: pd.DataFrame, plan: FoldPlan, schema: object) -> CVResult:
        i = len(calls)
        calls.append(plan)
        return CVResult([means[i]] * 3, means[i], 0.01, pd.DataFrame(),
                        {"cv_mae": maes[i], "cv_mape": 1.0, "cv_r2": 0.5}, 0.0)  # fmt: skip

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _small(candidate: Candidate, **update: object) -> Candidate:
    return replace(candidate, **update)  # type: ignore[arg-type]


def test_objective_is_the_mean_log_rmse_not_a_secondary_metric(
    monkeypatch: pytest.MonkeyPatch,
    dev_and_plan: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig],
) -> None:
    dev, plan, features, schema = dev_and_plan
    # trial 1 has the lowest mean; trial 0 the lowest MAE: the mean must win
    fake = _fake_cv(means=[0.20, 0.10, 0.30], maes=[1.0, 9.0, 5.0])
    monkeypatch.setattr(tuning, "run_cv", fake)
    rf = _small(registry(load_models_config(CONFIG_DIR), 42)["random_forest"], n_trials=3)
    study = tuning.run_optuna_study(rf, dev, plan, features, schema, seed=42)
    assert study.best.number == 1 and study.best.result.mean == 0.10
    assert [t.result.secondary["cv_mae"] for t in study.trials] == [1.0, 9.0, 5.0]  # reported
    assert all(p is plan for p in fake.calls)  # type: ignore[attr-defined]  # shared folds


def test_grid_study_reuses_the_shared_folds_and_flags_edges(
    monkeypatch: pytest.MonkeyPatch,
    dev_and_plan: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig],
) -> None:
    dev, plan, features, schema = dev_and_plan
    ridge = registry(load_models_config(CONFIG_DIR), 42)["ridge"]
    small = _small(ridge, grid=ridge.grid.model_copy(update={"n": 3}))  # type: ignore[union-attr]
    for means, edge in (([0.3, 0.1, 0.2], False), ([0.1, 0.2, 0.3], True), ([0.3, 0.2, 0.1], True)):
        fake = _fake_cv(means, [0.0] * 3)
        monkeypatch.setattr(tuning, "run_cv", fake)
        seen: list[int] = []
        study = tuning.run_grid_study(small, dev, plan, features, schema,
                                      on_trial=lambda t, seen=seen: seen.append(t.number))  # type: ignore[misc]  # fmt: skip
        assert study.edge_warning is edge and study.params_at_bounds is None
        assert seen == [0, 1, 2] and all(p is plan for p in fake.calls)  # type: ignore[attr-defined]
        assert [t.params["alpha"] for t in study.trials] == grid_values(small.grid)  # type: ignore[arg-type]


def test_optuna_bound_information_is_informational_only() -> None:
    space = {
        "a": {"low": 1, "high": 10},
        "b": {"low": 0.1, "high": 0.9},
        "c": {"low": 0, "high": 5},
    }
    study = tuning.StudyResult("x", "optuna", [tuning.Trial(0, {"a": 10, "b": 0.1, "c": 3},
                               _result(0.1))], 1.0, space, seed=42)  # fmt: skip
    assert study.params_at_bounds == {"a": "high", "b": "low"}
    assert study.edge_warning is None  # no documented Optuna edge rule: never a warning


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Predicts the training mean; records every fitted instance and its training size."""

    fitted: ClassVar[list[tuple[int, int]]] = []

    def __init__(self, depth: int = 1):
        self.depth = depth

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        SpyRegressor.fitted.append((id(self), len(X)))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def test_every_trial_fits_a_fresh_pipeline_per_fold_on_training_rows_only(
    dev_and_plan: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig],
) -> None:
    dev, plan, features, schema = dev_and_plan

    def factory(fc: FeatureConfig, sc: SchemaConfig, overrides: object) -> object:
        return build_pipeline(SpyRegressor(**overrides), "tree", fc, sc)  # type: ignore[arg-type]

    spy = Candidate(name="spy", branch="tree", tuning="optuna", tier=2, eligible=True, params={},
                    factory=factory, n_trials=3,  # type: ignore[arg-type]
                    search_space={"depth": SearchParam(type="int", low=1, high=5)})  # fmt: skip
    SpyRegressor.fitted.clear()
    study = tuning.run_optuna_study(spy, dev, plan, features, schema, seed=42)
    fits = SpyRegressor.fitted
    assert len(study.trials) == 3 and len(fits) == 3 * len(plan.folds)
    assert len({fid for fid, _ in fits}) == len(fits)  # a fresh estimator every fold
    train_sizes = [len(f.train_ids) for f in plan.folds] * 3
    assert [n for _, n in fits] == train_sizes  # fitted on the fold's training rows only


def test_budgets_come_from_configuration(tmp_path: Path) -> None:
    """Changing models.yaml changes the study; no source edit (canonical files untouched)."""
    canonical = sha256_file(CONFIG_DIR / "models.yaml")
    env = make_env(tmp_path)
    use_fast_models(env)
    reduced = registry(load_models_config(env.config_dir), 42)
    assert reduced["ridge"].grid.n == FAST_GRID == reduced["lasso"].grid.n  # type: ignore[union-attr]
    assert reduced["random_forest"].n_trials == FAST_RF_TRIALS
    assert reduced["lightgbm"].n_trials == FAST_LGBM_TRIALS
    assert env.load().validation.cv_folds == FAST_FOLDS
    assert sha256_file(CONFIG_DIR / "models.yaml") == canonical
    full = registry(load_models_config(CONFIG_DIR), 42)
    assert (full["random_forest"].n_trials, full["lightgbm"].n_trials) == (30, 100)


def test_candidate_config_rejects_inconsistent_tuning() -> None:
    from pydantic import ValidationError

    from house_price.config import CandidateConfig

    with pytest.raises(ValidationError, match="Optuna candidate needs"):
        CandidateConfig(branch="tree", tier=2, tuning="optuna")
    with pytest.raises(ValidationError, match="grid candidate needs"):
        CandidateConfig(branch="linear", tier=1, tuning="grid")
    with pytest.raises(ValidationError, match="cannot also be searched"):
        CandidateConfig(branch="tree", tier=2, tuning="optuna", n_trials=1, fixed={"n_jobs": 1},
                        search_space={"n_jobs": SearchParam(type="int", low=1, high=2)})  # fmt: skip


def test_budget_deviation_is_recorded_against_the_canonical_configuration(tmp_path: Path) -> None:
    canonical_models, env = load_models_config(CONFIG_DIR), make_env(tmp_path)
    canonical = tuning.execution_budget(canonical_models, env.load().validation)
    assert (canonical["ridge_grid_points"], canonical["lasso_grid_points"]) == (25, 25)
    assert (canonical["random_forest_trials"], canonical["lightgbm_trials"]) == (30, 100)
    assert (canonical["cv_folds"], canonical["cv_repeats"]) == (5, 3)
    assert tuning.budget_deviations(canonical, canonical) == []
    c = canonical_models.candidates
    reduced_models = canonical_models.model_copy(update={"candidates": c.model_copy(update={
        "random_forest": c.random_forest.model_copy(update={"n_trials": 15}),
        "lightgbm": c.lightgbm.model_copy(update={"n_trials": 30}),
    })})  # fmt: skip
    reduced = tuning.execution_budget(reduced_models, env.load().validation)
    assert tuning.budget_deviations(reduced, canonical) == [
        "random_forest_trials: 15 (canonical 30)", "lightgbm_trials: 30 (canonical 100)",
    ]  # fmt: skip
    assert reduced["search_spaces"] == canonical["search_spaces"]

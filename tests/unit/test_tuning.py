"""The M7 grid-evaluation helper (DOC-03 §10.3, §10.4; DOC-05 M7 task 2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from house_price.config import (
    FeatureConfig,
    GridConfig,
    SchemaConfig,
    load_feature_config,
    load_models_config,
)
from house_price.data.load import load_raw
from house_price.data.scope import apply_scope_rule
from house_price.evaluation.cv import CVResult, FoldPlan, generate_folds
from house_price.models.registry import registry
from house_price.models.tuning import GridResult, evaluate_grid, grid_values
from tests.conftest import CONFIG_DIR, make_env


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

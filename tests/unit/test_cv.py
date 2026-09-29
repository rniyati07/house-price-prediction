"""Repeated stratified CV: shared deterministic folds, fresh pipeline per fold, OOF coverage,
standard error (DOC-03 §9.2, §9.3, §11.3; DOC-05 M6-2, M6-3; DOC-01 AC-029)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import ClassVar, Self

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

from house_price.config import FeatureConfig, SchemaConfig, load_feature_config
from house_price.data.load import load_raw
from house_price.data.scope import apply_scope_rule
from house_price.evaluation.cv import CVError, FoldPlan, generate_folds, run_cv, standard_error
from house_price.pipelines import build_pipeline
from tests.conftest import CONFIG_DIR, make_env

N_SPLITS, N_REPEATS, N_BINS = 5, 3, 10


@pytest.fixture(scope="module")
def dev(tmp_path_factory: pytest.TempPathFactory) -> tuple[pd.DataFrame, SchemaConfig]:
    """In-scope fixture rows with Id and SalePrice, standing in for the development set."""
    config = make_env(tmp_path_factory.mktemp("cv")).load()
    frame, _ = apply_scope_rule(load_raw(config), config.data.scope, config.schema.id_column)
    return frame, config.schema


@pytest.fixture(scope="module")
def features() -> FeatureConfig:
    return load_feature_config(CONFIG_DIR)


def _plan(frame: pd.DataFrame, seed: int = 42) -> FoldPlan:
    return generate_folds(frame, id_column="Id", target="SalePrice", n_bins=N_BINS,
                          n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=seed, dev_sha256="0" * 64)  # fmt: skip


# ------------------------------------------------------------------------ folds


def test_fifteen_folds_partition_each_repeat(dev: tuple[pd.DataFrame, SchemaConfig]) -> None:
    frame, _ = dev
    plan = _plan(frame)
    assert len(plan.folds) == 15
    assert [(f.repeat, f.fold) for f in plan.folds] == [
        (r, k) for r in (1, 2, 3) for k in range(1, 6)
    ]
    ids = set(frame["Id"])
    for repeat in (1, 2, 3):
        folds = [f for f in plan.folds if f.repeat == repeat]
        valid = [i for f in folds for i in f.valid_ids]
        assert sorted(valid) == sorted(ids)  # every Id validated exactly once per repeat
        for f in folds:
            assert not set(f.train_ids) & set(f.valid_ids)
            assert set(f.train_ids) | set(f.valid_ids) == ids


def test_folds_are_stratified_on_log_price_deciles(dev: tuple[pd.DataFrame, SchemaConfig]) -> None:
    frame, _ = dev
    bins = pd.qcut(np.log1p(frame["SalePrice"]), q=N_BINS, labels=False)
    by_id = pd.Series(bins.to_numpy(), index=frame["Id"])
    for fold in _plan(frame).folds:
        counts = by_id[fold.valid_ids].value_counts().reindex(range(N_BINS), fill_value=0)
        assert counts.max() - counts.min() <= 1  # each decile spread evenly over the folds


def test_same_seed_same_folds_different_seed_different(
    dev: tuple[pd.DataFrame, SchemaConfig],
) -> None:
    frame, _ = dev
    assert _plan(frame, seed=42) == _plan(frame, seed=42)
    assert _plan(frame, seed=42).folds != _plan(frame, seed=7).folds


def test_fold_file_round_trip_and_dev_hash_check(
    dev: tuple[pd.DataFrame, SchemaConfig], tmp_path: Path
) -> None:
    plan = _plan(dev[0])
    path = tmp_path / "artifacts" / "cv" / "folds.json"
    plan.save(path)
    assert FoldPlan.load(path, "0" * 64) == plan
    with pytest.raises(CVError, match="different development set"):
        FoldPlan.load(path, "1" * 64)


# ----------------------------------------------------------------------- runner


class SpyRegressor(RegressorMixin, BaseEstimator):
    """Predicts the training mean; remembers every instance that was fitted."""

    fitted: ClassVar[list[SpyRegressor]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        SpyRegressor.fitted.append(self)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


def test_every_fold_fits_a_fresh_pipeline(
    dev: tuple[pd.DataFrame, SchemaConfig], features: FeatureConfig
) -> None:
    frame, schema = dev
    SpyRegressor.fitted.clear()
    template = build_pipeline(SpyRegressor(), "tree", features, schema)
    run_cv(template, frame, _plan(frame), schema)
    fitted = SpyRegressor.fitted
    assert len(fitted) == 15
    assert len({id(model) for model in fitted}) == 15  # 15 distinct objects, all kept alive
    assert template.regressor.named_steps["model"] not in fitted
    assert not hasattr(template, "regressor_")  # the template itself is never fitted


def test_cv_result_scores_mean_se_and_oof(
    dev: tuple[pd.DataFrame, SchemaConfig], features: FeatureConfig
) -> None:
    frame, schema = dev
    template = build_pipeline(Ridge(alpha=1.0), "linear", features, schema)
    result = run_cv(template, frame, _plan(frame), schema)
    assert len(result.fold_scores) == 15
    assert all(math.isfinite(s) and s > 0 for s in result.fold_scores)
    assert result.mean == pytest.approx(np.mean(result.fold_scores))
    assert result.se == pytest.approx(np.std(result.fold_scores, ddof=1) / math.sqrt(15))
    oof = result.oof
    assert len(oof) == 3 * len(frame)
    for repeat in (1, 2, 3):
        assert sorted(oof.loc[oof["repeat"] == repeat, "Id"]) == sorted(frame["Id"])
    assert np.isfinite(oof["log_prediction"]).all()
    assert set(result.secondary) == {"cv_mae", "cv_mape", "cv_r2"}
    assert result.duration_seconds > 0


def test_standard_error_by_hand() -> None:
    # scores 1, 2, 3, 4: sample SD = sqrt(5/3); SE = sqrt(5/3) / 2
    assert standard_error([1.0, 2.0, 3.0, 4.0]) == pytest.approx(math.sqrt(5 / 3) / 2)
    with pytest.raises(ValueError):
        standard_error([1.0])


def test_plan_for_other_rows_is_refused(
    dev: tuple[pd.DataFrame, SchemaConfig], features: FeatureConfig
) -> None:
    frame, schema = dev
    template = build_pipeline(Ridge(), "linear", features, schema)
    with pytest.raises(CVError, match="does not partition"):
        run_cv(template, frame.iloc[:-5], _plan(frame), schema)

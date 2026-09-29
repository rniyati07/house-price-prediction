"""Leakage safety of the fitted preprocessing (DOC-01 AC-024, AC-025; DOC-05 M5-3).

Every fitted statistic (imputation values, missing-indicator columns, Yeo-Johnson lambdas,
scaler parameters, encoder categories) must come only from the rows the pipeline is fitted
on: transforming other data must not change them, and they must equal the statistics of a
fresh pipeline fitted on the same rows alone. This is what makes later CV folds safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge

from house_price.pipelines import build_pipeline
from house_price.pipelines.branches import Branch
from tests.conftest import ModelData

FITTED_ATTRIBUTES = ("statistics_", "lambdas_", "mean_", "scale_", "var_", "categories_")


def _estimator(branch: Branch) -> Ridge | LGBMRegressor:
    if branch == "linear":
        return Ridge(alpha=1.0)
    return LGBMRegressor(n_estimators=20, min_child_samples=5, random_state=42,
                         deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1)  # fmt: skip


def _fit(data: ModelData, branch: Branch, rows: pd.Index) -> TransformedTargetRegressor:
    model = build_pipeline(_estimator(branch), branch, data.features, data.schema)
    return model.fit(data.X.loc[rows], data.y.loc[rows])


def _ct(model: TransformedTargetRegressor) -> ColumnTransformer:
    ct: ColumnTransformer = model.regressor_.named_steps["preprocess"]
    return ct


def fitted_statistics(model: TransformedTargetRegressor) -> dict[str, Any]:
    """Every learned preprocessing statistic, as plain comparable values."""
    stats: dict[str, Any] = {}
    for group, pipeline in _ct(model).named_transformers_.items():
        if isinstance(pipeline, str):
            continue
        for step_name, step in pipeline.steps:
            for attribute in FITTED_ATTRIBUTES:
                if hasattr(step, attribute):
                    value = getattr(step, attribute)
                    if isinstance(value, list):
                        value = [list(map(str, v)) for v in value]
                    else:
                        value = np.asarray(value).tolist()
                    stats[f"{group}.{step_name}.{attribute}"] = value
            indicator = getattr(step, "indicator_", None)
            if indicator is not None:
                stats[f"{group}.{step_name}.indicator_features"] = indicator.features_.tolist()
    return stats


@pytest.fixture(scope="module")
def rows(model_data: ModelData) -> tuple[pd.Index, pd.Index]:
    """Data A (fitting rows) and data B (other rows), both with missing LotFrontage."""
    index = model_data.X.index
    a, b = index[:60], index[60:]
    assert model_data.X.loc[a, "LotFrontage"].isna().any()
    assert model_data.X.loc[b, "LotFrontage"].isna().any()
    return a, b


# ------------------------------------------------ G: imputation and indicators (AC-024)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_imputed_value_is_the_fitting_data_median(
    branch: Branch, model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    a, b = rows
    model = _fit(model_data, branch, a)
    numeric = _ct(model).named_transformers_["numeric"]
    imputer = numeric.named_steps["impute"]
    position = list(numeric.feature_names_in_).index("LotFrontage")
    median_a = model_data.X.loc[a, "LotFrontage"].median()
    assert imputer.statistics_[position] == median_a
    assert median_a != model_data.X["LotFrontage"].median()  # A's median, not all rows'

    out = model.regressor_[:-1].transform(model_data.X.loc[b])
    indicator = out["numeric__missingindicator_LotFrontage"]
    missing = model_data.X.loc[b, "LotFrontage"].isna()
    if branch == "tree":  # no later transform: a plain 0/1 flag and the raw imputed value
        assert (indicator[missing] == 1).all() and (indicator[~missing] == 0).all()
        assert (out.loc[missing, "numeric__LotFrontage"] == median_a).all()
    else:  # DOC-03 §7.2: the flag also passes through Yeo-Johnson and scaling, which
        # rescale it monotonically; it still takes one value per state, missing above present
        assert indicator[missing].nunique() == 1 and indicator[~missing].nunique() == 1
        assert indicator[missing].iloc[0] > indicator[~missing].iloc[0]


def test_indicators_exist_only_for_columns_missing_in_the_fitting_data(
    model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    a, _ = rows
    model = _fit(model_data, "tree", a)
    names = list(_ct(model).get_feature_names_out())
    indicated = {n.removeprefix("numeric__missingindicator_") for n in names
                 if n.startswith("numeric__missingindicator_")}  # fmt: skip
    engineered = model.regressor_[:2].transform(model_data.X.loc[a])
    numeric = _ct(model).named_transformers_["numeric"].feature_names_in_
    assert indicated == {c for c in numeric if engineered[c].isna().any()}


# ------------------------------------------------------------ H: leakage (AC-025)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_transforming_other_data_does_not_change_fitted_statistics(
    branch: Branch, model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    a, b = rows
    model = _fit(model_data, branch, a)
    before = fitted_statistics(model)
    model.regressor_[:-1].transform(model_data.X.loc[b])
    model.predict(model_data.X.loc[b])
    model.predict(model_data.X)
    assert fitted_statistics(model) == before


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_statistics_equal_those_of_fitting_on_a_alone(
    branch: Branch, model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    a, b = rows
    used = _fit(model_data, branch, a)
    used.predict(model_data.X.loc[b])
    fresh = _fit(model_data, branch, a)
    assert fitted_statistics(used) == fitted_statistics(fresh)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_the_leakage_check_is_sensitive(
    branch: Branch, model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    """Sanity: fitting on B does change the statistics, so the equality above is meaningful."""
    a, b = rows
    assert fitted_statistics(_fit(model_data, branch, a)) != fitted_statistics(
        _fit(model_data, branch, b)
    )


def test_linear_statistics_cover_every_fitted_step(
    model_data: ModelData, rows: tuple[pd.Index, pd.Index]
) -> None:
    stats = fitted_statistics(_fit(model_data, "linear", rows[0]))
    for key in ("numeric.impute.statistics_", "numeric.impute.indicator_features",
                "numeric.power.lambdas_", "numeric.scale.mean_", "numeric.scale.scale_",
                "ordinal.impute.statistics_", "ordinal.scale.mean_", "nominal.impute.statistics_",
                "nominal.encode.categories_"):  # fmt: skip
        assert key in stats, key

"""M5 pipeline architecture: branches, exclusions, dollars, unseen categories, clone,
configuration-driven groups, joblib round trip, group coverage, column order, feature names.

DOC-01 AC-018, AC-020 to AC-024, AC-026, AC-027; DOC-05 M5-1 to M5-6. The estimators here
(Ridge, LightGBM) are simple stand-ins to exercise the architecture, not project candidates.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from house_price.config import BranchGroups, FeatureConfig
from house_price.data import split
from house_price.data.load import load_raw
from house_price.data.schema import model_input_columns, select_model_input
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.features.engineer import FeatureEngineer
from house_price.features.semantic import SemanticNAFiller
from house_price.pipelines import build_column_transformer, build_pipeline, check_group_coverage
from house_price.pipelines.branches import Branch, branch_groups, feature_engineer_output
from house_price.pipelines.check import main as check_main
from house_price.results import read_index, read_result
from tests.conftest import CONFIG_DIR, REPO_ROOT, ModelData, make_env

EXCLUDED = {"Id", "PID", "SalePrice", "SaleType", "SaleCondition"}
TOLERANCE = 1e-6  # IN-08


def _ridge() -> Ridge:
    return Ridge(alpha=1.0)


def _lightgbm() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=30, min_child_samples=5, random_state=42, deterministic=True,
        force_row_wise=True, n_jobs=1, verbose=-1,
    )  # fmt: skip


def _fit(data: ModelData, branch: Branch, estimator: BaseEstimator | None = None,
         features: FeatureConfig | None = None, rows: pd.Index | None = None,
         ) -> TransformedTargetRegressor:  # fmt: skip
    estimator = (
        estimator if estimator is not None else (_ridge() if branch == "linear" else _lightgbm())
    )
    model = build_pipeline(estimator, branch, features or data.features, data.schema)
    index = rows if rows is not None else data.X.index
    return model.fit(data.X.loc[index], data.y.loc[index])


def _preprocessed(model: TransformedTargetRegressor, X: pd.DataFrame) -> pd.DataFrame:
    """The ColumnTransformer output (everything before the estimator)."""
    out: pd.DataFrame = model.regressor_[:-1].transform(X)
    return out


def _ct(model: TransformedTargetRegressor) -> ColumnTransformer:
    ct: ColumnTransformer = model.regressor_.named_steps["preprocess"]
    return ct


@pytest.fixture(scope="module")
def linear(model_data: ModelData) -> TransformedTargetRegressor:
    return _fit(model_data, "linear")


@pytest.fixture(scope="module")
def tree(model_data: ModelData) -> TransformedTargetRegressor:
    return _fit(model_data, "tree")


# ------------------------------------------------------------ A: exclusions (AC-018)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_pipeline_inputs_are_the_77_model_columns(
    branch: Branch, model_data: ModelData, linear: TransformedTargetRegressor,
    tree: TransformedTargetRegressor,
) -> None:  # fmt: skip
    model = linear if branch == "linear" else tree
    inputs = list(model.regressor_.named_steps["semantic_na"].feature_names_in_)
    assert inputs == model_input_columns(model_data.schema)
    assert len(inputs) == 77
    assert not EXCLUDED & set(inputs)
    assert "YrSold" in inputs  # kept as the valuation year (FR-014)


def test_rows_are_kept(model_data: ModelData, linear: TransformedTargetRegressor) -> None:
    assert len(_preprocessed(linear, model_data.X)) == len(model_data.X)
    assert len(linear.predict(model_data.X)) == len(model_data.X)
    assert model_data.X["SaleType" if "SaleType" in model_data.X else "YrSold"].notna().all()


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_only_the_dropped_group_is_removed(
    branch: Branch, linear: TransformedTargetRegressor, tree: TransformedTargetRegressor,
    model_data: ModelData,
) -> None:  # fmt: skip
    """``remainder="drop"`` removes exactly the effective dropped group: ``GarageYrBlt`` plus
    the branch's committed ablation drops (DOC-03 §7.5, §6.6)."""
    ct = _ct(linear if branch == "linear" else tree)
    name, action, columns = ct.transformers_[-1]
    assert (name, action) == ("remainder", "drop")
    expected = branch_groups(branch, model_data.features).dropped
    assert sorted(columns) == sorted(expected)
    configured = getattr(model_data.features, branch)
    assert expected[0] == "GarageYrBlt" and set(expected[1:]) == set(
        configured.dropped_engineered or []
    )


# ------------------------------------------------- B: forbidden techniques (AC-020)

ALLOWED_COMPONENTS = {
    TransformedTargetRegressor, Pipeline, ColumnTransformer, SemanticNAFiller, FeatureEngineer,
    SimpleImputer, PowerTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder,
}  # fmt: skip


def _components(obj: object) -> Iterator[object]:
    yield obj
    if isinstance(obj, TransformedTargetRegressor):
        yield from _components(obj.regressor)
    elif isinstance(obj, Pipeline):
        for _, step in obj.steps:
            yield from _components(step)
    elif isinstance(obj, ColumnTransformer):
        for _, transformer, _ in obj.transformers:
            if not isinstance(transformer, str):
                yield from _components(transformer)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_no_forbidden_steps(branch: Branch, model_data: ModelData) -> None:
    """Only approved components: no target encoder, polynomial features, PCA, or generator."""
    model = build_pipeline(_ridge(), branch, model_data.features, model_data.schema)
    classes = {type(c) for c in _components(model)} - {Ridge}
    assert classes <= ALLOWED_COMPONENTS, classes - ALLOWED_COMPONENTS
    names = " ".join(c.__name__.lower() for c in classes)
    for forbidden in ("target", "polynomial", "pca", "spline", "kernel"):
        assert forbidden not in names.replace("transformedtargetregressor", "")


# ------------------------------------------------------------ C: dollars (AC-021)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_predictions_are_finite_positive_dollars(
    branch: Branch, model_data: ModelData, linear: TransformedTargetRegressor,
    tree: TransformedTargetRegressor,
) -> None:  # fmt: skip
    model = linear if branch == "linear" else tree
    assert "SalePrice" not in model_data.X
    predictions = model.predict(model_data.X)
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    assert 20_000 < np.median(predictions) < 1_000_000  # dollars, not log dollars (~12)
    # The estimator itself learns log1p(price); expm1 brings it back to dollars.
    assert model.func is np.log1p and model.inverse_func is np.expm1
    np.testing.assert_allclose(np.expm1(model.regressor_.predict(model_data.X)), predictions)


# ------------------------------------------------ D and E: the branches (AC-022)


def _step_types(ct: ColumnTransformer, group: str) -> list[type]:
    return [type(step) for _, step in ct.named_transformers_[group].steps]


def test_linear_branch_steps(linear: TransformedTargetRegressor) -> None:
    ct = _ct(linear)
    assert _step_types(ct, "numeric") == [SimpleImputer, PowerTransformer, StandardScaler]
    assert _step_types(ct, "ordinal") == [SimpleImputer, StandardScaler]
    assert _step_types(ct, "nominal") == [SimpleImputer, OneHotEncoder]
    numeric = ct.named_transformers_["numeric"]
    assert numeric.named_steps["impute"].strategy == "median"
    assert numeric.named_steps["impute"].add_indicator is True
    assert numeric.named_steps["power"].method == "yeo-johnson"
    assert numeric.named_steps["power"].standardize is False
    assert ct.named_transformers_["ordinal"].named_steps["impute"].strategy == "most_frequent"
    onehot = ct.named_transformers_["nominal"].named_steps["encode"]
    assert (onehot.handle_unknown, onehot.sparse_output) == ("ignore", False)


def test_linear_output_is_scaled(model_data: ModelData, linear: TransformedTargetRegressor) -> None:
    """IN-08: numeric and ordinal columns have mean 0 and SD 1 on the fitting data."""
    out = _preprocessed(linear, model_data.X)
    scaled = [c for c in out if c.startswith(("numeric__", "ordinal__"))]
    varying = [c for c in scaled if out[c].std(ddof=0) > TOLERANCE]
    assert varying
    np.testing.assert_allclose(out[scaled].mean(), 0, atol=TOLERANCE)
    np.testing.assert_allclose(out[varying].std(ddof=0), 1, atol=TOLERANCE)
    onehot = out[[c for c in out if c.startswith("nominal__")]]
    assert set(np.unique(onehot.to_numpy())) <= {0.0, 1.0}


def test_tree_branch_steps(tree: TransformedTargetRegressor) -> None:
    ct = _ct(tree)
    assert _step_types(ct, "numeric") == [SimpleImputer]
    assert _step_types(ct, "ordinal") == [SimpleImputer]
    assert _step_types(ct, "nominal") == [SimpleImputer, OrdinalEncoder]
    encoder = ct.named_transformers_["nominal"].named_steps["encode"]
    assert (encoder.handle_unknown, encoder.unknown_value) == ("use_encoded_value", -1)


def test_tree_output_is_not_scaled(model_data: ModelData, tree: TransformedTargetRegressor) -> None:
    out = _preprocessed(tree, model_data.X)
    engineered = tree.regressor_[:2].transform(model_data.X)
    active = branch_groups("tree", model_data.features).numeric
    retained = [f for f in model_data.features.engineered
                if f in active and engineered[f].notna().all()]  # fmt: skip
    assert retained
    for column in ("GrLivArea", "LotArea", *retained):
        np.testing.assert_array_equal(out[f"numeric__{column}"], engineered[column])
    assert set(np.unique(out.filter(like="ordinal__").to_numpy())) <= {0, 1, 2, 3, 4, 5}
    codes = out.filter(like="nominal__").to_numpy()
    assert (codes >= 0).all() and (codes == np.round(codes)).all()


# ----------------------------------------------- F: unseen categories (AC-023)


def _unseen_neighborhood(data: ModelData, fit_rows: pd.Index) -> str:
    allowed = data.schema.by_name("Neighborhood").allowed_values or []
    seen = set(data.X.loc[fit_rows, "Neighborhood"])
    return str(next(v for v in allowed if v not in seen))


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_allowed_but_unseen_category_is_scored(branch: Branch, model_data: ModelData) -> None:
    fit_rows = model_data.X.index[:70]
    model = _fit(model_data, branch, rows=fit_rows)
    row = model_data.X.iloc[[80]].copy()
    row["Neighborhood"] = _unseen_neighborhood(model_data, fit_rows)
    prediction = model.predict(row)
    assert np.isfinite(prediction).all() and (prediction > 0).all()
    out = _preprocessed(model, row)
    if branch == "tree":
        assert out["nominal__Neighborhood"].iloc[0] == -1
    else:
        assert (out.filter(like="nominal__Neighborhood_").to_numpy() == 0).all()


# ------------------------------------------------------------ I: clone (AC-026)


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_clone_fits_and_predicts(
    branch: Branch, model_data: ModelData, linear: TransformedTargetRegressor,
    tree: TransformedTargetRegressor,
) -> None:  # fmt: skip
    fitted = linear if branch == "linear" else tree
    for template in (
        build_pipeline(_ridge(), branch, model_data.features, model_data.schema),
        fitted,
    ):
        copy = clone(template)
        assert not hasattr(copy, "regressor_")  # a clone is always unfitted
        predictions = copy.fit(model_data.X, model_data.y).predict(model_data.X)
        assert np.isfinite(predictions).all() and (predictions > 0).all()
    np.testing.assert_array_equal(clone(fitted).fit(model_data.X, model_data.y).predict(model_data.X),
                                  fitted.predict(model_data.X))  # fmt: skip


# ---------------------------------------------- J: configuration driven (AC-027)


def _move(features: FeatureConfig, column: str, source: str, target: str) -> FeatureConfig:
    groups = features.linear
    update = {
        source: [c for c in getattr(groups, source) if c != column],
        target: [*getattr(groups, target), column],
    }
    return features.model_copy(update={"linear": groups.model_copy(update=update)})


def test_moving_a_column_between_groups_changes_its_treatment(model_data: ModelData) -> None:
    before = _preprocessed(_fit(model_data, "linear"), model_data.X)
    moved = _move(model_data.features, "MoSold", "numeric", "nominal")
    after = _preprocessed(_fit(model_data, "linear", features=moved), model_data.X)
    assert "numeric__MoSold" in before and "numeric__MoSold" not in after
    one_hot = [c for c in after if c.startswith("nominal__MoSold_")]
    assert len(one_hot) == model_data.X["MoSold"].nunique()


def test_a_column_missing_from_every_group_is_an_error(model_data: ModelData) -> None:
    groups = model_data.features.linear
    broken = model_data.features.model_copy(update={"linear": groups.model_copy(
        update={"numeric": [c for c in groups.numeric if c != "LotArea"]})})  # fmt: skip
    with pytest.raises(ValueError, match="LotArea"):
        build_pipeline(_ridge(), "linear", broken, model_data.schema)


# ---------------------------------------------------------- K: joblib round trip


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_joblib_round_trip_is_exact(
    branch: Branch, model_data: ModelData, linear: TransformedTargetRegressor,
    tree: TransformedTargetRegressor, tmp_path: Path,
) -> None:  # fmt: skip
    model = linear if branch == "linear" else tree
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(reloaded.predict(model_data.X), model.predict(model_data.X))


# ------------------------------------------------------------ L: group coverage


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_groups_cover_the_feature_engineer_output_exactly_once(
    branch: Branch, model_data: ModelData
) -> None:
    check_group_coverage(branch, model_data.features, model_data.schema)
    groups: BranchGroups = getattr(model_data.features, branch)
    columns = groups.all_columns
    assert len(columns) == len(set(columns))
    engineered = FeatureEngineer.from_config(model_data.features).fit_transform(
        SemanticNAFiller.from_config(model_data.features).fit_transform(model_data.X)
    )
    assert sorted(columns) == sorted(engineered.columns)
    assert sorted(columns) == sorted(
        feature_engineer_output(model_data.schema, model_data.features)
    )


# ----------------------------------------------------------- M: column order


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_reordered_request_columns_give_identical_predictions(
    branch: Branch, model_data: ModelData, linear: TransformedTargetRegressor,
    tree: TransformedTargetRegressor,
) -> None:  # fmt: skip
    model = linear if branch == "linear" else tree
    shuffled = model_data.X[list(np.random.default_rng(42).permutation(model_data.X.columns))]
    assert list(shuffled.columns) != list(model_data.X.columns)
    reordered = select_model_input(shuffled, model_data.schema)
    np.testing.assert_array_equal(model.predict(reordered), model.predict(model_data.X))
    np.testing.assert_array_equal(model.predict(shuffled), model.predict(model_data.X))


# ------------------------------------------------------------ N: feature names


def test_ridge_feature_names(linear: TransformedTargetRegressor, model_data: ModelData) -> None:
    names = list(_ct(linear).get_feature_names_out())
    for prefix in ("numeric__", "ordinal__", "nominal__"):
        assert any(n.startswith(prefix) for n in names), prefix
    groups = branch_groups("linear", model_data.features)
    for feature in model_data.features.engineered:  # the committed linear outcome applies
        assert (f"numeric__{feature}" in names) == (feature in groups.numeric), feature
    assert "ordinal__KitchenQual" in names
    assert any(n.startswith("nominal__MSSubClass_") for n in names)
    assert any(n.startswith("numeric__missingindicator_") for n in names)
    for absent in ("GarageYrBlt", "SaleType", "SaleCondition", "SalePrice"):
        assert not any(absent in n for n in names), absent
    bases = {n.split("__", 1)[1] for n in names}
    assert not {"Id", "PID"} & bases


def test_impossible_garage_year_is_imputed_like_a_missing_year(
    model_data: ModelData, linear: TransformedTargetRegressor
) -> None:
    """DOC-03 §6.4 downstream: GarageYrBlt > YrSold reaches the fitted pipeline exactly as a
    missing garage year does (median imputation + indicator), never as a negative age."""
    garage = model_data.X[model_data.X["GarageType"].notna()].head(1)
    impossible = garage.assign(GarageYrBlt=2207.0)
    unknown = garage.assign(GarageYrBlt=np.nan)
    out_impossible, out_unknown = _preprocessed(linear, impossible), _preprocessed(linear, unknown)
    pd.testing.assert_frame_equal(out_impossible, out_unknown)
    assert np.isfinite(out_impossible.to_numpy(dtype="float64")).all()
    engineered = linear.regressor_[:2].transform(impossible)
    assert pd.isna(engineered["GarageAge"]).all()
    assert np.isfinite(linear.predict(impossible)).all()


# ------------------------------------------------------------------ real dataset


@pytest.mark.data
def test_ridge_pipeline_on_the_development_set(model_data: ModelData) -> None:
    """DOC-05 M5 recommended verification: fit on the development set, read the names."""
    from house_price.eda.context import EDAContext

    dev = EDAContext.load(CONFIG_DIR, REPO_ROOT).dev
    X, y = select_model_input(dev, model_data.schema), dev["SalePrice"]
    model = build_pipeline(_ridge(), "linear", model_data.features, model_data.schema).fit(X, y)
    names = list(_ct(model).get_feature_names_out())
    groups = {prefix: sum(n.startswith(f"{prefix}__") for n in names)
              for prefix in ("numeric", "ordinal", "nominal")}  # fmt: skip
    assert all(groups.values())
    assert not any("GarageYrBlt" in n for n in names)
    predictions = model.predict(X)
    assert np.isfinite(predictions).all() and (predictions > 0).all()


def test_build_column_transformer_is_unfitted_and_pandas(model_data: ModelData) -> None:
    for branch in ("linear", "tree"):
        ct = build_column_transformer(branch, model_data.features)  # type: ignore[arg-type]
        assert not hasattr(ct, "transformers_")
        assert ct.remainder == "drop" and ct.verbose_feature_names_out is True
        assert [name for name, _, _ in ct.transformers] == ["numeric", "ordinal", "nominal"]
    with pytest.raises(ValueError, match="unknown branch"):
        build_column_transformer("forest", model_data.features)  # type: ignore[arg-type]


# ------------------------------------------------------- M5 preprocessing check record


def test_preprocessing_check_record_matches_the_fitted_pipeline(
    tmp_path: Path, model_data: ModelData
) -> None:
    """``python -m house_price.pipelines`` records what the real pipeline produces on the
    development set; compared here with a full pipeline fitted independently."""
    env = make_env(tmp_path)
    assert split.main(env.cli_args()) == 0
    assert check_main(env.cli_args()) == 0
    (line,) = read_index(tmp_path / "results")
    record = read_result(tmp_path / "results" / line["result"])
    assert record["milestone"] == "M5" and record["status"] == "succeeded"
    assert record["params"]["estimator_fitted"] is False

    config = env.load()
    in_scope, scope = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    dev = create_or_load_split(in_scope, scope, config).dev
    X, y = select_model_input(dev, model_data.schema), dev["SalePrice"]
    for branch in ("linear", "tree"):
        model = build_pipeline(Ridge(), branch, model_data.features, model_data.schema).fit(X, y)
        names = list(_ct(model).get_feature_names_out())
        metrics = record["metrics"]
        assert metrics[f"{branch}.n_rows"] == len(dev)
        assert metrics[f"{branch}.n_inputs"] == len(model_input_columns(model_data.schema)) == 77
        assert metrics[f"{branch}.n_output"] == len(names)
        for group in ("numeric", "ordinal", "nominal"):
            expected = sum(n.startswith(f"{group}__") for n in names)
            assert metrics[f"{branch}.output.{group}"] == expected
        details = record["findings"][branch]
        assert details["indicators"] == [n for n in names if "missingindicator_" in n]
        assert details["target_func"] == "log1p" and details["target_inverse_func"] == "expm1"
        assert details["output_has_missing"] is False
    assert record["lineage"]["data_sha256"] == config.data.raw_sha256

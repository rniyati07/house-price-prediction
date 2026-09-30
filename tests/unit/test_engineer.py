"""FeatureEngineer: the 12 formulas, ordinal map, MSSubClass cast (FR-012, FR-013, AC-015 to AC-017)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from sklearn.base import clone

from house_price.config import BranchGroups, FeatureConfig, load_project_config
from house_price.features.engineer import FORMULAS, FeatureEngineer
from house_price.features.semantic import SemanticNAFiller
from tests.conftest import CONFIG_DIR, REPO_ROOT

NAN = math.nan
FEATURES = ["TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled", "TotalPorchSF",
            "HasPool", "HasGarage", "HasBsmt", "HasFireplace", "Has2ndFlr", "GarageAge"]  # fmt: skip
ORDINAL = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual",
           "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]  # fmt: skip
NOMINAL_ORDERED_LOOKING = ["BsmtExposure", "BsmtFinType1", "GarageFinish", "Functional"]

# Hand-computed from tests/fixtures/feature_rows.csv (after the semantic filler), in the
# order of FEATURES. Row notes: 1 typical; 2 no garage/basement/fireplace/2nd floor, not
# remodeled; 3 pool; 4 garage without a recorded year (and the 1950 remodel floor);
# 5 sold before completion (negative house/remodel ages; garage year 2009 after the 2007
# sale is impossible, so GarageAge is unknown); 6/7 basement anomalies; 8 basement unrecorded;
# 9 garage type recorded but every other garage field unrecorded (cars and area too).
EXPECTED = {
    1: [2700, 3.5, 8, 3, 1, 40, 0, 1, 1, 1, 1, 8],
    2: [900, 1.0, 60, 60, 0, 50, 0, 0, 0, 0, 0, 0],
    3: [3000, 3.5, 1, 1, 0, 280, 1, 1, 1, 1, 0, 1],
    4: [1900, 1.5, 84, 59, 1, 0, 0, 1, 1, 0, 1, NAN],
    5: [2400, 2.0, -1, -2, 1, 30, 0, 1, 1, 1, 0, NAN],
    6: [2000, 2.0, 33, 18, 1, 0, 0, 1, 1, 0, 0, 33],
    7: [3000, 3.0, 11, 11, 0, 50, 0, 1, 1, 1, 1, 11],
    8: [896, 1.0, 62, 58, 1, 0, 0, 1, 0, 0, 0, 62],
    9: [1800, 1.0, 84, 7, 1, 0, 0, 1, 1, 0, 0, NAN],
}


@pytest.fixture
def engineer(feature_config: FeatureConfig) -> FeatureEngineer:
    return FeatureEngineer.from_config(feature_config)


@pytest.fixture
def filled(feature_config: FeatureConfig, feature_rows: pd.DataFrame) -> pd.DataFrame:
    return SemanticNAFiller.from_config(feature_config).fit_transform(feature_rows)


@pytest.fixture
def out(engineer: FeatureEngineer, filled: pd.DataFrame) -> pd.DataFrame:
    return engineer.fit_transform(filled)


# ----------------------------------------------------------------- configuration


def test_configuration_matches_the_approved_design(feature_config: FeatureConfig) -> None:
    assert feature_config.engineered == FEATURES
    assert set(FORMULAS) == set(FEATURES)
    assert feature_config.ordinal.columns == ORDINAL
    assert feature_config.ordinal.mapping == {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5,
    }
    assert feature_config.categorical_codes == ["MSSubClass"]


def test_initial_groups_cover_the_engineer_output(feature_config: FeatureConfig) -> None:
    """Both branches start identical and partition the 77 inputs plus 12 features; only the
    M7 ablation outcome (``dropped_engineered``) differs per branch (DOC-03 §6.6)."""
    schema = load_project_config(CONFIG_DIR, REPO_ROOT).schema
    output = {spec.name for spec in schema.with_role("model_input")} | set(FEATURES)
    for branch in (feature_config.linear, feature_config.tree):
        assert sorted(branch.all_columns) == sorted(output)
        assert branch.dropped == ["GarageYrBlt"]
        assert branch.ordinal == ORDINAL
        assert set(NOMINAL_ORDERED_LOOKING) | {"MSSubClass"} <= set(branch.nominal)
        assert set(FEATURES) <= set(branch.numeric)

    def initial(groups: BranchGroups) -> BranchGroups:
        return groups.model_copy(update={"dropped_engineered": None})

    assert initial(feature_config.linear) == initial(feature_config.tree)


# ------------------------------------------------------------------- formulas


@pytest.mark.parametrize("feature", FEATURES)
def test_formula_matches_hand_computed_values(feature: str, out: pd.DataFrame) -> None:
    """AC-016: every formula, every fixture row, exactly."""
    position = FEATURES.index(feature)
    for row_id, values in EXPECTED.items():
        expected, observed = values[position], out.loc[row_id, feature]
        if math.isnan(expected):
            assert pd.isna(observed), (feature, row_id)
        else:
            assert observed == expected, (feature, row_id, observed)


def test_no_garage_gives_zero_age_and_flag(out: pd.DataFrame) -> None:
    assert out.loc[2, "HasGarage"] == 0
    assert out.loc[2, "GarageAge"] == 0


def test_garage_without_recorded_year_keeps_age_missing(out: pd.DataFrame) -> None:
    for row_id in (4, 9):
        assert out.loc[row_id, "HasGarage"] == 1
        assert pd.isna(out.loc[row_id, "GarageYrBlt"])  # not invented or repaired
        assert pd.isna(out.loc[row_id, "GarageAge"])  # left for the fitted imputation (M5)
    assert out.loc[9, "GarageCars"] == 0  # unrecorded -> 0 by the Layer 1 column rule
    assert out.loc[9, "GarageArea"] == 0


def test_negative_house_and_remodel_ages_pass_through(out: pd.DataFrame) -> None:
    assert (out.loc[5, ["HouseAge", "RemodAge"]] == [-1, -2]).all()


def _garage_rows(years: list[float], sold: int = 2007) -> pd.DataFrame:
    return pd.DataFrame({"YrSold": [sold] * len(years), "GarageYrBlt": years,
                         "GarageType": ["Attchd"] * len(years)})  # fmt: skip


def test_valid_garage_year_gives_the_expected_age() -> None:
    age = FORMULAS["GarageAge"][1](_garage_rows([1990.0, 2006.0, 2007.0]))
    assert list(age) == [17.0, 1.0, 0.0]  # GarageYrBlt <= YrSold: YrSold - GarageYrBlt


def test_missing_garage_year_keeps_age_missing() -> None:
    age = FORMULAS["GarageAge"][1](_garage_rows([NAN]))
    assert pd.isna(age.iloc[0])  # existing defensive behavior, unchanged


def test_garage_year_after_sale_is_unknown_not_negative(out: pd.DataFrame) -> None:
    """DOC-03 §6.4: GarageYrBlt > YrSold (e.g. the recorded 2207 of Id 2261) is treated as
    unknown, so GarageAge is missing and goes to the fitted imputation."""
    age = FORMULAS["GarageAge"][1](_garage_rows([2008.0, 2207.0]))
    assert age.isna().all()
    assert pd.isna(out.loc[5, "GarageAge"])  # fixture row 5: garage 2009, sold 2007
    assert out.loc[5, "GarageYrBlt"] == 2009  # the recorded value is not repaired
    assert not (out["GarageAge"] < 0).any()


def test_garage_year_after_sale_without_garage_stays_zero() -> None:
    rows = _garage_rows([2207.0]).assign(GarageType=["None"])
    assert FORMULAS["GarageAge"][1](rows).iloc[0] == 0.0  # no garage -> 0, as before


def test_wood_deck_is_not_a_porch(out: pd.DataFrame) -> None:
    assert out.loc[1, "WoodDeckSF"] == 100
    assert out.loc[1, "TotalPorchSF"] == 40


def test_flags_are_zero_or_one(out: pd.DataFrame) -> None:
    for feature in ("IsRemodeled", "HasPool", "HasGarage", "HasBsmt", "HasFireplace", "Has2ndFlr"):
        assert set(out[feature]) <= {0, 1}, feature
        assert out[feature].dtype == "int64", feature


# ------------------------------------------------------------ ordinal and codes


def test_ordinal_map_values(out: pd.DataFrame) -> None:
    """AC-017: None=0, Po=1, Fa=2, TA=3, Gd=4, Ex=5."""
    assert out.loc[2, "BsmtQual"] == 0  # absent -> "None" -> 0
    assert out.loc[2, "HeatingQC"] == 1  # Po
    assert out.loc[2, "ExterCond"] == 2  # Fa
    assert out.loc[1, "GarageQual"] == 3  # TA
    assert out.loc[1, "ExterQual"] == 4  # Gd
    assert out.loc[3, "PoolQC"] == 5  # Ex
    assert out.loc[4, "GarageQual"] == 0  # unrecorded garage field -> "None" -> 0
    for column in ORDINAL:
        assert set(out[column]) <= {0, 1, 2, 3, 4, 5}, column


def test_every_level_of_the_scale_is_exercised(filled: pd.DataFrame) -> None:
    seen = set(pd.unique(filled[ORDINAL].to_numpy().ravel()))
    assert {"None", "Po", "Fa", "TA", "Gd", "Ex"} <= seen


def test_value_outside_the_map_becomes_missing(
    engineer: FeatureEngineer, filled: pd.DataFrame
) -> None:
    odd = filled.copy()
    odd.loc[1, "ExterQual"] = "Excellent"
    assert pd.isna(engineer.fit_transform(odd).loc[1, "ExterQual"])


def test_other_ordered_looking_columns_stay_nominal(
    out: pd.DataFrame, filled: pd.DataFrame
) -> None:
    for column in NOMINAL_ORDERED_LOOKING:
        pd.testing.assert_series_equal(out[column], filled[column])


def test_mssubclass_becomes_text(out: pd.DataFrame, filled: pd.DataFrame) -> None:
    assert filled.loc[1, "MSSubClass"] == 60  # parsed from "060" in M2
    assert out.loc[1, "MSSubClass"] == "60"
    assert out.loc[8, "MSSubClass"] == "190"
    assert pd.api.types.is_string_dtype(out["MSSubClass"])


# ------------------------------------------------------------- estimator behavior


def test_input_is_not_mutated(engineer: FeatureEngineer, filled: pd.DataFrame) -> None:
    before = filled.copy(deep=True)
    engineer.fit(filled).transform(filled)
    pd.testing.assert_frame_equal(filled, before)


def test_output_does_not_depend_on_fit_data(
    engineer: FeatureEngineer, filled: pd.DataFrame
) -> None:
    first = clone(engineer).fit(filled.loc[[1, 2]]).transform(filled)
    second = clone(engineer).fit(filled.loc[[7, 8]]).transform(filled)
    pd.testing.assert_frame_equal(first, second)


def test_fit_learns_no_statistics(engineer: FeatureEngineer, filled: pd.DataFrame) -> None:
    engineer.fit(filled)
    assert {n for n in vars(engineer) if n.endswith("_")} == {"feature_names_in_", "n_features_in_"}


def test_clone_and_names(engineer: FeatureEngineer, filled: pd.DataFrame) -> None:
    copy = clone(engineer)
    out = copy.fit_transform(filled)
    assert list(copy.get_feature_names_out()) == [*filled.columns, *FEATURES]
    assert list(out.columns) == [*filled.columns, *FEATURES]
    assert list(out.index) == list(filled.index)


def test_missing_input_column_is_an_error(engineer: FeatureEngineer, filled: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="YrSold"):
        engineer.fit(filled.drop(columns=["YrSold"]))


def test_unknown_feature_name_is_an_error(filled: pd.DataFrame) -> None:
    engineer = FeatureEngineer(["TotalSF", "LuxuryIndex"], ORDINAL, {"None": 0}, ["MSSubClass"])
    with pytest.raises(ValueError, match="LuxuryIndex"):
        engineer.fit(filled)


# ------------------------------------------------------------------- real dataset


@pytest.mark.data
def test_development_set_through_both_transformers(feature_config: FeatureConfig) -> None:
    from house_price.eda.context import EDAContext

    dev = EDAContext.load(CONFIG_DIR, REPO_ROOT).dev
    model_inputs = [
        s.name for s in load_project_config(CONFIG_DIR, REPO_ROOT).schema.with_role("model_input")
    ]
    frame = dev[model_inputs]
    out = FeatureEngineer.from_config(feature_config).fit_transform(
        SemanticNAFiller.from_config(feature_config).fit_transform(frame)
    )
    assert out.shape == (len(dev), 89)
    filled = (
        feature_config.semantic_fill.categorical_none + feature_config.semantic_fill.numeric_zero
    )
    assert out[filled + ORDINAL].notna().all().all()
    unknown_year = frame["GarageYrBlt"].isna() | (frame["GarageYrBlt"] > frame["YrSold"])
    garage_without_year = (out["HasGarage"] == 1) & unknown_year
    assert out["GarageAge"].isna().equals(garage_without_year)
    assert not (out["GarageAge"] < 0).any()  # Id 2261 (GarageYrBlt 2207) is no longer -200
    assert pd.isna(out.loc[dev["Id"] == 2261, "GarageAge"]).all()

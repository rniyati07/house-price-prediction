"""SemanticNAFiller: Layer 1 semantic filling (FR-011, AC-014, AC-015)."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.base import clone

from house_price.config import FeatureConfig
from house_price.features.semantic import SemanticNAFiller

# The complete lists from DOC-03 §6.2 / DOC-01 FR-011.
DOC03_CATEGORICAL = {
    "PoolQC", "Alley", "Fence", "FireplaceQu", "MiscFeature", "GarageType", "GarageFinish",
    "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
    "BsmtFinType2", "MasVnrType",
}  # fmt: skip
DOC03_NUMERIC = {
    "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
}  # fmt: skip


@pytest.fixture
def filler(feature_config: FeatureConfig) -> SemanticNAFiller:
    return SemanticNAFiller.from_config(feature_config)


def test_configured_lists_are_the_doc03_lists(feature_config: FeatureConfig) -> None:
    assert set(feature_config.semantic_fill.categorical_none) == DOC03_CATEGORICAL
    assert set(feature_config.semantic_fill.numeric_zero) == DOC03_NUMERIC


def test_fixture_contains_every_absent_case(feature_rows: pd.DataFrame) -> None:
    """AC-014 precondition: every listed column has at least one missing value."""
    for column in DOC03_CATEGORICAL | DOC03_NUMERIC:
        assert feature_rows[column].isna().any(), column


def test_absent_categoricals_become_none(
    filler: SemanticNAFiller, feature_rows: pd.DataFrame
) -> None:
    out = filler.fit_transform(feature_rows)
    for column in DOC03_CATEGORICAL:
        was_missing = feature_rows[column].isna()
        assert (out.loc[was_missing, column] == "None").all(), column
        assert out[column].notna().all(), column


def test_absent_numerics_become_zero(filler: SemanticNAFiller, feature_rows: pd.DataFrame) -> None:
    out = filler.fit_transform(feature_rows)
    for column in DOC03_NUMERIC:
        was_missing = feature_rows[column].isna()
        assert (out.loc[was_missing, column] == 0).all(), column
        assert out[column].notna().all(), column


def test_recorded_values_and_other_columns_are_unchanged(
    filler: SemanticNAFiller, feature_rows: pd.DataFrame
) -> None:
    out = filler.fit_transform(feature_rows)
    listed = DOC03_CATEGORICAL | DOC03_NUMERIC
    for column in feature_rows.columns:
        present = feature_rows[column].notna()
        pd.testing.assert_series_equal(out.loc[present, column], feature_rows.loc[present, column])
        if column not in listed:
            pd.testing.assert_series_equal(out[column], feature_rows[column])
    assert out.loc[4, "MasVnrType"] == "None"  # literal text "None" is kept as recorded
    assert pd.isna(out.loc[4, "GarageYrBlt"])  # GarageYrBlt is never filled (ADR-05)


def test_basement_anomaly_patterns_get_the_normal_rule(
    filler: SemanticNAFiller, feature_rows: pd.DataFrame
) -> None:
    """DOC-02 §6.6: rule by column, no special case for rows where a basement exists."""
    out = filler.fit_transform(feature_rows)
    assert out.loc[6, "BsmtExposure"] == "None"  # basement exists, exposure unrecorded
    assert out.loc[7, "BsmtFinType2"] == "None"  # BsmtFinSF2 = 479, type unrecorded
    assert out.loc[7, "BsmtFinSF2"] == 479
    assert (
        out.loc[8, sorted(DOC03_NUMERIC - {"GarageArea", "GarageCars", "MasVnrArea"})] == 0
    ).all()


def test_input_is_not_mutated(filler: SemanticNAFiller, feature_rows: pd.DataFrame) -> None:
    before = feature_rows.copy(deep=True)
    filler.fit(feature_rows).transform(feature_rows)
    pd.testing.assert_frame_equal(feature_rows, before)


def test_output_does_not_depend_on_fit_data(
    filler: SemanticNAFiller, feature_rows: pd.DataFrame
) -> None:
    first = clone(filler).fit(feature_rows.loc[[1, 2, 3]]).transform(feature_rows)
    second = clone(filler).fit(feature_rows.loc[[6, 7, 8]]).transform(feature_rows)
    unfitted_order = filler.fit(feature_rows).transform(feature_rows)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, unfitted_order)


def test_fit_learns_no_statistics(filler: SemanticNAFiller, feature_rows: pd.DataFrame) -> None:
    filler.fit(feature_rows)
    learned = {name for name in vars(filler) if name.endswith("_")}
    assert learned == {"feature_names_in_", "n_features_in_"}


def test_clone_and_names(filler: SemanticNAFiller, feature_rows: pd.DataFrame) -> None:
    copy = clone(filler)
    assert copy.get_params() == filler.get_params()
    out = copy.fit_transform(feature_rows)
    assert list(copy.get_feature_names_out()) == list(feature_rows.columns)
    assert list(out.columns) == list(feature_rows.columns)
    assert list(out.index) == list(feature_rows.index)


def test_missing_configured_column_is_an_error(
    filler: SemanticNAFiller, feature_rows: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="PoolQC"):
        filler.fit(feature_rows.drop(columns=["PoolQC"]))

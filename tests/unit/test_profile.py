"""Profiling: duplicate detection, missing values, column profile, quality flags, metadata."""

from __future__ import annotations

import pandas as pd
import pytest

from house_price.config import ProjectConfig
from house_price.data.load import load_raw, sha256_file
from house_price.data.profile import (
    TARGET_WITHHELD,
    column_profile,
    dataset_metadata,
    detect_duplicates,
    missing_value_profile,
    quality_flags,
)


@pytest.fixture
def frame(sample_config: ProjectConfig) -> pd.DataFrame:
    return load_raw(sample_config)


# ---------------------------------------------------------------- duplicate detection


def test_fixture_has_no_duplicate_ids_or_records(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    report = detect_duplicates(frame, sample_config.schema)
    assert report.n_rows == len(frame)
    assert report.n_duplicate_ids == 0 and not report.has_duplicate_ids
    assert report.duplicate_secondary_identifiers == {"PID": 0}
    assert report.n_exact_duplicate_rows == 0


def test_detects_feature_duplicates_with_conflicting_prices(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    """Ids 2508 and 2509 share every feature but sold at different prices."""
    report = detect_duplicates(frame, sample_config.schema)
    assert report.n_duplicate_feature_rows == 1
    assert report.duplicate_feature_row_ids == [2509]
    assert report.n_conflicting_target_groups == 1


def test_detects_exact_duplicate_record(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    copy = frame.iloc[[0]].copy()
    copy["Id"] = 99_999
    copy["PID"] = "0999999999"
    report = detect_duplicates(pd.concat([frame, copy], ignore_index=True), sample_config.schema)
    assert report.n_exact_duplicate_rows == 1
    assert report.exact_duplicate_row_ids == [99_999]
    assert report.has_duplicate_rows


def test_detects_duplicate_identifiers(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    frame.loc[1, "Id"] = frame.loc[0, "Id"]
    frame.loc[3, "PID"] = frame.loc[2, "PID"]
    report = detect_duplicates(frame, sample_config.schema)
    assert report.n_duplicate_ids == 1
    assert report.duplicate_ids == [int(frame.loc[0, "Id"])]
    assert report.duplicate_secondary_identifiers == {"PID": 1}


def test_duplicate_detection_never_modifies_the_frame(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    before = frame.copy()
    detect_duplicates(frame, sample_config.schema)
    pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------- missing values


def test_missing_profile_counts_and_order(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    profile = missing_value_profile(frame, sample_config.schema)
    expected = frame.isna().sum()
    expected = expected[expected > 0]
    assert set(profile["column"]) == set(expected.index)
    for row in profile.itertuples():
        assert row.n_missing == expected[row.column]
        assert row.pct_missing == pytest.approx(100 * expected[row.column] / len(frame), abs=1e-3)
    assert profile["n_missing"].is_monotonic_decreasing
    assert profile["nullable"].all()  # only nullable columns may contain missing values


# ------------------------------------------------------------------- column profile


def test_column_profile_covers_every_column(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    profile = column_profile(frame, sample_config.schema).set_index("column")
    assert list(profile.index) == sample_config.schema.names
    assert profile.loc["GrLivArea", "max"] == frame["GrLivArea"].max()
    assert profile.loc["Neighborhood", "n_unique"] == frame["Neighborhood"].nunique()
    assert profile.loc["Neighborhood", "top_count"] == frame["Neighborhood"].value_counts().iloc[0]
    assert "TwnhsI" in profile.loc["BldgType", "unobserved_allowed"].split("|")


def test_column_profile_withholds_target_distribution(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    """FR-009: target analysis belongs to the development set, not the raw file."""
    row = column_profile(frame, sample_config.schema).set_index("column").loc["SalePrice"]
    assert row["note"] == TARGET_WITHHELD
    for stat in ("min", "median", "mean", "max", "std"):
        assert row[stat] is None or pd.isna(row[stat])


# ------------------------------------------------------------------- quality flags


def test_quality_flags_find_documented_anomalies(frame: pd.DataFrame) -> None:
    flags = {flag.name: flag for flag in quality_flags(frame, "Id", "GrLivArea", 4000)}
    assert flags["basement_exposure_missing_with_basement"].ids == [67]
    assert flags["basement_fintype2_missing_with_area"].ids == [445]
    assert flags["basement_areas_unrecorded"].ids == [1342]
    assert flags["garage_partially_recorded"].ids == [1357, 2237]
    assert 2261 in flags["garage_built_after_sale"].ids
    assert flags["electrical_unrecorded"].ids == [1578]
    assert flags["out_of_scope_rows"].ids == [1499, 1768]
    assert flags["grlivarea_components_differ"].status == "clear"


def test_quality_flags_skip_rules_whose_columns_are_absent(frame: pd.DataFrame) -> None:
    flags = {f.name: f for f in quality_flags(frame.drop(columns=["PoolQC"]), "Id")}
    assert flags["pool_area_without_quality"].status == "skipped"
    assert "out_of_scope_rows" not in flags


def test_quality_flags_do_not_modify_values(frame: pd.DataFrame) -> None:
    before = frame.copy()
    quality_flags(frame, "Id", "GrLivArea", 4000)
    pd.testing.assert_frame_equal(frame, before)


# ------------------------------------------------------------------------ metadata


def test_dataset_metadata(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    raw_sha = sha256_file(sample_config.raw_path)
    metadata = dataset_metadata(frame, sample_config, raw_sha)
    assert metadata.raw_sha256 == raw_sha == sample_config.data.raw_sha256
    assert metadata.raw_path == "data/raw/train.csv"
    assert (metadata.n_rows, metadata.n_columns) == frame.shape
    assert metadata.size_bytes == sample_config.raw_path.stat().st_size
    assert metadata.columns_by_role == {
        "identifier": 2,
        "model_input": 77,
        "excluded": 2,
        "target": 1,
    }
    assert sum(metadata.columns_by_dtype.values()) == 82
    assert metadata.total_missing_cells == int(frame.isna().to_numpy().sum())
    assert metadata.config_hash == sample_config.config_hash
    assert len(metadata.schema_hash) == 64
    assert {"python", "pandas", "pandera"} <= set(metadata.library_versions)

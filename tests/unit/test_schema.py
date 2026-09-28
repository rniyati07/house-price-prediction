"""Ingestion schema: the five rejection cases and the shared configuration source
(AC-004, AC-005)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from house_price.config import ProjectConfig
from house_price.data.errors import DataValidationError
from house_price.data.load import cast_to_schema, load_raw, read_text_csv, rename_source_columns
from house_price.data.schema import (
    allowed_values,
    build_inference_schema,
    build_ingestion_schema,
    model_input_columns,
    select_model_input,
    validate_frame,
    validate_raw,
)


@pytest.fixture
def frame(sample_config: ProjectConfig) -> pd.DataFrame:
    return load_raw(sample_config)


def _failures(exc: DataValidationError) -> set[tuple[str, str]]:
    return {(str(r.column), str(r.check)) for r in exc.failure_cases.itertuples()}


def _assert_rejected(
    frame: pd.DataFrame, config: ProjectConfig, column: str, check_fragment: str
) -> DataValidationError:
    with pytest.raises(DataValidationError) as info:
        validate_raw(frame, config.schema)
    failures = _failures(info.value)
    assert any(c == column and check_fragment in chk for c, chk in failures), failures
    assert column in str(info.value) and check_fragment in str(info.value)
    return info.value


def test_fixture_passes_the_ingestion_schema(frame: pd.DataFrame) -> None:
    assert len(frame) == 102


# -------------------------------------------------------- AC-004: five rejection cases


def test_rejects_missing_expected_column(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    _assert_rejected(
        frame.drop(columns=["LotArea"]), sample_config, "LotArea", "column_in_dataframe"
    )


def test_rejects_non_numeric_value_in_numeric_column(sample_config: ProjectConfig) -> None:
    text = rename_source_columns(
        read_text_csv(sample_config.raw_path, sample_config.data.missing_tokens),
        sample_config.schema,
    )
    text.loc[3, "LotArea"] = "abc"
    cast = cast_to_schema(text, sample_config.schema)
    with pytest.raises(DataValidationError) as info:
        validate_raw(cast.frame, sample_config.schema, cast_failures=cast.failures)
    cases = info.value.failure_cases
    hit = cases[(cases["column"] == "LotArea") & (cases["check"] == "cast_to_int")]
    assert hit["index"].tolist() == [3]
    assert hit["failure_case"].tolist() == ["abc"]
    assert "cast_to_int" in str(info.value)


def test_rejects_category_not_in_allowed_list(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    frame.loc[0, "Neighborhood"] = "Atlantis"
    exc = _assert_rejected(frame, sample_config, "Neighborhood", "isin")
    assert "Atlantis" in exc.failure_cases["failure_case"].astype(str).tolist()


def test_rejects_duplicate_id(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    frame.loc[1, "Id"] = frame.loc[0, "Id"]
    _assert_rejected(frame, sample_config, "Id", "field_uniqueness")


@pytest.mark.parametrize("price", [0, -100])
def test_rejects_non_positive_sale_price(
    frame: pd.DataFrame, sample_config: ProjectConfig, price: int
) -> None:
    frame.loc[0, "SalePrice"] = price
    _assert_rejected(frame, sample_config, "SalePrice", "in_range")


# ------------------------------------------------------------- further contract checks


def test_rejects_null_in_non_nullable_column(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    frame.loc[0, "Street"] = np.nan
    _assert_rejected(frame, sample_config, "Street", "not_nullable")


def test_rejects_out_of_range_value(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    frame.loc[0, "YrSold"] = 2011
    _assert_rejected(frame, sample_config, "YrSold", "in_range")


def test_rejects_unexpected_extra_column(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    frame["Surprise"] = 1
    _assert_rejected(frame, sample_config, "Surprise", "column_in_schema")


def test_rejects_malformed_parcel_id(frame: pd.DataFrame, sample_config: ProjectConfig) -> None:
    frame.loc[0, "PID"] = "12345"
    _assert_rejected(frame, sample_config, "PID", "str_matches")


def test_lazy_validation_reports_every_failure(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    frame.loc[0, "Neighborhood"] = "Atlantis"
    frame.loc[5, "SalePrice"] = 0
    frame = frame.drop(columns=["LotArea"])
    with pytest.raises(DataValidationError) as info:
        validate_raw(frame, sample_config.schema)
    columns = {column for column, _ in _failures(info.value)}
    assert {"Neighborhood", "SalePrice", "LotArea"} <= columns
    summary = info.value.summary()
    assert set(summary.columns) == {"column", "check", "n_failures", "example_indices"}


# ------------------------------------------------ AC-005: one configuration source


def test_ingestion_and_inference_schemas_share_allowed_values(
    sample_config: ProjectConfig,
) -> None:
    ingestion = allowed_values(build_ingestion_schema(sample_config.schema))
    inference = allowed_values(build_inference_schema(sample_config.schema))
    assert inference  # every model-input categorical is present
    for column, values in inference.items():
        assert values == ingestion[column], column
        assert values == list(sample_config.schema.by_name(column).allowed_values or [])


def test_changing_config_changes_both_schemas(sample_config: ProjectConfig) -> None:
    schema = sample_config.schema
    columns = [
        spec.model_copy(update={"allowed_values": [*spec.allowed_values, "NewCode"]})
        if spec.name == "Neighborhood" and spec.allowed_values
        else spec
        for spec in schema.columns
    ]
    changed = schema.model_copy(update={"columns": columns})
    assert "NewCode" in allowed_values(build_ingestion_schema(changed))["Neighborhood"]
    assert "NewCode" in allowed_values(build_inference_schema(changed))["Neighborhood"]
    ingestion_ranges = build_ingestion_schema(changed).columns["GrLivArea"].checks
    inference_ranges = build_inference_schema(changed).columns["GrLivArea"].checks
    assert [c.statistics for c in ingestion_ranges] == [c.statistics for c in inference_ranges]


def test_inference_schema_is_the_model_input_subset(
    frame: pd.DataFrame, sample_config: ProjectConfig
) -> None:
    schema = sample_config.schema
    inputs = model_input_columns(schema)
    assert len(inputs) == 77
    assert not {"Id", "PID", "SalePrice", "SaleType", "SaleCondition"} & set(inputs)
    assert list(build_inference_schema(schema).columns) == inputs
    subset = select_model_input(frame, schema)
    assert list(subset.columns) == inputs
    validate_frame(subset, build_inference_schema(schema))

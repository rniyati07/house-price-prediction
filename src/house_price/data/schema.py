"""Pandera schemas generated from ``configs/schema.yaml`` (DOC-03 §5.3, FR-004, FR-005).

``schema.yaml`` is the single source of the data contract. The ingestion schema (all
columns) and the inference schema (the model-input columns reused by M11) are built from
the same column specifications, so they always share allowed values and ranges (AC-005).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from house_price.config import ColumnSpec, SchemaConfig
from house_price.data.errors import FAILURE_COLUMNS, DataValidationError

_PANDAS_DTYPES = {"int": "int64", "float": "float64", "category": str, "string": str}


def _checks(spec: ColumnSpec) -> list[pa.Check]:
    checks: list[pa.Check] = []
    if spec.allowed_values:
        checks.append(pa.Check.isin(list(spec.allowed_values)))
    if spec.min is not None and spec.max is not None:
        checks.append(pa.Check.in_range(spec.min, spec.max, include_min=not spec.strict_min))
    elif spec.min is not None:
        checks.append(pa.Check.gt(spec.min) if spec.strict_min else pa.Check.ge(spec.min))
    elif spec.max is not None:
        checks.append(pa.Check.le(spec.max))
    if spec.pattern is not None:
        checks.append(pa.Check.str_matches(spec.pattern))
    return checks


def build_column(spec: ColumnSpec) -> pa.Column:
    """Translate one ``schema.yaml`` entry into a Pandera column."""
    return pa.Column(
        _PANDAS_DTYPES[spec.dtype],
        checks=_checks(spec),
        nullable=spec.nullable,
        unique=spec.unique,
        coerce=False,
        required=True,
        description=spec.note,
    )


def build_ingestion_schema(schema: SchemaConfig) -> pa.DataFrameSchema:
    """All columns, no extras (FR-004): presence, dtype, allowed values, range,
    nullability, unique identifiers, and ``SalePrice > 0``."""
    return pa.DataFrameSchema(
        {spec.name: build_column(spec) for spec in schema.columns},
        strict=True,
        name="ingestion",
    )


def model_input_columns(schema: SchemaConfig) -> list[str]:
    """The model-input columns in schema order (DN-11). Identifiers, the target, and the
    excluded transaction-outcome columns are never model inputs."""
    return [spec.name for spec in schema.with_role("model_input")]


def build_inference_schema(schema: SchemaConfig) -> pa.DataFrameSchema:
    """The model-input subset, built from the same specifications (reused by M11)."""
    return pa.DataFrameSchema(
        {spec.name: build_column(spec) for spec in schema.with_role("model_input")},
        strict=True,
        name="inference",
    )


def select_model_input(frame: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """Return exactly the model-input columns, in schema order."""
    return frame.loc[:, model_input_columns(schema)]


def allowed_values(pandera_schema: pa.DataFrameSchema) -> dict[str, list[object]]:
    """Allowed categories per column as exposed by a built Pandera schema."""
    result: dict[str, list[object]] = {}
    for name, column in pandera_schema.columns.items():
        for check in column.checks:
            if check.name == "isin":
                result[name] = list(check.statistics["allowed_values"])
    return result


def validate_frame(
    frame: pd.DataFrame,
    pandera_schema: pa.DataFrameSchema,
    cast_failures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Validate lazily: every failing check is collected and reported together.

    Raises ``DataValidationError`` carrying one row per failure (column, check, row index,
    failing value), including casting failures from the loader.
    """
    failures: list[pd.DataFrame] = []
    if cast_failures is not None and not cast_failures.empty:
        failures.append(cast_failures.loc[:, FAILURE_COLUMNS])
    try:
        validated = pandera_schema.validate(frame, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.copy()
        # Frame-level failures (a missing or unexpected column) carry the schema name in
        # ``column`` and the offending column in ``failure_case``: report the column itself.
        frame_level = cases["schema_context"] == "DataFrameSchema"
        cases.loc[frame_level, "column"] = cases.loc[frame_level, "failure_case"].astype(str)
        failures.append(cases.loc[:, FAILURE_COLUMNS])
        validated = frame
    if failures:
        raise DataValidationError(pd.concat(failures, ignore_index=True))
    return validated


def validate_raw(
    frame: pd.DataFrame, schema: SchemaConfig, cast_failures: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Validate a cast raw frame against the ingestion schema."""
    return validate_frame(frame, build_ingestion_schema(schema), cast_failures)

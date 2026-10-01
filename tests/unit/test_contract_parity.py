"""Pydantic / Pandera contract parity (DOC-04 §9.3, §16.2; FR-005, FR-044, AC-005, AC-063;
DOC-05 M11-6).

The API's generated request model and the batch CLI's validation (DN-18 parsing, casting,
the Pandera inference schema) must accept exactly the same valid rows and reject exactly the
same single-fault rows: one fault per rule (missing column, wrong type, disallowed value, out
of range, ``null`` in a non-nullable column), applied to every column the rule applies to.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest
from pydantic import BaseModel, ValidationError

from house_price.api.schemas import MAX_BATCH_SIZE, build_request_models, field_name
from house_price.cli import check_input
from house_price.config import ColumnSpec, SchemaConfig, load_project_config
from house_price.data.load import read_text_csv
from house_price.data.schema import build_inference_schema, model_input_columns
from house_price.persistence import metadata
from tests.conftest import (
    API_EXAMPLE,
    CONFIG_DIR,
    CONSISTENCY_CSV,
    REPO_ROOT,
    as_json_records,
    consistency_frame,
)


@pytest.fixture(scope="module")
def schema() -> SchemaConfig:
    return load_project_config(CONFIG_DIR, REPO_ROOT).schema


@pytest.fixture(scope="module")
def property_input(schema: SchemaConfig) -> type[BaseModel]:
    return build_request_models(schema).property_input


def pydantic_accepts(model: type[BaseModel], record: dict[str, Any]) -> bool:
    try:
        model.model_validate_json(json.dumps(record))
    except ValidationError:
        return False
    return True


def text_row() -> dict[str, str]:
    """The first fixture row as CSV text (the CLI's input form)."""
    frame = read_text_csv(CONSISTENCY_CSV, ["NA", ""])  # DN-18, as the CLI reads it
    return {str(k): v for k, v in frame.iloc[0].items() if k != "Id"}


def json_row() -> dict[str, Any]:
    return as_json_records(consistency_frame().head(1))[0]


def _number(spec: ColumnSpec, value: float) -> Any:
    return int(value) if spec.dtype == "int" else float(value)


def faults(schema: SchemaConfig) -> list[tuple[str, str, Any, str]]:
    """(rule, column, JSON value, CSV text) for every applicable single-cell fault."""
    cases: list[tuple[str, str, Any, str]] = []
    for spec in schema.with_role("model_input"):
        if not spec.nullable:
            cases.append(("null_in_non_nullable", spec.name, None, "NA"))
        if spec.dtype in ("int", "float") and not spec.allowed_values:
            cases.append(("wrong_type", spec.name, "abc", "abc"))
        if spec.allowed_values:
            bad: Any = 999_999 if spec.dtype == "int" else "NotAllowed"
            cases.append(("disallowed_value", spec.name, bad, str(bad)))
        elif spec.dtype in ("int", "float"):
            if spec.max is not None:
                value = _number(spec, spec.max + 1)
                cases.append(("above_max", spec.name, value, str(value)))
            if spec.min is not None:
                value = _number(spec, spec.min if spec.strict_min else spec.min - 1)
                cases.append(("below_min", spec.name, value, str(value)))
    return cases


# ------------------------------------------------------------------ the 77 columns


def test_request_fields_equal_model_input_schema(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    """FR-044: exactly the model-input columns of ``metadata.input_schema``, in schema order."""
    aliases = [f.alias for f in property_input.model_fields.values()]
    expected = [c.name for c in metadata.input_schema(schema)]
    assert aliases == expected == model_input_columns(schema)
    assert len(aliases) == 77
    assert list(property_input.model_fields) == [field_name(c) for c in expected]
    assert all(f.is_required() for f in property_input.model_fields.values())  # SD-01
    assert set(build_inference_schema(schema).columns) == set(aliases)
    for excluded in ("Id", "PID", "SaleType", "SaleCondition", "SalePrice"):
        assert excluded not in aliases


def test_nullable_fields_match_the_schema(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    nullable = {s.name for s in schema.with_role("model_input") if s.nullable}
    assert len(nullable) == 27
    accepted_null = {
        name for name in nullable
        if pydantic_accepts(property_input, {**json_row(), name: None})
    }  # fmt: skip
    assert accepted_null == nullable


def test_field_names_are_sanitized_with_canonical_aliases() -> None:
    assert field_name("1stFlrSF") == "f_1stFlrSF" and field_name("3SsnPorch") == "f_3SsnPorch"


def test_example_validates(property_input: type[BaseModel]) -> None:
    example = json.loads(API_EXAMPLE.read_text(encoding="utf-8"))
    assert list(example) == [f.alias for f in property_input.model_fields.values()]
    assert pydantic_accepts(property_input, example)


# ------------------------------------------------------------------------- parity


def test_valid_rows_accepted_by_both(schema: SchemaConfig, property_input: type[BaseModel]) -> None:
    for record in as_json_records(consistency_frame()):
        assert pydantic_accepts(property_input, record)
    assert check_input(read_text_csv(CONSISTENCY_CSV, ["NA", ""]), schema).problems.empty


def test_single_fault_rows_rejected_by_both(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    cases = faults(schema)
    rules = {rule for rule, *_ in cases}
    assert rules == {"null_in_non_nullable", "wrong_type", "disallowed_value", "above_max",
                     "below_min"}  # fmt: skip
    base_json, base_text = json_row(), text_row()
    # Pydantic: each fault on its own request.
    accepted = [(r, c) for r, c, value, _ in cases
                if pydantic_accepts(property_input, {**base_json, c: value})]  # fmt: skip
    assert accepted == []
    # CLI / Pandera: each fault on its own row of one lazily validated file; every row
    # must be reported (the batch-size finding for this oversized file has no row index).
    text = pd.DataFrame([{**base_text, column: csv} for _, column, _, csv in cases], dtype=str)
    text = text.replace({"NA": None})
    problems = check_input(text, schema).problems
    flagged = set(problems["index"].dropna().astype(int))
    missed = [(cases[i][0], cases[i][1]) for i in range(len(cases)) if i not in flagged]
    assert missed == []
    assert len(text) > MAX_BATCH_SIZE  # many faults in one file: the size finding too
    assert problems["check"].str.startswith("batch_size").sum() == 1


def test_missing_columns_rejected_by_both(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    columns = model_input_columns(schema)
    base_json = json_row()
    for column in columns:
        assert not pydantic_accepts(property_input, {k: v for k, v in base_json.items()
                                                     if k != column}), column  # fmt: skip
    text = pd.DataFrame([text_row()], dtype=str)
    for column in columns:
        problems = check_input(text.drop(columns=[column]), schema).problems
        assert list(problems["column"]) == [column], column
        assert list(problems["check"]) == ["column_in_dataframe"]


def test_extra_columns_rejected_by_both(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    assert not pydantic_accepts(property_input, {**json_row(), "Colour": "blue"})
    text = pd.DataFrame([{**text_row(), "Colour": "blue"}], dtype=str)
    assert list(check_input(text, schema).problems["check"]) == ["unexpected_column"]


def test_unseen_but_allowed_category_accepted_by_both(
    schema: SchemaConfig, property_input: type[BaseModel]
) -> None:
    """A category allowed by the schema (here ``Utilities = NoSeWa``) is valid input even if
    a model never saw it; the pipeline handles it by design (NFR-006)."""
    assert "NoSeWa" in (schema.by_name("Utilities").allowed_values or [])
    assert pydantic_accepts(property_input, {**json_row(), "Utilities": "NoSeWa"})
    text = pd.DataFrame([{**text_row(), "Utilities": "NoSeWa"}], dtype=str)
    assert check_input(text, schema).problems.empty

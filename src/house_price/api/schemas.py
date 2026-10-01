"""Request and response models (DOC-04 §6, §7; SD-01 to SD-04; FR-044, FR-005).

The request models are generated at startup from ``schema.yaml`` by
:func:`build_request_models`; there is no hand-written field list anywhere in the API, so the
API contract cannot drift from the training contract (DOC-04 §7.1).

- One field per model-input column, in schema order. The internal name is a sanitized
  identifier (``f_1stFlrSF``); the alias is the canonical column name, used in JSON (SD-02).
- Every field is required; nullable columns are typed ``T | None`` but still required
  (SD-01): absence must be stated with an explicit ``null``.
- ``int`` columns accept JSON integers only; ``float`` columns accept JSON numbers but never
  strings, booleans or non-finite values (SD-03). Ranges come from ``schema.yaml``.
- Categoricals are closed ``Literal`` sets, case-sensitive; integer codes (``MSSubClass``)
  also accept integers only.
- Unknown fields are rejected (``extra="forbid"``), at the top level of a batch too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    create_model,
)
from pydantic_core import PydanticCustomError

from house_price.config import ColumnSpec, SchemaConfig

# Project requirement change (M11): the batch endpoint and the batch CLI accept at most 20
# properties per request / file (DOC-04 originally specified 100).
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 20

_NOT_IDENTIFIER = re.compile(r"\W")


def field_name(column: str) -> str:
    """The internal Python name of a column (``1stFlrSF`` -> ``f_1stFlrSF``)."""
    return "f_" + _NOT_IDENTIFIER.sub("_", column)


def _integer_only(value: Any) -> Any:
    """SD-03 for integer codes: ``Literal[20, 30, ...]`` alone would accept ``20.0``."""
    if type(value) is not int:
        raise PydanticCustomError("int_type", "Input should be a valid integer")
    return value


def _field_type(spec: ColumnSpec) -> Any:
    bounds: dict[str, Any] = {}
    if spec.min is not None:
        bounds["gt" if spec.strict_min else "ge"] = spec.min
    if spec.max is not None:
        bounds["le"] = spec.max
    if spec.allowed_values:
        choices: Any = Literal.__getitem__(tuple(spec.allowed_values))
        if spec.dtype == "int":
            return Annotated[choices, BeforeValidator(_integer_only)]
        return choices
    if spec.dtype == "int":
        return Annotated[StrictInt, Field(**bounds)]
    if spec.dtype == "float":
        return Annotated[StrictFloat, Field(allow_inf_nan=False, **bounds)]
    if spec.pattern is not None:
        return Annotated[StrictStr, Field(pattern=spec.pattern)]
    return StrictStr


@dataclass(frozen=True)
class RequestModels:
    property_input: type[BaseModel]
    batch_input: type[BaseModel]


def build_request_models(
    schema: SchemaConfig, example: dict[str, Any] | None = None
) -> RequestModels:
    """``PropertyInput`` (the model-input columns) and ``BatchInput`` (1 to
    ``MAX_BATCH_SIZE`` properties), with ``example`` attached to the OpenAPI schema."""
    fields: dict[str, Any] = {}
    for spec in schema.with_role("model_input"):
        annotation = _field_type(spec)
        if spec.nullable:
            annotation = annotation | None
        fields[field_name(spec.name)] = (annotation, Field(alias=spec.name, description=spec.note))
    extra: dict[str, Any] = {"examples": [example]} if example is not None else {}
    property_input: type[BaseModel] = create_model(
        "PropertyInput",
        __config__=ConfigDict(extra="forbid", json_schema_extra=extra),
        __doc__="One property: every model-input column of schema.yaml, nullable ones as null.",
        **fields,
    )
    batch_extra: dict[str, Any] = (
        {"examples": [{"properties": [example]}]} if example is not None else {}
    )
    batch_input: type[BaseModel] = create_model(
        "BatchInput",
        __config__=ConfigDict(extra="forbid", json_schema_extra=batch_extra),
        __doc__=f"{MIN_BATCH_SIZE} to {MAX_BATCH_SIZE} properties, predicted in input order.",
        properties=(
            list[property_input],  # type: ignore[valid-type]
            Field(
                min_length=MIN_BATCH_SIZE,
                max_length=MAX_BATCH_SIZE,
                description=f"{MIN_BATCH_SIZE} to {MAX_BATCH_SIZE} properties (inclusive).",
            ),
        ),
    )
    return RequestModels(property_input, batch_input)


# ----------------------------------------------------------------------------- responses


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(_Response):
    status: Literal["ok"] = "ok"
    model_version: str


class PredictionResponse(_Response):
    predicted_price: float = Field(gt=0, description="Estimated price in US dollars")
    model_version: str
    out_of_domain: bool = Field(
        description="True if GrLivArea exceeds metadata.scope_rule.max_in_domain"
    )


class BatchPrediction(_Response):
    index: int = Field(ge=0, description="0-based position of the property in the request")
    predicted_price: float = Field(gt=0, description="Estimated price in US dollars")
    model_version: str
    out_of_domain: bool


class BatchPredictionResponse(_Response):
    count: int = Field(ge=MIN_BATCH_SIZE, le=MAX_BATCH_SIZE)
    predictions: list[BatchPrediction]


class InternalErrorResponse(_Response):
    detail: Literal["Internal server error"] = "Internal server error"
    request_id: str

"""The prediction path (DOC-04 §8) and the verified model context it runs on (DOC-04 §5.2).

Serving calls ``pipeline.predict()`` on a correctly built input frame and nothing else
(DOC-04 §2): missing-value semantics, feature engineering, preprocessing and the inverse
target transform all run inside the persisted M10 artifact. This module only

- builds the verified :class:`ModelContext` (startup steps 3 to 11, shared by the service
  and the batch CLI), reusing the M10 metadata model and verified load;
- builds the frame (schema column order, ``None`` -> ``NaN``, the training dtypes);
- applies the SD-16 guard and the ``out_of_domain`` flag from ``metadata.scope_rule``;
- assembles the responses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from house_price.api.logging import log_event, set_model_version
from house_price.api.schemas import (
    BatchPrediction,
    BatchPredictionResponse,
    PredictionResponse,
    RequestModels,
    build_request_models,
)
from house_price.config import SchemaConfig, load_model
from house_price.data.schema import model_input_columns
from house_price.persistence import artifact, metadata
from house_price.persistence.artifact import ArtifactError
from house_price.persistence.metadata import ArtifactMetadata

EXAMPLE_FILE = "api_example.json"  # SD-15, in the configuration directory
SCHEMA_FILE = "schema.yaml"


class StartupError(RuntimeError):
    """A startup verification step failed; the service must not start (SD-13)."""

    def __init__(self, step: str, reason: str) -> None:
        self.step = step
        self.reason = reason
        super().__init__(f"startup step {step!r} failed: {reason}")


class GuardViolation(RuntimeError):
    """SD-16: a prediction that is not finite and positive is never returned."""

    def __init__(self, rows: Sequence[int]) -> None:
        self.rows = list(rows)
        super().__init__(f"prediction guard violated for row(s) {self.rows}")


@dataclass(frozen=True)
class ModelContext:
    """Everything a verified service holds (DOC-04 §5.2 step 12)."""

    model: Any
    metadata: ArtifactMetadata
    schema: SchemaConfig
    columns: list[str]
    models: RequestModels
    example: dict[str, Any] | None


@dataclass(frozen=True)
class Predictions:
    prices: list[float]
    out_of_domain: list[bool]


# ------------------------------------------------------------------------ startup


@contextmanager
def _step(logger: logging.Logger, name: str) -> Iterator[None]:
    try:
        yield
    except StartupError:
        raise
    except Exception as exc:  # every failure stops startup, logged first
        reason = str(exc) or type(exc).__name__
        log_event(logger, logging.CRITICAL, "startup.failed", step=name, reason=reason,
                  error_type=type(exc).__name__)  # fmt: skip
        raise StartupError(name, reason) from exc


def load_model_context(
    model_dir: Path,
    config_dir: Path,
    *,
    allow_non_release: bool,
    logger: logging.Logger,
    warm_up: bool = True,
) -> ModelContext:
    """DOC-04 §5.2 steps 3 to 11 (the batch CLI runs 3 to 10: ``warm_up=False``).

    Nothing is unpickled before the release check, the SHA-256 check and the library
    versions have passed (``artifact.verify``, then ``artifact.load_verified``, which
    re-checks immediately before ``joblib.load``)."""
    require_release = not allow_non_release
    with _step(logger, "load_metadata"):  # 3: the same Pydantic model M10 writes with
        meta = artifact.read_metadata(model_dir)
    set_model_version(meta.model_version)
    with _step(logger, "release_check"):  # 4
        if meta.artifact_role != "production":
            raise ArtifactError(f"{model_dir} holds a {meta.artifact_role} artifact; only the "
                                "production artifact of the selected model can be served")  # fmt: skip
        if require_release and not meta.is_release:
            raise ArtifactError(f"{model_dir} is not a released artifact (is_release=false); "
                                "HPP_ALLOW_NON_RELEASE=true is for CI and smoke tests only")  # fmt: skip
    try:  # 5 and 6: SHA-256, then library versions (M10 verified-load checks)
        artifact.verify(model_dir, require_release=require_release)
    except ArtifactError as exc:
        step = "environment_check" if "library versions" in str(exc) else "artifact_integrity"
        with _step(logger, step):
            raise
    log_event(logger, logging.INFO, "artifact.verified", model_sha256=meta.model_sha256,
              is_release=meta.is_release)  # fmt: skip
    log_event(logger, logging.INFO, "environment.verified", library_versions=meta.library_versions)
    with _step(logger, "load_schema"):  # 7
        schema = load_model(SchemaConfig, config_dir / SCHEMA_FILE)
        computed = metadata.schema_hash(metadata.input_schema(schema))
    with _step(logger, "contract_check"):  # 8 (FR-005)
        if computed != meta.schema_hash:
            raise ValueError(f"schema hash {computed} of {config_dir / SCHEMA_FILE} differs from "
                             f"the artifact's {meta.schema_hash}: the API contract is not the "
                             "training contract")  # fmt: skip
    log_event(logger, logging.INFO, "schema.verified", schema_hash=computed)
    with _step(logger, "build_request_models"):  # 9
        example = _read_example(config_dir / EXAMPLE_FILE) if warm_up else None
        models = build_request_models(schema, example)
    with _step(logger, "load_model"):  # 10
        started = time.perf_counter()
        model, _ = artifact.load_verified(model_dir, require_release=require_release)
    log_event(logger, logging.INFO, "model.loaded", load_ms=_ms(started))
    context = ModelContext(model, meta, schema, model_input_columns(schema), models, example)
    if example is not None:
        with _step(logger, "warm_up"):  # 11: the example validates and passes the guard
            started = time.perf_counter()
            record = models.property_input.model_validate(example).model_dump(by_alias=True)
            predict_records(context, [record])
        log_event(logger, logging.INFO, "warmup.completed", latency_ms=_ms(started))
    return context


def _read_example(path: Path) -> dict[str, Any]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise TypeError(f"{path}: expected one JSON object with the model-input fields")
    return content


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


# ---------------------------------------------------------------- prediction path


def build_frame(records: Sequence[dict[str, Any]], schema: SchemaConfig) -> pd.DataFrame:
    """Validated records (aliases as keys) -> an n-row frame in schema column order, with
    ``None`` as ``NaN`` and the dtypes the training loader produces (DN-18): ``int64``,
    ``float64``, and pandas ``str`` for categoricals."""
    data: dict[str, pd.Series] = {}
    for spec in schema.with_role("model_input"):
        values = [record[spec.name] for record in records]
        if spec.dtype == "int":
            data[spec.name] = pd.Series(values, dtype="int64")
        elif spec.dtype == "float":
            data[spec.name] = pd.Series(values, dtype="float64")
        else:
            data[spec.name] = pd.Series(values, dtype=str)
    return pd.DataFrame(data)


def predict_frame(context: ModelContext, frame: pd.DataFrame) -> Predictions:
    """One ``model.predict`` call, then the SD-16 guard and the domain flag (FR-045)."""
    prices = np.asarray(context.model.predict(frame.loc[:, context.columns]), dtype="float64")
    bad = ~(np.isfinite(prices) & (prices > 0))
    if bad.any():
        raise GuardViolation(np.flatnonzero(bad).tolist())
    rule = context.metadata.scope_rule
    flags = (frame[rule.column] > rule.max_in_domain).to_numpy()
    return Predictions([float(p) for p in prices], [bool(f) for f in flags])


def predict_records(context: ModelContext, records: Sequence[dict[str, Any]]) -> Predictions:
    return predict_frame(context, build_frame(records, context.schema))


def inputs_sha256(payload: Any) -> str:
    """SHA-256 of the canonical JSON of the validated input (sorted keys, aliases, ``null``
    preserved) (DOC-04 §13.3): identical requests can be matched without logging them."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def single_response(context: ModelContext, predictions: Predictions) -> PredictionResponse:
    return PredictionResponse(
        predicted_price=predictions.prices[0],
        model_version=context.metadata.model_version,
        out_of_domain=predictions.out_of_domain[0],
    )


def batch_response(context: ModelContext, predictions: Predictions) -> BatchPredictionResponse:
    items = [
        BatchPrediction(index=i, predicted_price=price, model_version=context.metadata.model_version,
                        out_of_domain=flag)
        for i, (price, flag) in enumerate(zip(predictions.prices, predictions.out_of_domain,
                                              strict=True))
    ]  # fmt: skip
    return BatchPredictionResponse(count=len(items), predictions=items)

"""The FastAPI service (DOC-04 §5, §6; FR-040 to FR-047).

Run with one worker (SD-09)::

    uvicorn house_price.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log

The **lifespan** performs the 13 startup steps of DOC-04 §5.2 before any request is
accepted; any failure raises, so Uvicorn exits non-zero and the service never becomes ready
(SD-13). The request models are generated from ``schema.yaml`` at step 9, so the two
prediction routes are registered then, with those models as their bodies; ``/health`` and
``/model-info`` only read the published context.

A request-ID and timing middleware gives every response an ``X-Request-ID`` (SD-05), writes
the single ``prediction.completed`` line of each successful prediction request (FR-047) and
contains unexpected exceptions as a generic 500 (DOC-04 §14.2). ``/health`` and
``/model-info`` are never logged.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, Request, Response
from pydantic import BaseModel

from house_price.api import errors
from house_price.api.logging import API_LOGGER, configure_logging, log_event, set_model_version
from house_price.api.predict import (
    ModelContext,
    StartupError,
    batch_response,
    inputs_sha256,
    load_model_context,
    predict_records,
    single_response,
)
from house_price.api.schemas import (
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    BatchPredictionResponse,
    HealthResponse,
    InternalErrorResponse,
    PredictionResponse,
)
from house_price.api.settings import Settings, SettingsError
from house_price.persistence.metadata import ArtifactMetadata

TITLE = "House Price Prediction API"
DESCRIPTION = (
    "Prices Ames, Iowa residential properties with the frozen model artifact.\n\n"
    "Requests carry every model-input column of `schema.yaml` under its canonical name; "
    "nullable columns must be sent as explicit `null`. `/predict/batch` accepts "
    f"{MIN_BATCH_SIZE} to {MAX_BATCH_SIZE} properties per request (MAX_BATCH_SIZE = "
    f"{MAX_BATCH_SIZE}). Every response carries an `X-Request-ID` header."
)
PREDICTION_PATHS = ("/predict", "/predict/batch")
_ERRORS: dict[int | str, dict[str, Any]] = {
    500: {"model": InternalErrorResponse, "description": "Internal error (generic body)"}
}


def _context(request: Request) -> ModelContext:
    context: ModelContext = request.app.state.model_context
    return context


def create_app(settings: Settings | None = None) -> FastAPI:
    """A new application; ``settings`` defaults to the environment (startup step 1)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        started = time.perf_counter()
        set_model_version(None)
        logger = configure_logging("INFO")
        try:  # 1
            current = settings or Settings.from_env()
        except SettingsError as exc:
            log_event(logger, logging.CRITICAL, "startup.failed", step="read_settings",
                      reason=str(exc), error_type=type(exc).__name__)  # fmt: skip
            raise StartupError("read_settings", str(exc)) from exc
        logger = configure_logging(current.log_level)  # 2
        log_event(logger, logging.INFO, "service.starting", settings=current.public())
        context = load_model_context(  # 3 to 11
            current.model_dir,
            current.config_dir,
            allow_non_release=current.allow_non_release,
            logger=logger,
        )
        _add_prediction_routes(app, context)  # the step 9 models become the request bodies
        app.state.model_context = context  # 12
        app.state.ready = True
        log_event(logger, logging.INFO, "service.ready",  # 13
                  model_version=context.metadata.model_version,
                  startup_ms=round((time.perf_counter() - started) * 1000, 3),
                  schema_hash=context.metadata.schema_hash)  # fmt: skip
        yield
        app.state.ready = False
        log_event(logger, logging.INFO, "service.stopping")

    app = FastAPI(title=TITLE, description=DESCRIPTION, version="1.0.0", lifespan=lifespan)
    app.state.ready = False
    errors.install(app)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - contained as a generic 500 (DOC-04 §14.2)
            response = errors.unexpected_error(request, exc)
        response.headers["X-Request-ID"] = request_id
        entry = getattr(request.state, "prediction_log", None)
        if entry is not None and response.status_code == 200:
            log_event(logging.getLogger(API_LOGGER), logging.INFO, "prediction.completed",
                      request_id=request_id, **entry,
                      latency_ms=round((time.perf_counter() - started) * 1000, 3),
                      status_code=response.status_code)  # fmt: skip
        return response

    @app.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
    def health(request: Request) -> HealthResponse:
        """200 only once every startup verification has passed (DOC-04 §5.4). Not logged."""
        return HealthResponse(model_version=_context(request).metadata.model_version)

    @app.get("/model-info", response_model=ArtifactMetadata, summary="Loaded model metadata")
    def model_info(request: Request) -> ArtifactMetadata:
        """The loaded artifact's complete `metadata.json` (DOC-03 §15.3). Not logged."""
        return _context(request).metadata

    return app


def _add_prediction_routes(app: FastAPI, context: ModelContext) -> None:
    property_input = context.models.property_input
    batch_input = context.models.batch_input
    version = context.metadata.model_version

    def predict(request: Request, body: BaseModel) -> PredictionResponse:
        record = body.model_dump(by_alias=True)
        digest = request.state.inputs_sha256 = inputs_sha256(record)
        predictions = predict_records(_context(request), [record])
        request.state.prediction_log = {
            "endpoint": "/predict", "n_items": 1, "inputs_sha256": digest,
            "prediction": predictions.prices[0],
            "out_of_domain_count": sum(predictions.out_of_domain), "model_version": version,
        }  # fmt: skip
        return single_response(_context(request), predictions)

    def predict_batch(request: Request, body: BaseModel) -> BatchPredictionResponse:
        records = [item.model_dump(by_alias=True) for item in body.properties]  # type: ignore[attr-defined]
        digest = request.state.inputs_sha256 = inputs_sha256(records)
        predictions = predict_records(_context(request), records)  # one n-row predict call
        request.state.prediction_log = {
            "endpoint": "/predict/batch", "n_items": len(records), "inputs_sha256": digest,
            "prediction": predictions.prices,
            "out_of_domain_count": sum(predictions.out_of_domain), "model_version": version,
        }  # fmt: skip
        return batch_response(_context(request), predictions)

    _body(predict, property_input, PredictionResponse)
    _body(predict_batch, batch_input, BatchPredictionResponse)
    app.router.routes[:] = [
        r for r in app.router.routes if getattr(r, "path", None) not in PREDICTION_PATHS
    ]
    app.add_api_route("/predict", predict, methods=["POST"], response_model=PredictionResponse,
                      summary="Price one property", responses=_ERRORS,
                      description="One property with every model-input field. Out-of-domain "
                      "inputs (GrLivArea above the scope rule) are priced and flagged.")  # fmt: skip
    app.add_api_route("/predict/batch", predict_batch, methods=["POST"],
                      response_model=BatchPredictionResponse, responses=_ERRORS,
                      summary=f"Price {MIN_BATCH_SIZE} to {MAX_BATCH_SIZE} properties",
                      description=f"{MIN_BATCH_SIZE} to {MAX_BATCH_SIZE} properties "
                      f"(MAX_BATCH_SIZE = {MAX_BATCH_SIZE}); validation is atomic and "
                      "predictions are returned in input order.")  # fmt: skip
    _document_example(app, context)


def _document_example(app: FastAPI, context: ModelContext) -> None:
    """Attach ``api_example.json`` to the OpenAPI schema (SD-15, AC-058) with its ``null``
    values intact: FastAPI serializes the schema with ``exclude_none``, which would drop the
    explicit ``null``s that SD-01 requires and make the documented example invalid."""
    app.openapi_schema = None  # regenerate with the new routes
    generate = FastAPI.openapi.__get__(app)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            spec = generate()
            if context.example is not None:
                schemas = spec["components"]["schemas"]
                schemas["PropertyInput"]["examples"] = [context.example]
                schemas["BatchInput"]["examples"] = [{"properties": [context.example]}]
            app.openapi_schema = spec
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _body(endpoint: Callable[..., Any], model: type[BaseModel], returns: type[BaseModel]) -> None:
    """Declare the generated model as the endpoint's JSON body (FastAPI reads signatures)."""
    endpoint.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter("request", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation=Request),
            inspect.Parameter("body", inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              annotation=Annotated[model, Body()]),
        ],
        return_annotation=returns,
    )  # fmt: skip


app = create_app()

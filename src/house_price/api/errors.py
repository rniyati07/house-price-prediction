"""Error handling (DOC-04 §6.6, §7.5, §8.3, §13.4; SD-05).

- 422: FastAPI's standard body; ``request.validation_failed`` logs the request ID, endpoint
  and the ``(loc, type)`` pairs only, never input values.
- SD-16 guard violation: ``prediction.guard_violation`` at ERROR, then the generic 500.
- Any other exception: ``request.failed`` with the exception type and traceback (logs only),
  then the generic 500 ``{"detail": "Internal server error", "request_id": ...}``. The
  request-ID middleware in ``app.py`` calls :func:`unexpected_error`, so the 500 also carries
  ``X-Request-ID`` and the service keeps running.
"""

from __future__ import annotations

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from house_price.api.logging import API_LOGGER, log_event
from house_price.api.predict import GuardViolation
from house_price.api.schemas import InternalErrorResponse

logger = logging.getLogger(API_LOGGER)


def request_id_of(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def internal_error(request_id: str) -> JSONResponse:
    body = InternalErrorResponse(request_id=request_id).model_dump()
    return JSONResponse(status_code=500, content=body)


async def validation_failed(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, RequestValidationError)
    errors = [{"loc": list(e.get("loc", ())), "type": e.get("type")} for e in exc.errors()]
    log_event(logger, logging.WARNING, "request.validation_failed",
              request_id=request_id_of(request), endpoint=request.url.path,
              errors=errors)  # fmt: skip
    return await request_validation_exception_handler(request, exc)


async def guard_violated(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, GuardViolation)
    log_event(logger, logging.ERROR, "prediction.guard_violation",
              request_id=request_id_of(request), endpoint=request.url.path,
              inputs_sha256=getattr(request.state, "inputs_sha256", None),
              rows=exc.rows)  # fmt: skip
    return internal_error(request_id_of(request))


def unexpected_error(request: Request, exc: BaseException) -> Response:
    log_event(logger, logging.ERROR, "request.failed",
              request_id=request_id_of(request), endpoint=request.url.path,
              error_type=type(exc).__name__,
              traceback="".join(traceback.format_exception(exc)))  # fmt: skip
    return internal_error(request_id_of(request))


def install(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_failed)
    app.add_exception_handler(GuardViolation, guard_violated)

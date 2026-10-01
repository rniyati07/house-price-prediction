"""Structured JSON logging (DOC-04 §13, SD-10): standard-library ``logging``, one JSON object
per line, named events.

Every line carries ``timestamp`` (UTC ISO-8601), ``level``, ``event``, ``service`` and, once
the artifact's metadata is loaded, ``model_version``. Event-specific fields are passed as
keyword arguments to :func:`log_event`. Property values are never passed in: prediction
logs carry an inputs hash (DOC-04 §13.3), validation logs only ``(loc, type)`` pairs.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Literal

SERVICE = "house-price-api"
API_LOGGER = "house_price.api"
BATCH_LOGGER = "house_price.batch"

_state: dict[str, str | None] = {"model_version": None}


def set_model_version(version: str | None) -> None:
    """Add ``model_version`` to every subsequent line (set once the metadata is loaded)."""
    _state["model_version"] = version


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "service": SERVICE,
        }
        if _state["model_version"] is not None:
            line["model_version"] = _state["model_version"]
        line.update(getattr(record, "fields", {}))
        return json.dumps(line, default=str, allow_nan=False)


class _StreamHandler(logging.Handler):
    """Writes to the *current* ``sys.stdout`` / ``sys.stderr`` (looked up per record, so a
    redirected stream, e.g. under pytest, receives the lines)."""

    def __init__(self, stream: Literal["stdout", "stderr"]) -> None:
        super().__init__()
        self._stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            stream = getattr(sys, self._stream)
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:  # noqa: BLE001 - logging must never break a request
            self.handleError(record)


def configure_logging(
    level: str = "INFO",
    name: str = API_LOGGER,
    stream: Literal["stdout", "stderr"] = "stdout",
) -> logging.Logger:
    """Install the JSON handler on ``name`` (idempotent; replaces earlier handlers)."""
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = _StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})

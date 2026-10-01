"""Environment settings of the service and the batch CLI (DOC-04 §11.6, SD-08; startup step 1).

The variables are parsed into a strict Pydantic model, as every other configuration file of
the project is (``house_price.config``). Defaults are the container paths of DOC-04 §11.6;
``make serve`` and ``make predict`` point them at a local artifact. The model version is
deliberately **not** a setting: it comes only from ``metadata.json``.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# environment variable -> settings field (DOC-04 §11.6)
ENVIRONMENT = {
    "PORT": "port",
    "HPP_MODEL_DIR": "model_dir",
    "HPP_CONFIG_DIR": "config_dir",
    "HPP_LOG_LEVEL": "log_level",
    "HPP_ALLOW_NON_RELEASE": "allow_non_release",
}


class SettingsError(ValueError):
    """An environment variable is invalid (startup step 1: exit, invalid configuration)."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int = Field(default=8000, ge=1, le=65535)
    model_dir: Path = Path("/app/model")
    config_dir: Path = Path("/app/configs")
    log_level: LogLevel = "INFO"
    # Allows a non-release (smoke) artifact to load; set only for CI and smoke tests (SD-11).
    allow_non_release: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if environ is None else environ
        values: dict[str, object] = {}
        for variable, name in ENVIRONMENT.items():
            raw = source.get(variable)
            if raw is None or raw == "":
                continue
            if name == "log_level":
                raw = raw.upper()
            elif name == "allow_non_release":
                if raw.lower() not in ("true", "false"):
                    raise SettingsError(f"{variable} must be 'true' or 'false', not {raw!r}")
                values[name] = raw.lower() == "true"
                continue
            values[name] = raw
        try:
            return cls.model_validate(values)
        except ValidationError as exc:
            problems = "; ".join(f"{_variable(str(e['loc'][0]))}: {e['msg']}" for e in exc.errors())
            raise SettingsError(f"invalid settings: {problems}") from exc

    def public(self) -> dict[str, object]:
        """The settings as logged by ``service.starting`` (paths only, never their contents)."""
        return {_variable(name): str(value) for name, value in self.model_dump().items()}


def _variable(field: str) -> str:
    return next((v for v, f in ENVIRONMENT.items() if f == field), field)

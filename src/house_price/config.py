"""Validated configuration models, loaders, and the configuration hash (DOC-03 §5.1).

Every YAML file in ``configs/`` is loaded into a Pydantic model with ``extra="forbid"``,
so unknown keys, missing keys, and wrong types fail immediately with a clear error
(NFR-019). No threshold, size, or path used by the data layer is hard-coded (NFR-020).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

DEFAULT_CONFIG_DIR = Path("configs")

ColumnRole = Literal["model_input", "excluded", "identifier", "target"]
ColumnDtype = Literal["int", "float", "category", "string"]


class ConfigError(ValueError):
    """Raised when a configuration file is missing, unreadable, or invalid."""


class StrictModel(BaseModel):
    """Base model: unknown keys are rejected and instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- data.yaml


class ProcessedPaths(StrictModel):
    dev_path: Path
    holdout_path: Path
    manifest_path: Path


class ScopeConfig(StrictModel):
    column: str
    threshold: float
    expected_rows_after: int = Field(gt=0)


class DataConfig(StrictModel):
    dataset_name: str
    raw_path: Path
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    missing_tokens: list[str] = Field(min_length=1)
    processed: ProcessedPaths
    scope: ScopeConfig
    reports_dir: Path


# --------------------------------------------------------------------- validation.yaml


class ValidationConfig(StrictModel):
    seed: int
    holdout_fraction: float = Field(gt=0.0, lt=1.0)
    n_bins: int = Field(ge=2)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(ge=1)
    reproducibility_tolerance: float = Field(gt=0.0)
    split_balance_tolerance_pp: float = Field(gt=0.0)


# ------------------------------------------------------------------------- schema.yaml


class ColumnSpec(StrictModel):
    """Contract for one column of the raw dataset (DOC-03 §5.3)."""

    name: str
    source_name: str
    dtype: ColumnDtype
    nullable: bool
    role: ColumnRole
    allowed_values: list[int] | list[str] | None = None
    min: float | None = None
    max: float | None = None
    strict_min: bool = False
    unique: bool = False
    pattern: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ColumnSpec:
        numeric = self.dtype in ("int", "float")
        if self.dtype == "category" and not self.allowed_values:
            raise ValueError(f"column {self.name!r}: category columns need allowed_values")
        if self.dtype == "string" and self.allowed_values:
            raise ValueError(f"column {self.name!r}: string columns cannot have allowed_values")
        if not numeric and (self.min is not None or self.max is not None or self.strict_min):
            raise ValueError(f"column {self.name!r}: ranges apply to numeric columns only")
        if self.strict_min and self.min is None:
            raise ValueError(f"column {self.name!r}: strict_min requires min")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"column {self.name!r}: min is greater than max")
        if numeric and self.pattern is not None:
            raise ValueError(f"column {self.name!r}: pattern applies to text columns only")
        if self.allowed_values:
            expected_type = int if self.dtype == "int" else str
            if not all(type(value) is expected_type for value in self.allowed_values):
                raise ValueError(
                    f"column {self.name!r}: allowed_values must all be {expected_type.__name__}"
                )
            if len(set(self.allowed_values)) != len(self.allowed_values):
                raise ValueError(f"column {self.name!r}: allowed_values contains duplicates")
        return self


class SchemaConfig(StrictModel):
    """The single source of the data contract (FR-005)."""

    columns: list[ColumnSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_columns(self) -> SchemaConfig:
        for attribute in ("name", "source_name"):
            values = [getattr(column, attribute) for column in self.columns]
            duplicates = sorted({value for value in values if values.count(value) > 1})
            if duplicates:
                raise ValueError(f"duplicate column {attribute}s in schema: {duplicates}")
        roles = [column.role for column in self.columns]
        if roles.count("target") != 1:
            raise ValueError("schema must declare exactly one target column")
        if roles.count("identifier") < 1:
            raise ValueError("schema must declare at least one identifier column")
        return self

    @property
    def names(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def target(self) -> str:
        return next(column.name for column in self.columns if column.role == "target")

    @property
    def id_column(self) -> str:
        """The primary identifier: the first identifier column in schema order."""
        return next(column.name for column in self.columns if column.role == "identifier")

    @property
    def source_to_name(self) -> dict[str, str]:
        return {column.source_name: column.name for column in self.columns}

    def by_name(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"column {name!r} is not declared in schema.yaml")

    def with_role(self, *roles: ColumnRole) -> list[ColumnSpec]:
        return [column for column in self.columns if column.role in roles]


# ------------------------------------------------------------------------------ loading

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, failing clearly if the file is missing or not a mapping."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(content, dict):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    return content


def load_model(model: type[ModelT], path: Path) -> ModelT:
    """Load ``path`` into ``model``; validation errors name the file and every bad field."""
    try:
        return model.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}: invalid configuration\n{exc}") from exc


def config_hash(*models: BaseModel) -> str:
    """SHA-256 of the normalized, concatenated configuration (DOC-03 §5.1)."""
    payload = [model.model_dump(mode="json") for model in models]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProjectConfig:
    """All configuration the data layer needs, with paths resolved against ``root``."""

    root: Path
    data: DataConfig
    validation: ValidationConfig
    schema: SchemaConfig

    def resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @property
    def raw_path(self) -> Path:
        return self.resolve(self.data.raw_path)

    @property
    def dev_path(self) -> Path:
        return self.resolve(self.data.processed.dev_path)

    @property
    def holdout_path(self) -> Path:
        return self.resolve(self.data.processed.holdout_path)

    @property
    def manifest_path(self) -> Path:
        return self.resolve(self.data.processed.manifest_path)

    @property
    def reports_dir(self) -> Path:
        return self.resolve(self.data.reports_dir)

    @property
    def config_hash(self) -> str:
        return config_hash(self.data, self.validation, self.schema)

    @property
    def schema_hash(self) -> str:
        return config_hash(self.schema)


def load_project_config(
    config_dir: Path = DEFAULT_CONFIG_DIR, root: Path | None = None
) -> ProjectConfig:
    """Load ``data.yaml``, ``validation.yaml`` and ``schema.yaml`` from ``config_dir``.

    Relative paths inside the configuration are resolved against ``root``
    (default: the current working directory, i.e. the repository root).
    """
    return ProjectConfig(
        root=(root or Path.cwd()).resolve(),
        data=load_model(DataConfig, config_dir / "data.yaml"),
        validation=load_model(ValidationConfig, config_dir / "validation.yaml"),
        schema=load_model(SchemaConfig, config_dir / "schema.yaml"),
    )

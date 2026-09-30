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


class SmokeConfig(StrictModel):
    """Smoke training (DN-17, DOC-03 §16.3): a reduced run that exercises every stage and
    can never be released. The development sample and the holdout substitute are disjoint
    samples of development rows; the real holdout is never used."""

    sample_rows: int = Field(ge=10)
    holdout_rows: int = Field(ge=1)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(ge=1)
    grid_points: int = Field(ge=2)
    random_forest_trials: int = Field(ge=1)
    lightgbm_trials: int = Field(ge=1)


class ValidationConfig(StrictModel):
    seed: int
    holdout_fraction: float = Field(gt=0.0, lt=1.0)
    n_bins: int = Field(ge=2)
    cv_folds: int = Field(ge=2)
    cv_repeats: int = Field(ge=1)
    reproducibility_tolerance: float = Field(gt=0.0)
    split_balance_tolerance_pp: float = Field(gt=0.0)
    smoke: SmokeConfig | None = None


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


# ----------------------------------------------------------------------- features.yaml


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


class SemanticFillConfig(StrictModel):
    """Layer 1 "feature absent" columns (ADR-05, DOC-03 §6.2)."""

    categorical_none: list[str] = Field(min_length=1)
    numeric_zero: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_disjoint(self) -> SemanticFillConfig:
        both = self.categorical_none + self.numeric_zero
        if _duplicates(both):
            raise ValueError(f"columns listed more than once: {_duplicates(both)}")
        return self


class OrdinalConfig(StrictModel):
    """The Po-Ex ordinal map and the columns it applies to (DN-13)."""

    columns: list[str] = Field(min_length=1)
    mapping: dict[str, int] = Field(min_length=2)

    @model_validator(mode="after")
    def _check_map(self) -> OrdinalConfig:
        if _duplicates(self.columns):
            raise ValueError(f"ordinal columns listed more than once: {_duplicates(self.columns)}")
        codes = list(self.mapping.values())
        if sorted(codes) != list(range(len(codes))):
            raise ValueError("ordinal codes must be 0, 1, 2, ... with no gaps or repeats")
        return self


class BranchGroups(StrictModel):
    """Column groups of one preprocessing branch (DOC-03 §5.6, §7.5)."""

    numeric: list[str]
    ordinal: list[str]
    nominal: list[str]
    dropped: list[str]
    # M7 ablation outcome (DOC-03 §6.6 step 5): engineered features this branch drops.
    # Absent (None) = no committed outcome yet; [] = outcome committed, nothing dropped.
    # Written only by a reviewed commit, never by code (RC-02).
    dropped_engineered: list[str] | None = None

    @property
    def all_columns(self) -> list[str]:
        return self.numeric + self.ordinal + self.nominal + self.dropped

    @model_validator(mode="after")
    def _check_disjoint(self) -> BranchGroups:
        if _duplicates(self.all_columns):
            raise ValueError(f"columns in more than one group: {_duplicates(self.all_columns)}")
        if self.dropped_engineered is not None:
            if _duplicates(self.dropped_engineered):
                raise ValueError(
                    f"dropped_engineered lists a feature twice: {_duplicates(self.dropped_engineered)}"
                )
            active = set(self.numeric + self.ordinal + self.nominal)
            inactive = [f for f in self.dropped_engineered if f not in active]
            if inactive:
                raise ValueError(
                    f"dropped_engineered features must be in an active group: {inactive}"
                )
        return self


class FeatureConfig(StrictModel):
    """``features.yaml``: filler lists, ordinal map, engineered features, branch groups."""

    semantic_fill: SemanticFillConfig
    ordinal: OrdinalConfig
    categorical_codes: list[str]
    engineered: list[str] = Field(min_length=1)
    linear: BranchGroups
    tree: BranchGroups

    @model_validator(mode="after")
    def _check_features(self) -> FeatureConfig:
        if _duplicates(self.engineered):
            raise ValueError(f"engineered features listed twice: {_duplicates(self.engineered)}")
        for branch in ("linear", "tree"):
            dropped = getattr(self, branch).dropped_engineered or []
            unknown = [f for f in dropped if f not in self.engineered]
            if unknown:
                raise ValueError(f"{branch}.dropped_engineered: not engineered features {unknown}")
        return self

    @property
    def has_ablation_outcome(self) -> bool:
        """Both branches carry a committed outcome (DOC-05 RC-02)."""
        return self.linear.dropped_engineered is not None and (
            self.tree.dropped_engineered is not None
        )


# ------------------------------------------------------------------------- models.yaml


class DummyBaselineConfig(StrictModel):
    """``dummy_median`` (DOC-03 §8.2): the full pipeline with ``DummyRegressor``."""

    branch: Literal["linear", "tree"]
    strategy: Literal["median"]


class TwoFeatureBaselineConfig(StrictModel):
    """``linear_2feat`` (DOC-03 §8.3): ``LinearRegression`` on the two dominant drivers."""

    features: list[str] = Field(min_length=1)


class BaselinesConfig(StrictModel):
    dummy_median: DummyBaselineConfig
    linear_2feat: TwoFeatureBaselineConfig


ParamValue = int | float | bool | str


class GridConfig(StrictModel):
    """A log-spaced grid over one hyperparameter (DOC-03 §10.3, §10.4)."""

    param: str
    low: float = Field(gt=0)
    high: float = Field(gt=0)
    n: int = Field(ge=2)

    @model_validator(mode="after")
    def _check_range(self) -> GridConfig:
        if self.low >= self.high:
            raise ValueError(f"grid low {self.low} must be below high {self.high}")
        return self


class SearchParam(StrictModel):
    """One Optuna search dimension (DOC-03 §10.5, §10.6, DN-06)."""

    type: Literal["int", "float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _check_range(self) -> SearchParam:
        if self.low >= self.high:
            raise ValueError(f"search range low {self.low} must be below high {self.high}")
        if self.type == "int" and not (
            float(self.low).is_integer() and float(self.high).is_integer()
        ):
            raise ValueError("integer search bounds must be whole numbers")
        if self.log and self.low <= 0:
            raise ValueError("a log-scaled search range must be positive")
        return self


class CandidateConfig(StrictModel):
    """One model candidate (DOC-03 §8.1, §8.4 to §8.7, §10).

    ``fixed`` settings never change; ``reference`` is the configuration used for the M7
    ablation and development check. The global seed (``validation.yaml``, DN-02) is passed
    to the estimator by the registry and is not repeated here. The tuning budget is
    configuration: ``grid.n`` points for a grid, ``n_trials`` over ``search_space`` for an
    Optuna study (M8).
    """

    branch: Literal["linear", "tree"]
    tier: int = Field(ge=1)
    tuning: Literal["grid", "optuna"]
    grid: GridConfig | None = None
    n_trials: int | None = Field(default=None, ge=1)
    search_space: dict[str, SearchParam] | None = None
    fixed: dict[str, ParamValue] = Field(default_factory=dict)
    reference: dict[str, ParamValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_tuning(self) -> CandidateConfig:
        if self.tuning == "grid" and (self.grid is None or self.search_space or self.n_trials):
            raise ValueError("a grid candidate needs 'grid' and no 'search_space'/'n_trials'")
        if self.tuning == "optuna" and (not self.search_space or self.n_trials is None):
            raise ValueError("an Optuna candidate needs 'search_space' and 'n_trials'")
        overlap = sorted(set(self.search_space or {}) & set(self.fixed))
        if overlap:
            raise ValueError(f"fixed settings cannot also be searched: {overlap}")
        return self


class CandidatesConfig(StrictModel):
    ridge: CandidateConfig
    lasso: CandidateConfig
    random_forest: CandidateConfig
    lightgbm: CandidateConfig


class AblationConfig(StrictModel):
    """Reference candidate per branch for the feature ablation (DOC-03 §6.6, DN-05)."""

    linear: Literal["ridge"]
    tree: Literal["lightgbm"]


class BlendConfig(StrictModel):
    """The DN-07 blend (DOC-03 §8.8): the better tuned linear candidate plus LightGBM.

    Weights are fixed at 0.5/0.5 by DN-07 and live in code, not configuration.
    """

    tier: int = Field(ge=1)
    linear: list[Literal["ridge", "lasso"]] = Field(min_length=1)
    tree: Literal["lightgbm"]


class ModelsConfig(StrictModel):
    """``models.yaml``: candidate definitions (DOC-03 §5.1, §8). M6 defines the baselines;
    M7 adds the four candidates and the ablation references; M8 the search spaces, trial
    budgets, and the blend."""

    baselines: BaselinesConfig
    candidates: CandidatesConfig
    ablation: AblationConfig
    blend: BlendConfig


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


def load_feature_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> FeatureConfig:
    """Load ``features.yaml`` (DOC-03 §5.1) from ``config_dir``."""
    return load_model(FeatureConfig, config_dir / "features.yaml")


def load_models_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> ModelsConfig:
    """Load ``models.yaml`` (DOC-03 §5.1) from ``config_dir``."""
    return load_model(ModelsConfig, config_dir / "models.yaml")


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

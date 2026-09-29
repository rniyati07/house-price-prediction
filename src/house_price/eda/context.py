"""The data an EDA run may use, loaded once through the M2 data layer.

``raw`` is the validated raw file (all rows). Only target-free analyses and the ADR-06 scope
review may use it; target-free code reads ``raw_features``, which has no ``SalePrice``.
``dev`` is the persisted development set. The holdout rows are never loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    FeatureConfig,
    ProjectConfig,
    SchemaConfig,
    load_feature_config,
    load_project_config,
)
from house_price.data.errors import DataError
from house_price.data.load import load_raw, sha256_file
from house_price.data.scope import ScopeRecord, apply_scope_rule
from house_price.data.split import SplitManifest, create_or_load_split

TABLES_DIR = Path("reports/eda")
FIGURES_DIR = Path("reports/figures/eda")

# Po-Ex quality/condition scales (DOC-02 §8.6, DN-13) and their order, absent first.
ORDINAL_SCALES = (
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
    "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
)  # fmt: skip
ORDINAL_LEVELS = ("None", "Po", "Fa", "TA", "Gd", "Ex")


class EDAError(DataError):
    """The EDA cannot run on the current project state."""


def find_project_root(start: Path) -> Path:
    """Walk up from ``start`` to the directory that contains ``configs/data.yaml``."""
    for candidate in (start, *start.parents):
        if (candidate / DEFAULT_CONFIG_DIR / "data.yaml").is_file():
            return candidate
    raise EDAError(f"no project root (configs/data.yaml) found above {start}")


def numeric_features(schema: SchemaConfig) -> list[str]:
    """Numeric model inputs that are quantities (codes such as ``MSSubClass`` excluded)."""
    return [
        spec.name
        for spec in schema.with_role("model_input")
        if spec.dtype in ("int", "float") and not spec.allowed_values
    ]


def categorical_features(schema: SchemaConfig, *, include_excluded: bool = False) -> list[str]:
    """Categorical columns: text categories plus numeric codes (``MSSubClass``)."""
    roles = ("model_input", "excluded") if include_excluded else ("model_input",)
    return [
        spec.name
        for spec in schema.with_role(*roles)
        if spec.dtype == "category" or (spec.dtype == "int" and spec.allowed_values)
    ]


def log_price(frame: pd.DataFrame, target: str) -> pd.Series:
    """``log1p`` of the target, the modeling scale fixed by ADR-02."""
    return pd.Series(np.log1p(frame[target].astype("float64")), index=frame.index, name="log_price")


@dataclass(frozen=True)
class EDAContext:
    """Everything an EDA module needs, plus where it writes."""

    config: ProjectConfig
    raw: pd.DataFrame
    dev: pd.DataFrame
    scope_record: ScopeRecord
    manifest: SplitManifest
    raw_sha256: str
    features: FeatureConfig
    tables_dir: Path
    figures_dir: Path

    @property
    def schema(self) -> SchemaConfig:
        return self.config.schema

    @property
    def target(self) -> str:
        return self.config.schema.target

    @property
    def id_column(self) -> str:
        return self.config.schema.id_column

    @property
    def raw_features(self) -> pd.DataFrame:
        """The raw file without the target, for target-free profiling (FR-009)."""
        return self.raw.drop(columns=[self.target])

    @classmethod
    def load(
        cls,
        config_dir: Path | None = None,
        root: Path | None = None,
        tables_dir: Path | None = None,
        figures_dir: Path | None = None,
    ) -> EDAContext:
        """Load and validate the data through the M2 layer.

        The persisted split must already exist (``make split``); the EDA never creates it.
        Paths default to the project layout under ``root`` (found from the working
        directory, so notebooks can run from ``notebooks/``).
        """
        root = (root or find_project_root(Path.cwd())).resolve()
        config_dir = config_dir or root / DEFAULT_CONFIG_DIR
        config = load_project_config(config_dir, root)
        if not config.manifest_path.is_file():
            raise EDAError(
                f"split manifest not found at {config.manifest_path}; "
                "create the split first (make split)"
            )
        raw = load_raw(config)
        in_scope, record = apply_scope_rule(raw, config.data.scope, config.schema.id_column)
        split = create_or_load_split(in_scope, record, config)
        return cls(
            config=config,
            raw=raw,
            dev=split.dev,
            scope_record=record,
            manifest=split.manifest,
            raw_sha256=sha256_file(config.raw_path),
            features=load_feature_config(config_dir),
            tables_dir=tables_dir or root / TABLES_DIR,
            figures_dir=figures_dir or root / FIGURES_DIR,
        )

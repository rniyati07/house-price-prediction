"""Initial profiling of the validated raw dataset: missing values, column profiles,
duplicate detection, data-quality consistency flags, and dataset metadata.

Profiling only *reports*. It never edits, drops, or repairs values (DOC-03 §5.4): the only
row removal in the project is the scope rule, and duplicate handling is limited to the
schema's unique-``Id`` check. Flags raised here feed the data card's known issues, which
M3 verifies (DOC-02 §6.6, E-09 to E-11).

The raw-file profile does not describe the target's distribution: target analysis runs on
the development set only (FR-009, DOC-02 E-05).
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from house_price.config import ProjectConfig, SchemaConfig

TARGET_WITHHELD = "withheld: target distribution is analysed on the development set only"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------- missing values


def missing_value_profile(frame: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """Missing count and share for every column that has missing values, largest first.

    Counts come from the explicit token set (DN-18), so they are stable across runs and
    environments (AC-003).
    """
    rows = []
    for spec in schema.columns:
        n_missing = int(frame[spec.name].isna().sum())
        if n_missing:
            rows.append(
                {
                    "column": spec.name,
                    "source_name": spec.source_name,
                    "dtype": spec.dtype,
                    "role": spec.role,
                    "nullable": spec.nullable,
                    "n_missing": n_missing,
                    "pct_missing": round(100.0 * n_missing / len(frame), 3),
                }
            )
    columns = ["column", "source_name", "dtype", "role", "nullable", "n_missing", "pct_missing"]
    profile = pd.DataFrame(rows, columns=columns)
    return profile.sort_values(["n_missing", "column"], ascending=[False, True], ignore_index=True)


# ---------------------------------------------------------------------- column profile


def column_profile(frame: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """One row per column: type, missingness, cardinality, and summary statistics."""
    rows = []
    for spec in schema.columns:
        series = frame[spec.name]
        row: dict[str, object] = {
            "column": spec.name,
            "source_name": spec.source_name,
            "role": spec.role,
            "dtype": spec.dtype,
            "nullable": spec.nullable,
            "n_missing": int(series.isna().sum()),
            "pct_missing": round(100.0 * series.isna().mean(), 3),
            "n_unique": int(series.nunique(dropna=True)),
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
            "std": None,
            "n_zero": None,
            "top_value": None,
            "top_count": None,
            "n_allowed": len(spec.allowed_values) if spec.allowed_values else None,
            "unobserved_allowed": None,
            "note": None,
        }
        if spec.role == "target":
            row["note"] = TARGET_WITHHELD
        elif spec.dtype in ("int", "float"):
            values = series.dropna().astype("float64")
            row.update(
                min=float(values.min()),
                p25=float(values.quantile(0.25)),
                median=float(values.median()),
                mean=round(float(values.mean()), 4),
                p75=float(values.quantile(0.75)),
                max=float(values.max()),
                std=round(float(values.std()), 4),
                n_zero=int((values == 0).sum()),
            )
        if spec.dtype in ("category", "string") or (spec.allowed_values and spec.role != "target"):
            counts = series.value_counts(dropna=True)
            if not counts.empty:
                row.update(top_value=str(counts.index[0]), top_count=int(counts.iloc[0]))
        if spec.allowed_values:
            observed = set(series.dropna().tolist())
            unobserved = [value for value in spec.allowed_values if value not in observed]
            row["unobserved_allowed"] = "|".join(str(value) for value in unobserved)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- duplicate detection


class DuplicateReport(_Record):
    """Duplicate findings. Only duplicate primary identifiers fail validation."""

    n_rows: int
    id_column: str
    n_duplicate_ids: int
    duplicate_ids: list[int]
    duplicate_secondary_identifiers: dict[str, int]
    n_exact_duplicate_rows: int
    exact_duplicate_row_ids: list[int]
    n_duplicate_feature_rows: int
    duplicate_feature_row_ids: list[int]
    n_conflicting_target_groups: int

    @property
    def has_duplicate_ids(self) -> bool:
        return self.n_duplicate_ids > 0

    @property
    def has_duplicate_rows(self) -> bool:
        return self.n_exact_duplicate_rows > 0 or self.n_duplicate_feature_rows > 0


def detect_duplicates(frame: pd.DataFrame, schema: SchemaConfig) -> DuplicateReport:
    """Find duplicate identifiers and duplicate records.

    * duplicate primary ``Id`` values (a schema violation, AC-004);
    * duplicate values in any other identifier column (e.g. ``PID``);
    * exact duplicate records: identical in every non-identifier column, target included;
    * duplicate feature rows: identical in every non-identifier, non-target column, and how
      many of those groups carry different target values (conflicting labels).

    ``Id`` lists report every row after the first occurrence of each duplicate group.
    """
    id_col = schema.id_column
    identifiers = [spec.name for spec in schema.with_role("identifier")]
    record_columns = [name for name in schema.names if name not in identifiers]
    feature_columns = [name for name in record_columns if name != schema.target]

    id_dupes = frame[id_col].duplicated(keep="first")
    exact_dupes = frame.duplicated(subset=record_columns, keep="first")
    feature_dupes = frame.duplicated(subset=feature_columns, keep="first")
    all_feature_dupes = frame.duplicated(subset=feature_columns, keep=False)
    conflicting = 0
    if all_feature_dupes.any():
        groups = frame.loc[all_feature_dupes].groupby(feature_columns, dropna=False)
        conflicting = int((groups[schema.target].nunique(dropna=False) > 1).sum())

    return DuplicateReport(
        n_rows=len(frame),
        id_column=id_col,
        n_duplicate_ids=int(id_dupes.sum()),
        duplicate_ids=sorted({int(v) for v in frame.loc[id_dupes, id_col]}),
        duplicate_secondary_identifiers={
            name: int(frame[name].duplicated(keep="first").sum())
            for name in identifiers
            if name != id_col
        },
        n_exact_duplicate_rows=int(exact_dupes.sum()),
        exact_duplicate_row_ids=[int(v) for v in frame.loc[exact_dupes, id_col]],
        n_duplicate_feature_rows=int(feature_dupes.sum()),
        duplicate_feature_row_ids=[int(v) for v in frame.loc[feature_dupes, id_col]],
        n_conflicting_target_groups=conflicting,
    )


# ------------------------------------------------------------------ data-quality flags


class QualityFlag(_Record):
    name: str
    description: str
    severity: Literal["info", "warning"]
    status: Literal["clear", "flagged", "skipped"]
    n_rows: int
    ids: list[int]


@dataclass(frozen=True)
class _Rule:
    name: str
    description: str
    severity: Literal["info", "warning"]
    columns: tuple[str, ...]
    predicate: Callable[[pd.DataFrame], pd.Series]


_GARAGE_CATS = ("GarageFinish", "GarageQual", "GarageCond")
_BSMT_CATS = ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2")


def _garage_partial(f: pd.DataFrame) -> pd.Series:
    others_missing = f[list(_GARAGE_CATS) + ["GarageYrBlt"]].isna().any(axis=1)
    return f["GarageType"].notna() & others_missing


def _garage_absent_inconsistent(f: pd.DataFrame) -> pd.Series:
    absent = f["GarageType"].isna()
    return absent & ((f["GarageArea"].fillna(0) != 0) | (f["GarageCars"].fillna(0) != 0))


def _bsmt_absent_inconsistent(f: pd.DataFrame) -> pd.Series:
    all_cats_missing = f[list(_BSMT_CATS)].isna().all(axis=1)
    return all_cats_missing & (f["TotalBsmtSF"].fillna(0) != 0)


def _sums_differ(total: str, parts: tuple[str, ...]) -> Callable[[pd.DataFrame], pd.Series]:
    def predicate(f: pd.DataFrame) -> pd.Series:
        complete = f[[total, *parts]].notna().all(axis=1)
        return complete & ((f[list(parts)].sum(axis=1) - f[total]).abs() > 0.5)

    return predicate


QUALITY_RULES: tuple[_Rule, ...] = (
    _Rule(
        "garage_partially_recorded",
        "A garage type is recorded but finish/quality/condition/year is missing (unknown "
        "values inside 'feature absent' columns).",
        "warning",
        ("GarageType", *_GARAGE_CATS, "GarageYrBlt"),
        _garage_partial,
    ),
    _Rule(
        "garage_absent_with_area",
        "No garage type recorded but garage area or car capacity is non-zero.",
        "warning",
        ("GarageType", "GarageArea", "GarageCars"),
        _garage_absent_inconsistent,
    ),
    _Rule(
        "basement_exposure_missing_with_basement",
        "Basement exists (BsmtQual recorded) but BsmtExposure is missing (DOC-02 §6.6).",
        "warning",
        ("BsmtQual", "BsmtExposure"),
        lambda f: f["BsmtQual"].notna() & f["BsmtExposure"].isna(),
    ),
    _Rule(
        "basement_fintype2_missing_with_area",
        "BsmtFinSF2 > 0 but BsmtFinType2 is missing (DOC-02 §6.6).",
        "warning",
        ("BsmtFinType2", "BsmtFinSF2"),
        lambda f: f["BsmtFinType2"].isna() & (f["BsmtFinSF2"].fillna(0) > 0),
    ),
    _Rule(
        "basement_areas_unrecorded",
        "Basement square-footage fields are missing (unknown, not zero).",
        "warning",
        ("TotalBsmtSF",),
        lambda f: f["TotalBsmtSF"].isna(),
    ),
    _Rule(
        "basement_absent_with_area",
        "All basement categoricals missing but TotalBsmtSF is non-zero.",
        "warning",
        (*_BSMT_CATS, "TotalBsmtSF"),
        _bsmt_absent_inconsistent,
    ),
    _Rule(
        "masvnr_type_area_missing_mismatch",
        "MasVnrType and MasVnrArea are not missing together.",
        "warning",
        ("MasVnrType", "MasVnrArea"),
        lambda f: f["MasVnrType"].isna() != f["MasVnrArea"].isna(),
    ),
    _Rule(
        "masvnr_none_with_area",
        "MasVnrType is 'None' but MasVnrArea > 0.",
        "warning",
        ("MasVnrType", "MasVnrArea"),
        lambda f: (f["MasVnrType"] == "None") & (f["MasVnrArea"].fillna(0) > 0),
    ),
    _Rule(
        "pool_area_without_quality",
        "PoolArea > 0 but PoolQC is missing.",
        "warning",
        ("PoolArea", "PoolQC"),
        lambda f: (f["PoolArea"] > 0) & f["PoolQC"].isna(),
    ),
    _Rule(
        "electrical_unrecorded",
        "Electrical system type is missing (genuinely unknown).",
        "info",
        ("Electrical",),
        lambda f: f["Electrical"].isna(),
    ),
    _Rule(
        "garage_built_after_sale",
        "GarageYrBlt is later than YrSold (e.g. the documented 2207 typo).",
        "warning",
        ("GarageYrBlt", "YrSold"),
        lambda f: f["GarageYrBlt"] > f["YrSold"],
    ),
    _Rule(
        "house_built_after_sale",
        "YearBuilt is later than YrSold (sold before completion).",
        "info",
        ("YearBuilt", "YrSold"),
        lambda f: f["YearBuilt"] > f["YrSold"],
    ),
    _Rule(
        "remodel_before_built",
        "YearRemodAdd is earlier than YearBuilt.",
        "warning",
        ("YearRemodAdd", "YearBuilt"),
        lambda f: f["YearRemodAdd"] < f["YearBuilt"],
    ),
    _Rule(
        "remodel_year_at_floor",
        "YearRemodAdd = 1950 for a house built before 1950 (recording floor, DOC-02 §10.5).",
        "info",
        ("YearRemodAdd", "YearBuilt"),
        lambda f: (f["YearRemodAdd"] == 1950) & (f["YearBuilt"] < 1950),
    ),
    _Rule(
        "grlivarea_components_differ",
        "GrLivArea differs from 1stFlrSF + 2ndFlrSF + LowQualFinSF.",
        "warning",
        ("GrLivArea", "1stFlrSF", "2ndFlrSF", "LowQualFinSF"),
        _sums_differ("GrLivArea", ("1stFlrSF", "2ndFlrSF", "LowQualFinSF")),
    ),
    _Rule(
        "total_bsmt_components_differ",
        "TotalBsmtSF differs from BsmtFinSF1 + BsmtFinSF2 + BsmtUnfSF.",
        "warning",
        ("TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF"),
        _sums_differ("TotalBsmtSF", ("BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF")),
    ),
)


def quality_flags(
    frame: pd.DataFrame,
    id_column: str,
    scope_column: str | None = None,
    scope_threshold: float | None = None,
) -> list[QualityFlag]:
    """Evaluate every consistency rule; a rule whose columns are absent is skipped."""
    rules = list(QUALITY_RULES)
    if scope_column is not None and scope_threshold is not None:
        rules.append(
            _Rule(
                "out_of_scope_rows",
                f"{scope_column} > {scope_threshold:g}: removed by the scope rule before "
                "the split (ADR-06).",
                "info",
                (scope_column,),
                lambda f: f[scope_column] > scope_threshold,
            )
        )
    flags = []
    for rule in rules:
        if not set(rule.columns) <= set(frame.columns):
            flags.append(
                QualityFlag(
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    status="skipped",
                    n_rows=0,
                    ids=[],
                )
            )
            continue
        mask = rule.predicate(frame).fillna(False).astype(bool)
        ids = [int(v) for v in frame.loc[mask, id_column]]
        flags.append(
            QualityFlag(
                name=rule.name,
                description=rule.description,
                severity=rule.severity,
                status="flagged" if ids else "clear",
                n_rows=len(ids),
                ids=ids,
            )
        )
    return flags


# ------------------------------------------------------------------------ metadata


class DatasetMetadata(_Record):
    dataset: str
    raw_path: str
    raw_sha256: str
    size_bytes: int
    n_rows: int
    n_columns: int
    columns_by_role: dict[str, int]
    columns_by_dtype: dict[str, int]
    n_model_input_columns: int
    missing_tokens: list[str]
    total_missing_cells: int
    columns_with_missing: int
    rows_with_any_missing: int
    config_hash: str
    schema_hash: str
    library_versions: dict[str, str]
    generated_at_utc: str


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("pandas", "numpy", "pandera", "pydantic", "scikit-learn", "pyyaml"):
        versions[package] = _package_version(package)
    return versions


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def dataset_metadata(
    frame: pd.DataFrame, config: ProjectConfig, raw_sha256: str
) -> DatasetMetadata:
    """Describe the validated raw dataset and the exact configuration that validated it."""
    schema = config.schema
    missing = frame.isna()
    roles: dict[str, int] = {}
    dtypes: dict[str, int] = {}
    for spec in schema.columns:
        roles[spec.role] = roles.get(spec.role, 0) + 1
        dtypes[spec.dtype] = dtypes.get(spec.dtype, 0) + 1
    try:
        raw_path = str(config.raw_path.relative_to(config.root).as_posix())
    except ValueError:
        raw_path = str(config.raw_path)
    return DatasetMetadata(
        dataset=config.data.dataset_name,
        raw_path=raw_path,
        raw_sha256=raw_sha256,
        size_bytes=config.raw_path.stat().st_size,
        n_rows=len(frame),
        n_columns=frame.shape[1],
        columns_by_role=roles,
        columns_by_dtype=dtypes,
        n_model_input_columns=roles.get("model_input", 0),
        missing_tokens=list(config.data.missing_tokens),
        total_missing_cells=int(missing.to_numpy().sum()),
        columns_with_missing=int(missing.any(axis=0).sum()),
        rows_with_any_missing=int(missing.any(axis=1).sum()),
        config_hash=config.config_hash,
        schema_hash=config.schema_hash,
        library_versions=library_versions(),
        generated_at_utc=utc_now(),
    )

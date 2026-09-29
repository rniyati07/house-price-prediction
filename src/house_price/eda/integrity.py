"""Data integrity report, E-01 to E-04 (DOC-02 §13.1; full raw file, no target relationships)."""

from __future__ import annotations

import pandas as pd

from house_price.config import SchemaConfig
from house_price.data.load import read_text_csv
from house_price.data.profile import detect_duplicates
from house_price.eda.context import EDAContext
from house_price.eda.outputs import Outputs, save_table

_PANDAS_DTYPE = {"int": "int64", "float": "float64"}


def e01_shape_schema(raw: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """E-01: every column with its declared contract and the dtype actually loaded."""
    rows = []
    for spec in schema.columns:
        loaded = str(raw[spec.name].dtype)
        expected = _PANDAS_DTYPE.get(spec.dtype, "str")
        rows.append({
            "column": spec.name,
            "source_name": spec.source_name,
            "role": spec.role,
            "schema_dtype": spec.dtype,
            "loaded_dtype": loaded,
            "dtype_matches": loaded == expected,
            "nullable": spec.nullable,
            "n_missing": int(raw[spec.name].isna().sum()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e02_value_counts(raw: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """E-02 (detail): observed count of every value in every column with allowed values."""
    frames = []
    for spec in schema.columns:
        if not spec.allowed_values:
            continue
        counts = raw[spec.name].value_counts(dropna=False)
        allowed = set(spec.allowed_values)
        frames.append(pd.DataFrame({
            "column": spec.name,
            "value": ["<missing>" if pd.isna(v) else str(v) for v in counts.index],
            "count": counts.to_numpy(),
            "allowed": [pd.isna(v) or v in allowed for v in counts.index],
        }))  # fmt: skip
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["column", "count", "value"], ascending=[True, False, True],
                             ignore_index=True)  # fmt: skip


def e02_allowed_values_audit(raw: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """E-02: observed values versus the allowed codes (dictionary plus file spellings)."""
    rows = []
    for spec in schema.columns:
        if not spec.allowed_values:
            continue
        observed = set(raw[spec.name].dropna().tolist())
        allowed = list(spec.allowed_values)
        rows.append({
            "column": spec.name,
            "n_allowed": len(allowed),
            "n_observed": len(observed),
            "not_allowed": "|".join(sorted(str(v) for v in observed - set(allowed))),
            "allowed_unobserved": "|".join(str(v) for v in allowed if v not in observed),
            "note": spec.note or "",
        })  # fmt: skip
    return pd.DataFrame(rows)


def e03_key_integrity(ctx: EDAContext) -> pd.DataFrame:
    """E-03: identifier uniqueness, ``SalePrice > 0``, and the confirmed raw-file hash."""
    raw, target = ctx.raw, ctx.target
    duplicates = detect_duplicates(raw, ctx.schema)
    non_positive = int((raw[target] <= 0).sum())
    checks = [
        ("row_count", str(len(raw)), "rows in the raw file"),
        ("column_count", str(raw.shape[1]), "columns in the raw file"),
        (f"{ctx.id_column}_unique", str(duplicates.n_duplicate_ids == 0),
         f"{duplicates.n_duplicate_ids} duplicate value(s)"),
        *[(f"{name}_unique", str(count == 0), f"{count} duplicate value(s)")
          for name, count in duplicates.duplicate_secondary_identifiers.items()],
        (f"{target}_positive", str(non_positive == 0), f"{non_positive} value(s) <= 0"),
        ("raw_sha256_confirmed", str(ctx.raw_sha256 == ctx.config.data.raw_sha256),
         ctx.raw_sha256),
    ]  # fmt: skip
    return pd.DataFrame(checks, columns=["check", "result", "detail"])


def e04_parsing_comparison(ctx: EDAContext) -> pd.DataFrame:
    """E-04: per-column missing counts under the explicit contract versus pandas defaults."""
    path, schema = ctx.config.raw_path, ctx.schema
    explicit = read_text_csv(path, ctx.config.data.missing_tokens).isna().sum()
    default = pd.read_csv(path, dtype=str).isna().sum()
    rows = [
        {
            "column": spec.name,
            "explicit_missing": int(explicit[spec.source_name]),
            "default_missing": int(default[spec.source_name]),
            "difference": int(default[spec.source_name] - explicit[spec.source_name]),
        }
        for spec in schema.columns
    ]
    return pd.DataFrame(rows)


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-01 to E-04."""
    out = Outputs()
    raw = ctx.raw_features
    save_table(out, ctx.tables_dir, "E-01_shape_schema", e01_shape_schema(ctx.raw, ctx.schema))
    save_table(out, ctx.tables_dir, "E-02_allowed_values_audit",
               e02_allowed_values_audit(raw, ctx.schema))  # fmt: skip
    save_table(out, ctx.tables_dir, "E-02_value_counts", e02_value_counts(raw, ctx.schema))
    save_table(out, ctx.tables_dir, "E-03_key_integrity", e03_key_integrity(ctx))
    save_table(out, ctx.tables_dir, "E-04_parsing_comparison", e04_parsing_comparison(ctx))
    return out

"""Raw data loading: file checks, SHA-256 verification, explicit parsing, casting (DOC-03 §5.2).

Implements FR-001 (raw file never modified), FR-002 (hash verification) and FR-003
(explicit missing tokens, DN-18). Every column is read as text first and then cast to its
``schema.yaml`` dtype, so casting failures are reported as schema violations rather than
parser crashes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from house_price.config import ProjectConfig, SchemaConfig
from house_price.data.errors import FAILURE_COLUMNS, DataIntegrityError, RawFileError
from house_price.data.schema import validate_raw

_CHUNK_SIZE = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes, read in chunks (the file is never written)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_file(path: Path) -> None:
    """Fail clearly if the raw dataset is absent, not a regular file, or empty."""
    if not path.exists():
        raise RawFileError(f"raw dataset not found: {path}")
    if not path.is_file():
        raise RawFileError(f"raw dataset path is not a regular file: {path}")
    if path.stat().st_size == 0:
        raise RawFileError(f"raw dataset is empty: {path}")


def verify_sha256(path: Path, expected: str) -> str:
    """Compare the file hash with the committed value; raise on mismatch (AC-002)."""
    actual = sha256_file(path)
    if actual != expected:
        raise DataIntegrityError(str(path), expected, actual)
    return actual


def read_text_csv(path: Path, missing_tokens: Sequence[str]) -> pd.DataFrame:
    """Read every column as text; only ``missing_tokens`` become missing (DN-18).

    pandas' default token list is disabled, so the literal text ``None`` (``MasVnrType``)
    stays a category instead of silently becoming missing (DOC-02 §6.5).
    """
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=list(missing_tokens),
        na_filter=True,
    )


def rename_source_columns(frame: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """Rename raw-file headers to canonical names. Unknown headers are left for validation."""
    return frame.rename(columns=schema.source_to_name)


@dataclass(frozen=True)
class CastResult:
    frame: pd.DataFrame
    failures: pd.DataFrame


def cast_to_schema(text_frame: pd.DataFrame, schema: SchemaConfig) -> CastResult:
    """Cast text columns to their declared dtypes and collect casting failures.

    A value that cannot be parsed as a number (or, for ``int`` columns, is not a whole
    number) is recorded as a failure of check ``cast_to_<dtype>`` with its row index, and
    becomes missing in the returned frame. ``int`` columns that end up containing missing
    values are left as ``float64`` so the Pandera dtype and nullability checks report them.
    """
    frame = text_frame.copy()
    failures: list[pd.DataFrame] = []
    for spec in schema.columns:
        if spec.name not in frame.columns or spec.dtype not in ("int", "float"):
            continue
        text = frame[spec.name]
        numeric = pd.to_numeric(text, errors="coerce").astype("float64")
        bad = text.notna() & numeric.isna()
        if spec.dtype == "int":
            bad |= numeric.notna() & (numeric % 1 != 0)
        if bad.any():
            failures.append(
                pd.DataFrame(
                    {
                        "column": spec.name,
                        "check": f"cast_to_{spec.dtype}",
                        "index": text.index[bad],
                        "failure_case": text[bad].to_numpy(),
                    }
                )
            )
            numeric = numeric.mask(bad)
        if spec.dtype == "int" and numeric.notna().all():
            frame[spec.name] = numeric.astype("int64")
        else:
            frame[spec.name] = numeric
    failure_frame = (
        pd.concat(failures, ignore_index=True)
        if failures
        else pd.DataFrame(columns=FAILURE_COLUMNS)
    )
    return CastResult(frame=frame, failures=failure_frame)


def read_typed_csv(
    path: Path,
    schema: SchemaConfig,
    missing_tokens: Sequence[str],
    *,
    source_headers: bool,
) -> CastResult:
    """Text-first read, optional header renaming, and casting. No validation."""
    frame = read_text_csv(path, missing_tokens)
    if source_headers:
        frame = rename_source_columns(frame, schema)
    return cast_to_schema(frame, schema)


def load_raw(config: ProjectConfig) -> pd.DataFrame:
    """Load the raw dataset exactly as specified and return it validated (DOC-03 §5.2).

    1. The file must exist and be non-empty.
    2. Its SHA-256 must equal ``data.yaml: raw_sha256``; otherwise ``DataIntegrityError``
       is raised before anything is parsed or written.
    3. Parse with explicit missing tokens, text first; rename headers to canonical names.
    4. Cast to ``schema.yaml`` dtypes and validate lazily against the ingestion schema.

    The raw file is only ever opened for reading (FR-001).
    """
    path = config.raw_path
    verify_raw_file(path)
    verify_sha256(path, config.data.raw_sha256)
    cast = read_typed_csv(path, config.schema, config.data.missing_tokens, source_headers=True)
    return validate_raw(cast.frame, config.schema, cast_failures=cast.failures)

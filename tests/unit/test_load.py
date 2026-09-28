"""Raw loading: file checks, hashing, explicit NA parsing, raw file untouched (AC-001 to AC-003)."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from house_price.data import split, validate
from house_price.data.errors import DataIntegrityError, RawFileError
from house_price.data.load import (
    load_raw,
    read_text_csv,
    sha256_file,
    verify_raw_file,
    verify_sha256,
)
from tests.conftest import FIXTURE_CSV, FIXTURE_ROWS, SampleEnv, make_env

# ------------------------------------------------------------------------- hashing


def test_sha256_of_known_content(tmp_path: Path) -> None:
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_sha256_matches_hashlib_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "big.bin"
    payload = os.urandom(3 * (1 << 20) + 17)  # spans several read chunks
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_changes_when_one_byte_changes(tmp_path: Path) -> None:
    original = FIXTURE_CSV.read_bytes()
    modified = bytearray(original)
    modified[-2] = ord("9") if modified[-2] != ord("9") else ord("8")
    (tmp_path / "a.csv").write_bytes(original)
    (tmp_path / "b.csv").write_bytes(bytes(modified))
    assert sha256_file(tmp_path / "a.csv") != sha256_file(tmp_path / "b.csv")


def test_verify_sha256_mismatch_names_both_hashes(tmp_path: Path) -> None:
    path = tmp_path / "file.csv"
    path.write_bytes(b"content")
    expected = "0" * 64
    with pytest.raises(DataIntegrityError) as info:
        verify_sha256(path, expected)
    assert expected in str(info.value)
    assert sha256_file(path) in str(info.value)


# ------------------------------------------------------------------ file existence


def test_missing_raw_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RawFileError, match="not found"):
        verify_raw_file(tmp_path / "absent.csv")


def test_directory_instead_of_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RawFileError, match="not a regular file"):
        verify_raw_file(tmp_path)


def test_empty_raw_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(RawFileError, match="empty"):
        verify_raw_file(path)


def test_load_raw_fails_when_file_is_missing(sample_env: SampleEnv) -> None:
    config = sample_env.load()
    sample_env.raw_path.unlink()
    with pytest.raises(RawFileError):
        load_raw(config)


# ------------------------------------------------------------------- successful load


def test_load_raw_returns_typed_canonical_frame(sample_env: SampleEnv) -> None:
    config = sample_env.load()
    frame = load_raw(config)
    assert frame.shape == (FIXTURE_ROWS, 82)
    assert list(frame.columns) == config.schema.names
    for spec in config.schema.columns:
        dtype = frame[spec.name].dtype
        if spec.dtype == "int":
            assert dtype == "int64", spec.name
        elif spec.dtype == "float":
            assert dtype == "float64", spec.name
        else:
            assert pd.api.types.is_string_dtype(dtype), spec.name
    assert frame["PID"].str.fullmatch(r"\d{10}").all()
    assert frame["PID"].str.startswith("0").any()  # leading zeros preserved
    assert frame["MSSubClass"].dtype == "int64"  # "020" parsed as 20
    assert frame["Id"].is_unique


def test_file_spellings_are_preserved(sample_env: SampleEnv) -> None:
    frame = load_raw(sample_env.load())
    assert "WD " in set(frame["SaleType"])  # trailing space kept: no value repair
    assert "C (all)" in set(frame["MSZoning"])


# -------------------------------------------------------------- AC-001 / AC-002


def test_raw_file_unchanged_by_loading(sample_env: SampleEnv) -> None:
    """AC-001: the raw CSV is byte-identical after ingestion."""
    before_hash = sha256_file(sample_env.raw_path)
    before_mtime = sample_env.raw_path.stat().st_mtime_ns
    load_raw(sample_env.load())
    assert sha256_file(sample_env.raw_path) == before_hash
    assert sample_env.raw_path.stat().st_mtime_ns == before_mtime


def _modified_env(tmp_path: Path) -> SampleEnv:
    """A fixture copy with one changed byte, while data.yaml keeps the original hash."""
    original = FIXTURE_CSV.read_bytes()
    last_digit = original[-2:-1]  # final SalePrice digit, before the trailing newline
    modified = original[:-2] + (b"1" if last_digit != b"1" else b"2") + original[-1:]
    assert modified != original
    return make_env(tmp_path, modified, committed_sha256=sha256_file(FIXTURE_CSV))


def test_modified_file_is_refused(tmp_path: Path) -> None:
    """AC-002: a hash mismatch raises before anything is parsed."""
    env = _modified_env(tmp_path)
    with pytest.raises(DataIntegrityError, match="SHA-256 mismatch"):
        load_raw(env.load())


def test_modified_file_stops_split_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """AC-002: non-zero exit, an error naming the mismatch, and no split file written."""
    env = _modified_env(tmp_path)
    assert split.main(env.cli_args()) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not (env.root / "data" / "processed").exists()


def test_modified_file_stops_validation_report_and_writes_nothing(tmp_path: Path) -> None:
    env = _modified_env(tmp_path)
    assert validate.main(env.cli_args()) == 1
    assert not (env.root / "reports").exists()


# ------------------------------------------------------------------------- AC-003


def _token_counts(path: Path, tokens: set[str]) -> dict[str, int]:
    """Count missing tokens directly in the CSV text, independent of pandas."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        counts = dict.fromkeys(header, 0)
        for row in reader:
            for name, value in zip(header, row, strict=True):
                counts[name] += value in tokens
    return counts


def test_missing_counts_match_explicit_tokens_and_are_stable(sample_env: SampleEnv) -> None:
    """AC-003: per-column missing counts equal the NA/empty tokens in the file, every run."""
    config = sample_env.load()
    first = load_raw(config).isna().sum()
    second = load_raw(config).isna().sum()
    pd.testing.assert_series_equal(first, second)

    expected = _token_counts(sample_env.raw_path, set(config.data.missing_tokens))
    by_source = {spec.source_name: spec.name for spec in config.schema.columns}
    for source, count in expected.items():
        assert first[by_source[source]] == count, source


def test_masvnrtype_none_text_is_a_category_not_missing(sample_env: SampleEnv) -> None:
    """DOC-02 §6.5: only NA tokens count as missing; literal "None" is a category."""
    config = sample_env.load()
    frame = load_raw(config)
    token_count = _token_counts(sample_env.raw_path, {"NA", ""})["Mas Vnr Type"]
    none_text = _token_counts(sample_env.raw_path, {"None"})["Mas Vnr Type"]
    assert none_text > 0
    assert frame["MasVnrType"].isna().sum() == token_count
    assert (frame["MasVnrType"] == "None").sum() == none_text


def test_library_default_parsing_would_inflate_masvnrtype_missing(sample_env: SampleEnv) -> None:
    """Why explicit tokens matter: pandas' defaults also turn "None" into missing."""
    explicit = read_text_csv(sample_env.raw_path, ["NA", ""])["Mas Vnr Type"].isna().sum()
    default = pd.read_csv(sample_env.raw_path, dtype=str)["Mas Vnr Type"].isna().sum()
    assert default > explicit


# ------------------------------------------------------------------- real dataset


@pytest.mark.data
def test_real_dataset_loads(real_config) -> None:  # type: ignore[no-untyped-def]
    frame = load_raw(real_config)
    assert frame.shape == (2930, 82)
    assert frame["Id"].is_unique and frame["PID"].is_unique
    assert frame["MasVnrType"].isna().sum() == 23
    assert (frame["MasVnrType"] == "None").sum() == 1752
    assert int(frame.isna().to_numpy().sum()) == 13997

"""Automated validation report generation (python -m house_price.data.validate)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from house_price.data import validate
from house_price.data.validate import (
    COLUMN_PROFILE_CSV,
    FAILURES_CSV,
    METADATA_JSON,
    MISSING_CSV,
    QUALITY_CSV,
    REPORT_JSON,
    REPORT_MD,
    ValidationReport,
    build_report,
)
from tests.conftest import FIXTURE_CSV, FIXTURE_ROWS, SampleEnv, make_env

REPORT_FILES = [REPORT_JSON, REPORT_MD, METADATA_JSON, COLUMN_PROFILE_CSV, MISSING_CSV, QUALITY_CSV]


def _reports_dir(env: SampleEnv) -> Path:
    return env.root / "reports" / "data_validation"


def _read_report(env: SampleEnv) -> ValidationReport:
    text = (_reports_dir(env) / REPORT_JSON).read_text(encoding="utf-8")
    return ValidationReport.model_validate_json(text)


def _bad_category_env(tmp_path: Path) -> SampleEnv:
    """A fixture copy with an invalid Neighborhood and a matching committed hash."""
    original = FIXTURE_CSV.read_bytes()
    modified = original.replace(b",NAmes,", b",Atlantis,", 1)
    assert modified != original
    return make_env(tmp_path, modified)


def test_report_is_generated_for_a_valid_dataset(
    sample_env: SampleEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    assert validate.main(sample_env.cli_args()) == 0
    out_dir = _reports_dir(sample_env)
    for name in REPORT_FILES:
        assert (out_dir / name).is_file(), name
    assert not (out_dir / FAILURES_CSV).exists()
    assert "Validation PASSED" in capsys.readouterr().out

    report = _read_report(sample_env)
    assert report.status == "passed"
    assert report.raw_path == "data/raw/train.csv"
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["raw_file_exists"] == "passed"
    assert statuses["raw_sha256_matches"] == "passed"
    assert statuses["ingestion_schema"] == "passed"
    assert statuses["duplicate_ids"] == "passed"
    assert statuses["scope_rule_preview"] == "passed"
    assert report.metadata is not None and report.metadata.n_rows == FIXTURE_ROWS
    assert report.duplicates is not None
    assert report.missing_values and report.quality_flags


def test_report_tables_and_markdown(sample_env: SampleEnv) -> None:
    assert validate.main(sample_env.cli_args()) == 0
    out_dir = _reports_dir(sample_env)
    profile = pd.read_csv(out_dir / COLUMN_PROFILE_CSV)
    assert len(profile) == 82
    missing = pd.read_csv(out_dir / MISSING_CSV)
    assert missing["n_missing"].gt(0).all()
    flags = pd.read_csv(out_dir / QUALITY_CSV)
    assert "garage_partially_recorded" in set(flags["name"])
    metadata = json.loads((out_dir / METADATA_JSON).read_text(encoding="utf-8"))
    assert metadata["raw_sha256"] == sample_env.load().data.raw_sha256
    markdown = (out_dir / REPORT_MD).read_text(encoding="utf-8")
    for heading in (
        "# Data Validation Report",
        "## Checks",
        "## Dataset Metadata",
        "## Duplicates",
        "## Missing Values",
        "## Data-Quality Flags",
    ):
        assert heading in markdown


def test_report_is_reproducible_apart_from_timestamp(sample_env: SampleEnv) -> None:
    config = sample_env.load()
    first = build_report(config).report.model_dump(exclude={"generated_at_utc", "metadata"})
    second = build_report(config).report.model_dump(exclude={"generated_at_utc", "metadata"})
    assert first == second


def test_schema_failure_writes_failed_report(tmp_path: Path) -> None:
    env = _bad_category_env(tmp_path)
    assert validate.main(env.cli_args()) == 1
    report = _read_report(env)
    assert report.status == "failed"
    assert report.metadata is None
    assert any(
        f["column"] == "Neighborhood" and "isin" in str(f["check"]) for f in report.schema_failures
    )
    failures = pd.read_csv(_reports_dir(env) / FAILURES_CSV)
    assert "Atlantis" in set(failures["failure_case"])


def test_stale_failure_file_is_removed_after_a_passing_run(tmp_path: Path) -> None:
    env = _bad_category_env(tmp_path)
    assert validate.main(env.cli_args()) == 1
    assert (_reports_dir(env) / FAILURES_CSV).exists()
    fixed = make_env(tmp_path)  # restore the valid fixture and its hash
    assert validate.main(fixed.cli_args()) == 0
    assert not (_reports_dir(env) / FAILURES_CSV).exists()


def test_hash_mismatch_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = make_env(tmp_path, committed_sha256="0" * 64)
    assert validate.main(env.cli_args()) == 1
    assert "SHA-256 mismatch" in capsys.readouterr().err
    assert not (env.root / "reports").exists()


def test_missing_raw_file_writes_nothing(
    sample_env: SampleEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_env.raw_path.unlink()
    assert validate.main(sample_env.cli_args()) == 1
    assert "not found" in capsys.readouterr().err
    assert not (sample_env.root / "reports").exists()

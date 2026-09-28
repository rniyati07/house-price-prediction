"""Automated data validation report (``make validate-data``; DOC-03 §17.2).

Runs the file checks, the SHA-256 check, and the ingestion schema, then profiles the
validated dataset and writes the report to ``data.yaml: reports_dir``:

* ``validation_report.json`` / ``validation_report.md``: checks, metadata, duplicates,
  missing values, data-quality flags
* ``dataset_metadata.json``: dataset metadata and hashes
* ``column_profile.csv``, ``missing_values.csv``, ``quality_flags.csv``: profiling tables
* ``schema_failures.csv``: every schema failure (only when validation fails)

If the file is missing or its hash differs, nothing is written and the exit code is 1
(AC-002). A schema failure writes a ``failed`` report and exits with 1.

Run ``python -m house_price.data.validate``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from house_price.config import DEFAULT_CONFIG_DIR, ConfigError, ProjectConfig, load_project_config
from house_price.data.errors import DataError, DataValidationError
from house_price.data.load import read_typed_csv, verify_raw_file, verify_sha256
from house_price.data.profile import (
    DatasetMetadata,
    DuplicateReport,
    QualityFlag,
    column_profile,
    dataset_metadata,
    detect_duplicates,
    missing_value_profile,
    quality_flags,
    utc_now,
)
from house_price.data.schema import validate_raw

REPORT_JSON = "validation_report.json"
REPORT_MD = "validation_report.md"
METADATA_JSON = "dataset_metadata.json"
COLUMN_PROFILE_CSV = "column_profile.csv"
MISSING_CSV = "missing_values.csv"
QUALITY_CSV = "quality_flags.csv"
FAILURES_CSV = "schema_failures.csv"

CheckStatus = Literal["passed", "failed", "warning"]


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: CheckStatus
    detail: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["passed", "failed"]
    generated_at_utc: str
    raw_path: str
    raw_sha256: str
    checks: list[CheckResult]
    schema_failures: list[dict[str, object]]
    metadata: DatasetMetadata | None
    duplicates: DuplicateReport | None
    missing_values: list[dict[str, object]]
    quality_flags: list[QualityFlag]
    outputs: list[str]


@dataclass
class ReportBundle:
    """The report plus the tables written next to it."""

    report: ValidationReport
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)


def _status(checks: Sequence[CheckResult]) -> Literal["passed", "failed"]:
    return "failed" if any(check.status == "failed" for check in checks) else "passed"


def build_report(config: ProjectConfig) -> ReportBundle:
    """Validate and profile the raw dataset.

    File and hash errors propagate (nothing may be written after them); a schema failure
    produces a ``failed`` report instead of an exception.
    """
    path = config.raw_path
    display_path = config.data.raw_path.as_posix()
    verify_raw_file(path)
    raw_sha256 = verify_sha256(path, config.data.raw_sha256)
    checks = [
        CheckResult(
            name="raw_file_exists",
            status="passed",
            detail=f"{display_path} exists and is non-empty",
        ),
        CheckResult(
            name="raw_sha256_matches",
            status="passed",
            detail=f"SHA-256 {raw_sha256} matches configs/data.yaml",
        ),
    ]
    cast = read_typed_csv(path, config.schema, config.data.missing_tokens, source_headers=True)
    try:
        frame = validate_raw(cast.frame, config.schema, cast_failures=cast.failures)
    except DataValidationError as exc:
        summary = exc.summary()
        checks.append(
            CheckResult(
                name="ingestion_schema",
                status="failed",
                detail=f"{len(exc.failure_cases)} failure case(s) in {len(summary)} column/check pair(s)",
            )
        )
        report = ValidationReport(
            status="failed",
            generated_at_utc=utc_now(),
            raw_path=display_path,
            raw_sha256=raw_sha256,
            checks=checks,
            schema_failures=json.loads(summary.to_json(orient="records")),
            metadata=None,
            duplicates=None,
            missing_values=[],
            quality_flags=[],
            outputs=[REPORT_JSON, REPORT_MD, FAILURES_CSV],
        )
        return ReportBundle(report=report, tables={FAILURES_CSV: exc.failure_cases})

    schema = config.schema
    checks.append(
        CheckResult(
            name="ingestion_schema",
            status="passed",
            detail=f"{frame.shape[0]} rows x {frame.shape[1]} columns: presence, dtypes, allowed "
            "values, ranges, nullability, unique identifiers, SalePrice > 0",
        )
    )
    duplicates = detect_duplicates(frame, schema)
    checks.append(
        CheckResult(
            name="duplicate_ids",
            status="failed" if duplicates.has_duplicate_ids else "passed",
            detail=f"{duplicates.n_duplicate_ids} duplicate {duplicates.id_column} value(s)",
        )
    )
    checks.append(
        CheckResult(
            name="duplicate_records",
            status="warning" if duplicates.has_duplicate_rows else "passed",
            detail=f"{duplicates.n_exact_duplicate_rows} exact duplicate record(s), "
            f"{duplicates.n_duplicate_feature_rows} duplicate feature row(s), "
            f"{duplicates.n_conflicting_target_groups} group(s) with conflicting targets "
            "(reported only; never removed)",
        )
    )
    scope = config.data.scope
    n_in_scope = int((frame[scope.column] <= scope.threshold).sum())
    checks.append(
        CheckResult(
            name="scope_rule_preview",
            status="passed" if n_in_scope == scope.expected_rows_after else "warning",
            detail=f"{len(frame) - n_in_scope} row(s) with {scope.column} > {scope.threshold:g}; "
            f"{n_in_scope} in scope (expected {scope.expected_rows_after})",
        )
    )
    missing = missing_value_profile(frame, schema)
    flags = quality_flags(frame, schema.id_column, scope.column, scope.threshold)
    flagged = [flag for flag in flags if flag.status == "flagged" and flag.severity == "warning"]
    checks.append(
        CheckResult(
            name="data_quality_flags",
            status="warning" if flagged else "passed",
            detail=f"{len(flagged)} consistency rule(s) flagged; see quality flags "
            "(informational, no values are changed)",
        )
    )
    metadata = dataset_metadata(frame, config, raw_sha256)
    report = ValidationReport(
        status=_status(checks),
        generated_at_utc=metadata.generated_at_utc,
        raw_path=display_path,
        raw_sha256=raw_sha256,
        checks=checks,
        schema_failures=[],
        metadata=metadata,
        duplicates=duplicates,
        missing_values=json.loads(missing.to_json(orient="records")),
        quality_flags=flags,
        outputs=[
            REPORT_JSON,
            REPORT_MD,
            METADATA_JSON,
            COLUMN_PROFILE_CSV,
            MISSING_CSV,
            QUALITY_CSV,
        ],
    )
    flag_table = pd.DataFrame(
        [
            {**flag.model_dump(exclude={"ids"}), "ids": "|".join(map(str, flag.ids))}
            for flag in flags
        ]
    )
    return ReportBundle(
        report=report,
        tables={
            COLUMN_PROFILE_CSV: column_profile(frame, schema),
            MISSING_CSV: missing,
            QUALITY_CSV: flag_table,
        },
    )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for row in rows]
    return lines


def _ids(ids: Sequence[int], limit: int = 12) -> str:
    shown = ", ".join(str(i) for i in ids[:limit])
    return shown + (f", ... (+{len(ids) - limit})" if len(ids) > limit else "")


def render_markdown(report: ValidationReport) -> str:
    """Human-readable version of the report."""
    lines = [
        "# Data Validation Report",
        "",
        f"- **Status:** {report.status.upper()}",
        f"- **Generated (UTC):** {report.generated_at_utc}",
        f"- **Raw file:** `{report.raw_path}`",
        f"- **SHA-256:** `{report.raw_sha256}`",
        "",
        "Generated by `python -m house_price.data.validate`. Do not edit by hand.",
        "",
        "## Checks",
        "",
        *_md_table(
            ["Check", "Status", "Detail"], [(c.name, c.status, c.detail) for c in report.checks]
        ),
    ]
    if report.schema_failures:
        lines += [
            "",
            "## Schema Failures",
            "",
            *_md_table(
                ["Column", "Check", "Failures", "Example row indices"],
                [
                    (f["column"], f["check"], f["n_failures"], f["example_indices"])
                    for f in report.schema_failures
                ],
            ),
            "",
            f"Every failure case is listed in `{FAILURES_CSV}`.",
        ]
    if report.metadata is not None:
        m = report.metadata
        lines += [
            "",
            "## Dataset Metadata",
            "",
            *_md_table(
                ["Property", "Value"],
                [
                    ("Dataset", m.dataset),
                    ("Rows x columns", f"{m.n_rows} x {m.n_columns}"),
                    ("Size (bytes)", m.size_bytes),
                    (
                        "Columns by role",
                        ", ".join(f"{k}={v}" for k, v in m.columns_by_role.items()),
                    ),
                    (
                        "Columns by dtype",
                        ", ".join(f"{k}={v}" for k, v in m.columns_by_dtype.items()),
                    ),
                    ("Model-input columns", m.n_model_input_columns),
                    ("Missing tokens", ", ".join(repr(t) for t in m.missing_tokens)),
                    ("Missing cells", m.total_missing_cells),
                    ("Columns with missing values", m.columns_with_missing),
                    ("Rows with any missing value", m.rows_with_any_missing),
                    ("Config hash", f"`{m.config_hash}`"),
                    ("Schema hash", f"`{m.schema_hash}`"),
                    ("Libraries", ", ".join(f"{k} {v}" for k, v in m.library_versions.items())),
                ],
            ),
        ]
    if report.duplicates is not None:
        d = report.duplicates
        lines += [
            "",
            "## Duplicates",
            "",
            *_md_table(
                ["Finding", "Count", "Ids (after first occurrence)"],
                [
                    (f"Duplicate {d.id_column}", d.n_duplicate_ids, _ids(d.duplicate_ids)),
                    *[
                        (f"Duplicate {name}", count, "")
                        for name, count in d.duplicate_secondary_identifiers.items()
                    ],
                    (
                        "Exact duplicate records",
                        d.n_exact_duplicate_rows,
                        _ids(d.exact_duplicate_row_ids),
                    ),
                    (
                        "Duplicate feature rows",
                        d.n_duplicate_feature_rows,
                        _ids(d.duplicate_feature_row_ids),
                    ),
                    ("Groups with conflicting targets", d.n_conflicting_target_groups, ""),
                ],
            ),
        ]
    if report.missing_values:
        lines += [
            "",
            "## Missing Values",
            "",
            "Counted with the explicit token set only (DN-18).",
            "",
            *_md_table(
                ["Column", "Source name", "Dtype", "Missing", "%"],
                [
                    (r["column"], r["source_name"], r["dtype"], r["n_missing"], r["pct_missing"])
                    for r in report.missing_values
                ],
            ),
        ]
    if report.quality_flags:
        lines += [
            "",
            "## Data-Quality Flags",
            "",
            (
                "Informational consistency checks (DOC-02 §6.6). Nothing is changed; "
                "flagged items are recorded in `docs/data_card.md` for M3 verification."
            ),
            "",
            *_md_table(
                ["Rule", "Severity", "Status", "Rows", "Ids", "Description"],
                [
                    (f.name, f.severity, f.status, f.n_rows, _ids(f.ids), f.description)
                    for f in report.quality_flags
                ],
            ),
        ]
    lines += ["", "## Outputs", "", *[f"- `{name}`" for name in report.outputs], ""]
    return "\n".join(lines)


def write_report(bundle: ReportBundle, out_dir: Path) -> list[Path]:
    """Write the report, its Markdown rendering, and the profiling tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = bundle.report
    written = [out_dir / REPORT_JSON, out_dir / REPORT_MD]
    written[0].write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    written[1].write_text(render_markdown(report), encoding="utf-8")
    if report.metadata is not None:
        path = out_dir / METADATA_JSON
        path.write_text(report.metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(path)
    for name, table in bundle.tables.items():
        path = out_dir / name
        table.to_csv(path, index=False, lineterminator="\n")
        written.append(path)
    stale = out_dir / FAILURES_CSV
    if report.status == "passed" and stale.exists():
        stale.unlink()
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m house_price.data.validate",
        description="Verify the raw dataset (file, SHA-256, schema) and write the validation report.",
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--root", type=Path, default=None, help="base for relative paths")
    args = parser.parse_args(argv)
    try:
        config = load_project_config(args.config_dir, args.root)
        bundle = build_report(config)
    except (ConfigError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    written = write_report(bundle, config.reports_dir)
    report = bundle.report
    for check in report.checks:
        print(f"[{check.status.upper():7}] {check.name}: {check.detail}")
    print(f"Validation {report.status.upper()}. Report written to {config.reports_dir}:")
    for path in written:
        print(f"  - {path.name}")
    return 0 if report.status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

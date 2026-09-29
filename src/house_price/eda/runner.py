"""Run the whole M3 EDA: every deliverable, then the E-35 report and E-36 inputs.

``python -m house_price.eda`` reproduces all outputs; the notebooks call the same functions
one group at a time. Each execution also writes one M3 and one M4 run record under
``results/`` (see :mod:`house_price.results`); ``--record-only`` writes those records from
the saved deliverables without rerunning any analysis.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    ConfigError,
    config_hash,
    load_feature_config,
    load_project_config,
)
from house_price.data.errors import DataError
from house_price.data.load import sha256_file
from house_price.eda import (
    categorical,
    engineered,
    integrity,
    leakage_time,
    missing,
    numeric,
    outliers,
    report,
    summary,
    target,
)
from house_price.eda.context import (
    FIGURES_DIR,
    TABLES_DIR,
    EDAContext,
    EDAError,
    find_project_root,
)
from house_price.eda.outputs import Outputs, load_tables
from house_price.results import ResultRun, start_run

ANALYSES = (integrity, target, missing, numeric, categorical, outliers, engineered, leakage_time)
EXIT_ADR_DISCREPANCY = 2


def run_analyses(ctx: EDAContext) -> Outputs:
    """Compute and save E-01 to E-29 and E-31 to E-34."""
    outputs = Outputs()
    for module in ANALYSES:
        outputs.update(module.run(ctx))
    return outputs


def _record(
    m3: ResultRun,
    m4: ResultRun,
    *,
    config_dir: Path,
    root: Path,
    tables_dir: Path,
    figures_dir: Path,
    mode: str,
) -> None:
    """Write the M3 and M4 run records from the saved deliverables (never recomputed)."""
    config = load_project_config(config_dir, root)
    features = load_feature_config(config_dir)
    tables = load_tables(tables_dir)
    absent = summary.missing_tables(tables)
    if absent:
        raise EDAError(f"saved EDA tables missing from {tables_dir}: {absent}; "
                       "run python -m house_price.eda first")  # fmt: skip
    m3_files, m4_files = summary.deliverables(tables_dir, figures_dir)
    lineage = {
        "data_sha256": sha256_file(config.raw_path) if config.raw_path.is_file() else None,
        "split_manifest_sha256": (sha256_file(config.manifest_path)
                                  if config.manifest_path.is_file() else None),
        "config_hash": config_hash(config.data, config.validation, config.schema, features),
        "config_hash_covers": ["data", "validation", "schema", "features"],
        "seed": config.validation.seed,
    }  # fmt: skip
    source = ("analyses run by this execution" if mode == "full"
              else "saved deliverables read back; analyses not rerun")  # fmt: skip
    for run, files in ((m3, m3_files), (m4, m4_files)):
        run.log_lineage(**lineage)
        run.log_params({"mode": mode, "source": source, "tables_dir": tables_dir,
                        "figures_dir": figures_dir})  # fmt: skip
        run.log_value("deliverables", summary.source_summary(files))
        summary.register_artifacts(run, files)
    summary.record_m3(m3, tables)
    summary.record_m4(m4, tables, features)
    m4.log_artifact(tables_dir / "E-35_question_answers.csv", description="source of Q17")


def _full_run(
    args: argparse.Namespace, root: Path, config_dir: Path, m3: ResultRun, m4: ResultRun
) -> int:
    ctx = EDAContext.load(config_dir, root, args.tables_dir, args.figures_dir)
    outputs = run_analyses(ctx)
    result = report.build(ctx)
    print(f"{len(outputs.tables)} tables written to {ctx.tables_dir}")
    print(f"{len(outputs.figures)} figures written to {ctx.figures_dir}")
    register = result.register
    confirmed = int((register["status"] == "confirmed").sum())
    print(f"E-35 register: {confirmed} of {len(register)} documented properties confirmed")
    for row in register[register["status"] != "confirmed"].itertuples():
        print(f"  NOT CONFIRMED {row.property_id}: {row.documented_property} -> {row.observed}")
    print(f"Report: {result.path}")
    _record(m3, m4, config_dir=config_dir, root=root, tables_dir=ctx.tables_dir,
            figures_dir=ctx.figures_dir, mode="full")  # fmt: skip
    print(f"Run records: {m3.path} and {m4.path}")
    blocking = result.adr_fixed_discrepancies
    if not blocking.empty:
        ids = ", ".join(blocking["property_id"])
        m3.mark_failed(f"ADR-fixed number contradicted ({ids})")
        print(f"STOP: an ADR-fixed number is contradicted ({ids}); follow the ADR "
              "supersession process before continuing (DOC-02 §9.2).", file=sys.stderr)  # fmt: skip
        return EXIT_ADR_DISCREPANCY
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m house_price.eda",
        description="Produce every M3 EDA deliverable (DOC-02 §13).",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None, help="project root")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None, help="default: <root>/results")
    parser.add_argument(
        "--record-only", action="store_true",
        help="write the M3/M4 run records from the saved deliverables without rerunning",
    )  # fmt: skip
    args = parser.parse_args(argv)
    try:
        root = (args.root or find_project_root(Path.cwd())).resolve()
        config_dir = args.config_dir or root / DEFAULT_CONFIG_DIR
        with (
            start_run("M3", root=root, results_dir=args.results_dir,
                      entry_point="python -m house_price.eda", argv=argv) as m3,
            start_run("M4", root=root, results_dir=args.results_dir,
                      entry_point="python -m house_price.eda", argv=argv) as m4,
        ):  # fmt: skip
            m3.log_value("execution.shared_with", {"M4": m4.run_id})
            m4.log_value("execution.shared_with", {"M3": m3.run_id})
            if not args.record_only:
                return _full_run(args, root, config_dir, m3, m4)
            _record(m3, m4, config_dir=config_dir, root=root,
                    tables_dir=args.tables_dir or root / TABLES_DIR,
                    figures_dir=args.figures_dir or root / FIGURES_DIR, mode="record-only")  # fmt: skip
            print(f"Run records (from saved deliverables): {m3.path} and {m4.path}")
            return 0
    except (ConfigError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

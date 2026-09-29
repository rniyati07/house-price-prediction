"""Run the whole M3 EDA: every deliverable, then the E-35 report and E-36 inputs.

``python -m house_price.eda`` reproduces all outputs; the notebooks call the same functions
one group at a time.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from house_price.config import ConfigError
from house_price.data.errors import DataError
from house_price.eda import (
    categorical,
    integrity,
    leakage_time,
    missing,
    numeric,
    outliers,
    report,
    target,
)
from house_price.eda.context import EDAContext
from house_price.eda.outputs import Outputs

ANALYSES = (integrity, target, missing, numeric, categorical, outliers, leakage_time)
EXIT_ADR_DISCREPANCY = 2


def run_analyses(ctx: EDAContext) -> Outputs:
    """Compute and save E-01 to E-26 and E-31 to E-34."""
    outputs = Outputs()
    for module in ANALYSES:
        outputs.update(module.run(ctx))
    return outputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m house_price.eda",
        description="Produce every M3 EDA deliverable (DOC-02 §13).",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None, help="project root")
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        ctx = EDAContext.load(args.config_dir, args.root, args.tables_dir, args.figures_dir)
    except (ConfigError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
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
    blocking = result.adr_fixed_discrepancies
    if not blocking.empty:
        print("STOP: an ADR-fixed number is contradicted "
              f"({', '.join(blocking['property_id'])}); follow the ADR supersession process "
              "before continuing (DOC-02 §9.2).", file=sys.stderr)  # fmt: skip
        return EXIT_ADR_DISCREPANCY
    return 0

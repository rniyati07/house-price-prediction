"""Deterministic saving and loading of EDA tables and figures.

Tables are CSV with fixed float formatting and ``\\n`` line endings; figures are PNG written
by the non-interactive Agg backend without a software/date stamp, so reruns on the same
environment produce identical files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

FLOAT_FORMAT = "%.10g"
FIGURE_DPI = 100

sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class Outputs:
    """Tables and figures produced by one or more EDA modules, keyed by file stem."""

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)

    def update(self, other: Outputs) -> None:
        self.tables.update(other.tables)
        self.figures.update(other.figures)


def save_table(outputs: Outputs, directory: Path, name: str, table: pd.DataFrame) -> None:
    """Write ``table`` to ``directory/name.csv`` and register it in ``outputs``."""
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        directory / f"{name}.csv", index=False, float_format=FLOAT_FORMAT, lineterminator="\n"
    )
    outputs.tables[name] = table


def save_figure(outputs: Outputs, directory: Path, name: str, figure: Figure) -> None:
    """Write ``figure`` to ``directory/name.png``, close it, and register it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(figure)
    outputs.figures[name] = path


def load_tables(directory: Path) -> dict[str, pd.DataFrame]:
    """Read every saved ``E-*.csv`` table back, keyed by file stem."""
    return {
        path.stem: pd.read_csv(path, keep_default_na=False, na_values=[""])
        for path in sorted(directory.glob("E-*.csv"))
    }

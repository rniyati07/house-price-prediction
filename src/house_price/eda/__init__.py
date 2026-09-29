"""Exploratory data analysis for M3 (DOC-02 §13, DOC-05 M3).

Each module computes one group of DOC-02 deliverables and saves them under the names the
roadmap prescribes: tables to ``reports/eda/`` and figures to ``reports/figures/eda/``,
each file name starting with its deliverable ID (for example ``E-05_target_statistics.csv``).

This package is analysis only. It is never imported by training or serving code, and it
never opens the holdout file: target relationships use the development set, and the raw
file is used only for target-free profiling and the ADR-06 scope review (FR-009).
"""

from house_price.eda.context import EDAContext
from house_price.eda.outputs import Outputs

__all__ = ["EDAContext", "Outputs"]

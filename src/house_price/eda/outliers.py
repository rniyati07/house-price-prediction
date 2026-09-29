"""Scope-rule review, E-24 to E-26 (DOC-02 §9, §13.6).

This is the ADR-06 scope-review exception to FR-009: it must use the raw file, because the
homes above the threshold are removed before the split. It documents a decision already
fixed by the ADR and informs no other modeling choice.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from house_price.eda.context import EDAContext, log_price
from house_price.eda.outputs import Outputs, save_figure, save_table


def _out_of_scope(ctx: EDAContext) -> pd.Series:
    scope = ctx.config.data.scope
    return ctx.raw[scope.column] > scope.threshold


def _fit(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    slope, intercept = np.polyfit(x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64"), 1)
    return float(slope), float(intercept)


def e24_scope_figure(ctx: EDAContext) -> Figure:
    """E-24: ``GrLivArea`` against price (raw and log) with out-of-scope homes highlighted."""
    raw, scope = ctx.raw, ctx.config.data.scope
    outside = _out_of_scope(ctx)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scales = [
        ("SalePrice (dollars)", raw[ctx.target]),
        ("log1p(SalePrice)", log_price(raw, ctx.target)),
    ]
    for ax, (label, y) in zip(axes, scales, strict=True):
        ax.scatter(raw.loc[~outside, scope.column], y[~outside], s=5, alpha=0.35,
                   color="tab:blue", label="in scope")  # fmt: skip
        ax.scatter(raw.loc[outside, scope.column], y[outside], s=60, color="tab:red",
                   marker="X", label=f"{scope.column} > {scope.threshold:g}")  # fmt: skip
        for _, row in raw.loc[outside].iterrows():
            ax.annotate(f"Id {row[ctx.id_column]} ({row['SaleCondition']})",
                        (row[scope.column], y[row.name]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")  # fmt: skip
        ax.axvline(scope.threshold, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel(scope.column)
        ax.set_ylabel(label)
        ax.legend(loc="upper left")
    figure.suptitle("E-24 Scope rule: homes above the living-area threshold (raw file)")
    figure.tight_layout()
    return figure


def e25_scope_table(ctx: EDAContext) -> pd.DataFrame:
    """E-25: the out-of-scope homes, with their residual from the in-scope size trend."""
    raw, scope = ctx.raw, ctx.config.data.scope
    outside = _out_of_scope(ctx)
    y = log_price(raw, ctx.target)
    slope, intercept = _fit(raw.loc[~outside, scope.column], y[~outside])
    residual_sd = float((y[~outside] - (slope * raw.loc[~outside, scope.column] + intercept)).std())
    table = raw.loc[outside, [ctx.id_column, scope.column, ctx.target, "OverallQual",
                              "SaleCondition"]].copy()  # fmt: skip
    predicted = slope * table[scope.column] + intercept
    table["log_residual_vs_in_scope_trend"] = y[outside] - predicted
    table["residual_in_sd"] = table["log_residual_vs_in_scope_trend"] / residual_sd
    table["price_percentile_in_raw_file"] = raw[ctx.target].rank(pct=True)[outside] * 100
    return table.sort_values(ctx.id_column, ignore_index=True)


def e26_leverage(ctx: EDAContext) -> pd.DataFrame:
    """E-26: slope of log price on ``GrLivArea`` with and without the out-of-scope homes."""
    raw, column = ctx.raw, ctx.config.data.scope.column
    outside = _out_of_scope(ctx)
    y = log_price(raw, ctx.target)
    rows = []
    for label, mask in (("all raw rows", pd.Series(True, index=raw.index)),
                        ("in-scope rows only", ~outside)):  # fmt: skip
        slope, intercept = _fit(raw.loc[mask, column], y[mask])
        rows.append({"fit": label, "n": int(mask.sum()), "slope_per_1000_sqft": slope * 1000,
                     "intercept": intercept})  # fmt: skip
    table = pd.DataFrame(rows)
    base = table.loc[0, "slope_per_1000_sqft"]
    table["slope_change_pct_vs_all"] = 100 * (table["slope_per_1000_sqft"] - base) / base
    return table


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-24 to E-26."""
    out = Outputs()
    save_figure(out, ctx.figures_dir, "E-24_scope_rule", e24_scope_figure(ctx))
    save_table(out, ctx.tables_dir, "E-25_out_of_scope_homes", e25_scope_table(ctx))
    save_table(out, ctx.tables_dir, "E-26_leverage_comparison", e26_leverage(ctx))
    return out

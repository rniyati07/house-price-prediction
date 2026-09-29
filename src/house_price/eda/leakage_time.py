"""Leakage and time analysis, E-31 to E-34 (DOC-02 §11, §14.3, §13.8; development set only).

The sale-outcome price context in E-31 is descriptive: ``SaleType`` and ``SaleCondition``
are never model features (ADR-02).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from house_price.eda.context import EDAContext, log_price
from house_price.eda.outputs import Outputs, save_figure, save_table

# E-31 leakage review: the sale-information columns and identifiers (DOC-02 §11.4 to §11.6).
LEAKAGE_REVIEW = (
    ("YrSold", "yes, as the valuation year", "kept",
     "Valuation year; needed for ages (ADR-02)"),
    ("MoSold", "yes, as the valuation month", "kept", "Kept as provided (DOC-02 §11.6)"),
    ("SaleType", "no", "excluded",
     "Legal/financial form of the completed transaction, not a property attribute (ADR-02)"),
    ("SaleCondition", "no", "excluded",
     ("Circumstances of the completed sale (for example foreclosure), a transaction outcome "
     "(ADR-02); rows are kept")),
    ("Id", "not a property attribute", "excluded", "Record identifier (IN-03)"),
    ("PID", "not a property attribute", "excluded", "Parcel identifier (IN-03)"),
)  # fmt: skip


def e31_leakage_review(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E-31: decision per sale-information column, and descriptive price context."""
    review = pd.DataFrame(LEAKAGE_REVIEW,
                          columns=["column", "known_at_listing", "decision", "reason"])  # fmt: skip
    y = log_price(dev, target)
    overall = float(y.median())
    frames = []
    for column in ("SaleType", "SaleCondition"):
        stats = y.groupby(dev[column]).agg(n="size", median_log_price="median").reset_index()
        stats.columns = ["level", "n", "median_log_price"]
        stats.insert(0, "column", column)
        frames.append(stats)
    context = pd.concat(frames, ignore_index=True)
    context["difference_vs_overall_median"] = context["median_log_price"] - overall
    return review, context.sort_values(["column", "median_log_price"], ignore_index=True)


def e32_sale_years(dev: pd.DataFrame) -> tuple[pd.DataFrame, Figure]:
    """E-32: sales per year and the months each year covers."""
    table = (dev.groupby("YrSold", sort=True)["MoSold"]
             .agg(n="size", first_month="min", last_month="max").reset_index())  # fmt: skip
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table["YrSold"].astype(str), table["n"], color="tab:blue")
    for pos, (n, last) in enumerate(zip(table["n"], table["last_month"], strict=True)):
        ax.text(pos, n, f"{n} (months 1-{last})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("sales")
    ax.set_title("E-32 Sales per year (development set)")
    figure.tight_layout()
    return table, figure


def e33_price_by_year(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, Figure]:
    """E-33: median log price per year of sale."""
    y = log_price(dev, target)
    table = y.groupby(dev["YrSold"]).agg(n="size", median_log_price="median").reset_index()
    figure, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["YrSold"], table["median_log_price"], marker="o", color="tab:blue")
    ax.set_xticks(table["YrSold"])
    ax.set_ylabel("median log1p(SalePrice)")
    ax.set_title("E-33 Median log price by year of sale (development set)")
    figure.tight_layout()
    return table, figure


def e34_seasonality(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, Figure]:
    """E-34: sales count and median log price by month."""
    y = log_price(dev, target)
    table = y.groupby(dev["MoSold"]).agg(n="size", median_log_price="median").reset_index()
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    left.bar(table["MoSold"], table["n"], color="tab:blue")
    left.set_title("Sales per month")
    right.plot(table["MoSold"], table["median_log_price"], marker="o", color="tab:blue")
    right.set_title("Median log1p(SalePrice) per month")
    for ax in (left, right):
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("MoSold")
    figure.suptitle("E-34 Seasonality (development set)")
    figure.tight_layout()
    return table, figure


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-31 to E-34."""
    out = Outputs()
    dev, target = ctx.dev, ctx.target
    review, context = e31_leakage_review(dev, target)
    save_table(out, ctx.tables_dir, "E-31_leakage_review", review)
    save_table(out, ctx.tables_dir, "E-31_sale_outcome_context", context)
    for name, (table, figure) in {
        "E-32_sale_years": e32_sale_years(dev),
        "E-33_price_by_year": e33_price_by_year(dev, target),
        "E-34_seasonality": e34_seasonality(dev, target),
    }.items():
        save_table(out, ctx.tables_dir, name, table)
        save_figure(out, ctx.figures_dir, name, figure)
    return out

"""Categorical feature analysis, E-19 to E-23 (DOC-02 §8, §13.5; development set only).

Category importance (E-21) is summarised by the correlation ratio eta²: the share of
log-price variance explained by the categories. It is descriptive only; no model is fitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from house_price.eda.context import (
    ORDINAL_LEVELS,
    ORDINAL_SCALES,
    EDAContext,
    categorical_features,
    log_price,
)
from house_price.eda.outputs import Outputs, save_figure, save_table

RARE_SHARE = 0.01  # E-19/E-20: categories under about 1% of rows
MIN_LEVEL_ROWS = 10  # E-23: levels with fewer rows are shown but not used for monotonicity
PANELS_PER_FIGURE = 15  # E-21: box plots per figure file
ABSENT_LEVEL = "None"


def _levels(dev: pd.DataFrame, name: str) -> pd.Series:
    """Category labels as text, with missing (feature absent) shown as ``None``."""
    return dev[name].astype("object").where(dev[name].notna(), ABSENT_LEVEL).astype(str)


def e19_categorical_profile(
    dev: pd.DataFrame, features: list[str], roles: Mapping[str, str]
) -> pd.DataFrame:
    """E-19: cardinality, top category and its share, number of categories under 1%."""
    rows = []
    for name in features:
        shares = _levels(dev, name).value_counts(normalize=True)
        rows.append({
            "feature": name, "role": roles[name], "n_levels": int(shares.size),
            "top_level": str(shares.index[0]), "top_share": float(shares.iloc[0]),
            "n_levels_under_1pct": int((shares < RARE_SHARE).sum()),
        })  # fmt: skip
    return pd.DataFrame(rows).sort_values(["n_levels", "feature"], ascending=[False, True],
                                          ignore_index=True)  # fmt: skip


def e19_foundation_year_built(dev: pd.DataFrame) -> pd.DataFrame:
    """E-19 (context): median construction year per ``Foundation`` type (DOC-02 §3.9)."""
    return (dev.groupby("Foundation", sort=True)["YearBuilt"]
            .agg(n="size", median_year_built="median").reset_index()
            .sort_values("median_year_built", ascending=False, ignore_index=True))  # fmt: skip


def e20_rare_categories(dev: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """E-20: every category with fewer than about 1% of development rows."""
    rows = []
    for name in features:
        counts = _levels(dev, name).value_counts()
        for level, count in counts.items():
            if count / len(dev) < RARE_SHARE:
                rows.append({"feature": name, "level": level, "count": int(count),
                             "share_pct": 100 * count / len(dev)})  # fmt: skip
    table = pd.DataFrame(rows, columns=["feature", "level", "count", "share_pct"])
    return table.sort_values(["feature", "count", "level"], ignore_index=True)


def _eta_squared(groups: pd.Series, values: pd.Series) -> float:
    total = float(((values - values.mean()) ** 2).sum())
    means = values.groupby(groups).transform("mean")
    between = float(((means - values.mean()) ** 2).sum())
    return between / total if total else 0.0


def e21_price_by_category(
    dev: pd.DataFrame, features: list[str], target: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[Figure]]:
    """E-21: log price per level, feature importance (eta²), and box-plot figures."""
    y = log_price(dev, target)
    long_rows, summary_rows = [], []
    for name in features:
        levels = _levels(dev, name)
        stats = y.groupby(levels).agg(n="size", median_log_price="median").reset_index()
        stats.columns = ["level", "n", "median_log_price"]
        stats.insert(0, "feature", name)
        long_rows.append(stats)
        summary_rows.append({
            "feature": name, "n_levels": int(stats.shape[0]),
            "eta_squared": _eta_squared(levels, y),
            "median_range": float(stats["median_log_price"].max()
                                  - stats["median_log_price"].min()),
        })  # fmt: skip
    detail = pd.concat(long_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["eta_squared", "feature"],
                                                     ascending=[False, True], ignore_index=True)  # fmt: skip
    summary.insert(0, "rank", range(1, len(summary) + 1))
    ordered = summary["feature"].tolist()
    figures = []
    for start in range(0, len(ordered), PANELS_PER_FIGURE):
        chunk = ordered[start : start + PANELS_PER_FIGURE]
        nrows = math.ceil(len(chunk) / 3)
        figure, axes = plt.subplots(nrows, 3, figsize=(16, 3.4 * nrows))
        flat = np.atleast_1d(axes).ravel()
        for ax, name in zip(flat, chunk, strict=False):
            levels = _levels(dev, name)
            order = y.groupby(levels).median().sort_values().index.tolist()
            sns.boxplot(x=levels, y=y, order=order, ax=ax, color="tab:blue", fliersize=1)
            eta = summary.loc[summary["feature"] == name, "eta_squared"].iloc[0]
            ax.set_title(f"{name} (eta² {eta:.2f})", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("log1p(SalePrice)")
            ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        for ax in flat[len(chunk) :]:
            ax.set_visible(False)
        figure.suptitle(f"E-21 log1p(SalePrice) by category, ranked by eta² "
                        f"(features {start + 1}-{start + len(chunk)}, development set)")  # fmt: skip
        figure.tight_layout()
        figures.append(figure)
    return summary, detail, figures


def e22_neighborhoods(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, Figure]:
    """E-22: row count and median log price per neighborhood."""
    y = log_price(dev, target)
    table = (y.groupby(dev["Neighborhood"]).agg(n="size", median_log_price="median")
             .reset_index().sort_values(["median_log_price", "Neighborhood"], ignore_index=True))  # fmt: skip
    figure, ax = plt.subplots(figsize=(9, 9))
    ax.barh(table["Neighborhood"], table["median_log_price"], color="tab:blue")
    for pos, (value, n) in enumerate(zip(table["median_log_price"], table["n"], strict=True)):
        ax.text(value + 0.01, pos, f"n={n}", va="center", fontsize=7)
    ax.set_xlim(table["median_log_price"].min() - 0.3, table["median_log_price"].max() + 0.3)
    ax.set_xlabel("median log1p(SalePrice)")
    ax.set_title("E-22 Median log price by neighborhood (development set)")
    figure.tight_layout()
    return table, figure


def e23_ordinal_monotonicity(
    dev: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.DataFrame, Figure]:
    """E-23: median log price per Po-Ex level (None first) and a monotonicity note."""
    y = log_price(dev, target)
    detail_rows, summary_rows = [], []
    figure, axes = plt.subplots(2, 5, figsize=(18, 6.5), sharey=True)
    for ax, scale in zip(axes.ravel(), ORDINAL_SCALES, strict=True):
        levels = _levels(dev, scale)
        stats = y.groupby(levels).agg(n="size", median_log_price="median")
        stats = stats.reindex([lv for lv in ORDINAL_LEVELS if lv in stats.index])
        for level, rec in stats.iterrows():
            detail_rows.append({"scale": scale, "level": level, "n": int(rec["n"]),
                                "median_log_price": float(rec["median_log_price"])})  # fmt: skip
        used = stats[stats["n"] >= MIN_LEVEL_ROWS]["median_log_price"]
        assessable = len(used) > 1
        monotonic = bool(used.is_monotonic_increasing) if assessable else False
        spread = float(used.max() - used.min()) if len(used) else 0.0
        if not assessable:
            note = f"not assessable: fewer than two levels with at least {MIN_LEVEL_ROWS} rows"
        elif monotonic:
            note = f"median log price rises through every level with at least {MIN_LEVEL_ROWS} rows"
        else:
            note = f"not monotonic across levels with at least {MIN_LEVEL_ROWS} rows"
        summary_rows.append({
            "scale": scale, "levels_used": "|".join(used.index), "assessable": assessable,
            "monotonic": monotonic, "median_spread": spread, "note": note,
        })  # fmt: skip
        ax.plot(range(len(stats)), stats["median_log_price"], marker="o", color="tab:blue")
        ax.set_xticks(range(len(stats)), [f"{lv}\n(n={int(n)})" for lv, n
                                          in zip(stats.index, stats["n"], strict=True)],
                      fontsize=7)  # fmt: skip
        label = "monotonic" if monotonic else "not monotonic" if assessable else "not assessable"
        ax.set_title(f"{scale} ({label})", fontsize=10)
    figure.suptitle("E-23 Median log price per quality/condition level (development set)")
    figure.tight_layout()
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows), figure


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-19 to E-23."""
    out = Outputs()
    dev, schema = ctx.dev, ctx.schema
    profile_features = categorical_features(schema, include_excluded=True)
    roles = {spec.name: spec.role for spec in schema.columns}
    save_table(out, ctx.tables_dir, "E-19_categorical_profile",
               e19_categorical_profile(dev, profile_features, roles))  # fmt: skip
    save_table(out, ctx.tables_dir, "E-19_foundation_year_built", e19_foundation_year_built(dev))
    save_table(
        out, ctx.tables_dir, "E-20_rare_categories", e20_rare_categories(dev, profile_features)
    )
    summary, detail, figures = e21_price_by_category(dev, categorical_features(schema), ctx.target)
    save_table(out, ctx.tables_dir, "E-21_category_importance", summary)
    save_table(out, ctx.tables_dir, "E-21_price_by_category", detail)
    for index, figure in enumerate(figures, start=1):
        save_figure(out, ctx.figures_dir, f"E-21_price_by_category_{index}", figure)
    table, figure = e22_neighborhoods(dev, ctx.target)
    save_table(out, ctx.tables_dir, "E-22_neighborhoods", table)
    save_figure(out, ctx.figures_dir, "E-22_neighborhoods", figure)
    mono, levels, figure = e23_ordinal_monotonicity(dev, ctx.target)
    save_table(out, ctx.tables_dir, "E-23_ordinal_monotonicity", mono)
    save_table(out, ctx.tables_dir, "E-23_ordinal_levels", levels)
    save_figure(out, ctx.figures_dir, "E-23_ordinal_monotonicity", figure)
    return out

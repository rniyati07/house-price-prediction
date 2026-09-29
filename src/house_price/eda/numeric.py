"""Numerical feature analysis, E-13 to E-18 (DOC-02 §7, §13.4; development set only)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from house_price.data.profile import quality_flags
from house_price.eda.context import EDAContext, log_price, numeric_features
from house_price.eda.outputs import Outputs, save_figure, save_table

TOP_HEATMAP = 15  # E-16: "the top 15 predictors"
PAIR_REPORT_THRESHOLD = 0.5  # E-16 pairs table lists |r| >= 0.5 ...
STRONG_PAIR_THRESHOLD = 0.7  # ... and marks |r| >= 0.7 as strongly correlated
TOP_SCATTER = 8  # E-17 scatter panels
EXTREME_QUANTILE = 0.99  # E-18: "beyond the 99th percentile"
# E-18 key features: the size, lot, and area columns that drive price or leverage.
EXTREME_FEATURES = ("GrLivArea", "LotArea", "LotFrontage", "TotalBsmtSF", "1stFlrSF",
                    "BsmtFinSF1", "GarageArea", "MasVnrArea", "WoodDeckSF", "OpenPorchSF")  # fmt: skip


def e13_numeric_profile(dev: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """E-13: summary statistics, skewness, share of zeros, distinct values, top share."""
    rows = []
    for name in features:
        values = dev[name].dropna().astype("float64")
        top_share = float(values.value_counts(normalize=True).iloc[0]) if len(values) else 0.0
        rows.append({
            "feature": name, "n": int(values.size), "n_missing": int(dev[name].isna().sum()),
            "min": float(values.min()), "q1": float(values.quantile(0.25)),
            "median": float(values.median()), "mean": float(values.mean()),
            "q3": float(values.quantile(0.75)), "max": float(values.max()),
            "std": float(values.std()), "skewness": float(values.skew()),
            "share_zero": float((values == 0).mean()), "n_distinct": int(values.nunique()),
            "top_value_share": top_share,
        })  # fmt: skip
    return pd.DataFrame(rows)


def e14_distribution_grid(dev: pd.DataFrame, features: list[str]) -> Figure:
    """E-14: histograms of every numeric feature."""
    ncols = 6
    nrows = math.ceil(len(features) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, name in zip(flat, features, strict=False):
        values = dev[name].dropna()
        ax.hist(values, bins=30, color="tab:blue")
        ax.set_title(f"{name} (skew {values.skew():.1f})", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in flat[len(features) :]:
        ax.set_visible(False)
    figure.suptitle("E-14 Numeric feature distributions (development set)")
    figure.tight_layout()
    return figure


def e15_target_correlation(dev: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """E-15: Pearson and Spearman correlation with log price, ranked by |Spearman|."""
    y = log_price(dev, target)
    rows = [
        {"feature": name,
         "pearson": float(dev[name].corr(y, method="pearson")),
         "spearman": float(dev[name].corr(y, method="spearman"))}
        for name in features
    ]  # fmt: skip
    table = pd.DataFrame(rows)
    table["abs_spearman"] = table["spearman"].abs()
    table = table.sort_values(["abs_spearman", "feature"], ascending=[False, True],
                              ignore_index=True)  # fmt: skip
    table.insert(0, "rank", range(1, len(table) + 1))
    return table.drop(columns="abs_spearman")


def e16_correlations(
    dev: pd.DataFrame, features: list[str], ranking: pd.DataFrame
) -> tuple[pd.DataFrame, Figure]:
    """E-16: correlated feature pairs (Pearson) and a heatmap of the top predictors."""
    matrix = dev[features].astype("float64").corr(method="pearson")
    pairs = []
    for i, first in enumerate(features):
        for second in features[i + 1 :]:
            r = matrix.loc[first, second]
            if pd.notna(r) and abs(r) >= PAIR_REPORT_THRESHOLD:
                pairs.append({"feature_a": first, "feature_b": second, "pearson": float(r),
                              "strong": bool(abs(r) >= STRONG_PAIR_THRESHOLD)})  # fmt: skip
    table = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "pearson", "strong"])
    table = table.reindex(table["pearson"].abs().sort_values(ascending=False).index)
    top = ranking["feature"].head(TOP_HEATMAP).tolist()
    figure, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(matrix.loc[top, top], annot=True, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})  # fmt: skip
    ax.set_title(f"E-16 Pearson correlation among the top {len(top)} predictors "
                 "(development set)")  # fmt: skip
    figure.tight_layout()
    return table.reset_index(drop=True), figure


def e17_top_predictors(
    dev: pd.DataFrame, ranking: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, Figure]:
    """E-17: log price against the top predictors, plus the price profile by ``OverallQual``."""
    y = log_price(dev, target)
    top = ranking["feature"].head(TOP_SCATTER).tolist()
    figure, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, name in zip(axes.ravel(), top, strict=False):
        ax.scatter(dev[name], y, s=4, alpha=0.35, color="tab:blue")
        ax.set_title(name)
        ax.set_ylabel("log1p(SalePrice)")
    figure.suptitle("E-17 log1p(SalePrice) against the top predictors (development set)")
    figure.tight_layout()
    quality = (pd.DataFrame({"OverallQual": dev["OverallQual"],
                             "price": dev[target].astype("float64"), "log_price": y})
               .groupby("OverallQual", sort=True)
               .agg(n=("price", "size"), median_price=("price", "median"),
                    median_log_price=("log_price", "median"))
               .reset_index())  # fmt: skip
    return quality, figure


def e18_extremes(dev: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """E-18: values beyond the 99th percentile for key features, with an assessment.

    Every value has already passed the schema's physical ranges (M2). A row is additionally
    cross-checked against the M2 consistency rules; flagged rows are named in the assessment.
    """
    flagged: dict[int, list[str]] = {}
    for flag in quality_flags(dev, id_column):
        if flag.severity == "warning":
            for row_id in flag.ids:
                flagged.setdefault(row_id, []).append(flag.name)
    rows = []
    for name in EXTREME_FEATURES:
        values = dev[name]
        cutoff = float(values.quantile(EXTREME_QUANTILE))
        above = dev.loc[values > cutoff, [id_column, name]].sort_values(name, ascending=False)
        ids = [int(v) for v in above[id_column]]
        issues = sorted({f"{i}: {rule}" for i in ids for rule in flagged.get(i, [])})
        rows.append({
            "feature": name, "p99": cutoff, "n_above_p99": len(ids),
            "max": float(values.max()), "ids_above_p99": "|".join(map(str, ids)),
            "assessment": ("legitimate: within schema physical range; consistent with related "
                           "columns" if not issues else
                           "within schema range; related-column flags: " + "; ".join(issues)),
        })  # fmt: skip
    return pd.DataFrame(rows)


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-13 to E-18."""
    out = Outputs()
    dev, features = ctx.dev, numeric_features(ctx.schema)
    save_table(out, ctx.tables_dir, "E-13_numeric_profile", e13_numeric_profile(dev, features))
    save_figure(out, ctx.figures_dir, "E-14_numeric_distributions",
                e14_distribution_grid(dev, features))  # fmt: skip
    ranking = e15_target_correlation(dev, features, ctx.target)
    save_table(out, ctx.tables_dir, "E-15_target_correlation", ranking)
    pairs, heatmap = e16_correlations(dev, features, ranking)
    save_table(out, ctx.tables_dir, "E-16_correlated_pairs", pairs)
    save_figure(out, ctx.figures_dir, "E-16_correlation_heatmap", heatmap)
    quality, scatter = e17_top_predictors(dev, ranking, ctx.target)
    save_table(out, ctx.tables_dir, "E-17_overallqual_price_profile", quality)
    save_figure(out, ctx.figures_dir, "E-17_top_predictor_scatter", scatter)
    save_table(out, ctx.tables_dir, "E-18_numeric_extremes", e18_extremes(dev, ctx.id_column))
    return out

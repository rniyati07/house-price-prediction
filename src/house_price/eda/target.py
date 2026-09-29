"""Target analysis, E-05 to E-08 (DOC-02 §5, §13.2).

E-05 to E-07 use the development set only (FR-009). E-08 is the permitted split-balance
exception: it reads the per-bin counts recorded in the M2 split manifest and does not load
any held-out row.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

from house_price.data.split import SplitManifest
from house_price.eda.context import EDAContext, log_price
from house_price.eda.outputs import Outputs, save_figure, save_table

# E-07 uses a descriptive least-squares fit on the two documented dominant drivers
# (DOC-02 §2.2). It is an EDA device for reading residual spread, not a project model.
E07_PREDICTORS = ("OverallQual", "GrLivArea")
E07_BINS = 10


def _describe(values: pd.Series, scale: str) -> dict[str, object]:
    return {
        "scale": scale,
        "n": int(values.count()),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "mean": float(values.mean()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "std": float(values.std()),
        "skewness": float(values.skew()),
        "excess_kurtosis": float(values.kurt()),
    }


def e05_target_statistics(dev: pd.DataFrame, target: str) -> pd.DataFrame:
    """E-05: summary statistics of ``SalePrice`` on the raw and ``log1p`` scales."""
    return pd.DataFrame([
        _describe(dev[target].astype("float64"), "dollars"),
        _describe(log_price(dev, target), "log1p"),
    ])  # fmt: skip


def e06_target_figure(dev: pd.DataFrame, target: str) -> Figure:
    """E-06: histogram with KDE, box plot, and normal Q-Q plot on both scales."""
    figure, axes = plt.subplots(3, 2, figsize=(12, 12))
    scales = [("SalePrice (dollars)", dev[target].astype("float64")),
              ("log1p(SalePrice)", log_price(dev, target))]  # fmt: skip
    for col, (label, values) in enumerate(scales):
        sns.histplot(values, bins=50, kde=True, ax=axes[0, col], color="tab:blue")
        axes[0, col].set_title(f"{label}: skew {values.skew():.2f}, "
                               f"excess kurtosis {values.kurt():.2f}")  # fmt: skip
        sns.boxplot(x=values, ax=axes[1, col], color="tab:blue")
        axes[1, col].set_title(f"{label}: box plot")
        (osm, osr), (slope, intercept, _) = stats.probplot(values, dist="norm")
        axes[2, col].scatter(osm, osr, s=6, color="tab:blue")
        axes[2, col].plot(osm, slope * osm + intercept, color="tab:red", linewidth=1)
        axes[2, col].set_title(f"{label}: normal Q-Q")
        axes[2, col].set_xlabel("theoretical quantiles")
        axes[2, col].set_ylabel("ordered values")
    figure.suptitle("E-06 Target distribution (development set)")
    figure.tight_layout()
    return figure


def _residual_spread(fitted: np.ndarray, residuals: np.ndarray, scale: str) -> pd.DataFrame:
    bins = pd.qcut(fitted, q=E07_BINS, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "fitted": fitted, "residual": residuals})
    grouped = frame.groupby("bin", sort=True)
    table = grouped.agg(n=("residual", "size"), fitted_mean=("fitted", "mean"),
                        residual_sd=("residual", "std")).reset_index()  # fmt: skip
    table.insert(0, "scale", scale)
    return table


def _least_squares(dev: pd.DataFrame, response: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        [np.ones(len(dev)), *(dev[c].astype("float64") for c in E07_PREDICTORS)]
    )
    coef, *_ = np.linalg.lstsq(design, response.to_numpy(dtype="float64"), rcond=None)
    fitted = design @ coef
    return fitted, response.to_numpy(dtype="float64") - fitted


def e07_heteroscedasticity(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, Figure]:
    """E-07: residual spread versus fitted value on the raw and log scales."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    tables = []
    scales = [("dollars", dev[target].astype("float64")), ("log1p", log_price(dev, target))]
    for ax, (scale, response) in zip(axes, scales, strict=True):
        fitted, residuals = _least_squares(dev, response)
        spread = _residual_spread(fitted, residuals, scale)
        tables.append(spread)
        ax.scatter(fitted, residuals, s=5, alpha=0.4, color="tab:blue")
        ax.plot(spread["fitted_mean"], 2 * spread["residual_sd"], color="tab:red",
                label="±2 SD per fitted-value decile")  # fmt: skip
        ax.plot(spread["fitted_mean"], -2 * spread["residual_sd"], color="tab:red")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{scale}: residual vs fitted (least squares on "
                     f"{' + '.join(E07_PREDICTORS)})")  # fmt: skip
        ax.set_xlabel("fitted value")
        ax.set_ylabel("residual")
        ax.legend(loc="upper left")
    figure.suptitle("E-07 Heteroscedasticity (development set)")
    figure.tight_layout()
    table = pd.concat(tables, ignore_index=True)
    ratios = table.groupby("scale")["residual_sd"].agg(lambda s: s.iloc[-1] / s.iloc[0])
    table["top_to_bottom_sd_ratio"] = table["scale"].map(ratios)
    return table, figure


def e08_split_balance(manifest: SplitManifest, tolerance_pp: float) -> pd.DataFrame:
    """E-08: share of each log-price bin in the two sets, from the split manifest."""
    table = pd.DataFrame([row.model_dump() for row in manifest.balance])
    table["within_tolerance"] = table["diff_pp"] <= tolerance_pp
    return table


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-05 to E-08."""
    out = Outputs()
    dev, target = ctx.dev, ctx.target
    save_table(out, ctx.tables_dir, "E-05_target_statistics", e05_target_statistics(dev, target))
    save_figure(out, ctx.figures_dir, "E-06_target_distribution", e06_target_figure(dev, target))
    spread, figure = e07_heteroscedasticity(dev, target)
    save_table(out, ctx.tables_dir, "E-07_residual_spread", spread)
    save_figure(out, ctx.figures_dir, "E-07_heteroscedasticity", figure)
    tolerance = ctx.config.validation.split_balance_tolerance_pp
    save_table(
        out, ctx.tables_dir, "E-08_split_balance", e08_split_balance(ctx.manifest, tolerance)
    )
    return out

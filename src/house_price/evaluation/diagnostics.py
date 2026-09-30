"""Out-of-fold diagnostics for the selected candidate (FR-032, AC-037, DOC-03 §11.6, DN-09).

Evidence only: this module computes the three gate views and writes them to
``reports/selection/``; it never decides pass or fail (IN-10: ADR-11 describes the gates
qualitatively, so the judgement is a written human review, DN-19).

* **OOF prediction per row (DN-09):** the mean of the row's OOF log predictions over the
  repeats; converted to dollars only for dollar-scale views.
* **Gate (a) residuals:** signed log error versus predicted log price, with a smoothed trend
  (means over quantile bins of the prediction).
* **Gate (b) price deciles:** mean signed log error per decile of actual log price (the
  DN-01 development-set deciles), with a 95% confidence band.
* **Gate (c) neighborhoods:** log-RMSE per neighborhood with row counts, sorted, with the
  overall log-RMSE as the reference.

Signed log error is ``log1p(predicted) - log1p(actual)``: positive means over-prediction.
Only development rows are used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TREND_BINS = 20
FIGURE_DPI = 100
Z_95 = 1.96
FLOAT_FORMAT = "%.10g"

RESIDUAL_PLOT = "residual_vs_predicted.png"
DECILE_PLOT = "price_decile_bias.png"
NEIGHBORHOOD_PLOT = "neighborhood_log_rmse.png"
OOF_TABLE = "oof_predictions.csv"
TREND_TABLE = "residual_trend.csv"
DECILE_TABLE = "price_decile_bias.csv"
NEIGHBORHOOD_TABLE = "neighborhood_log_rmse.csv"
SUMMARY = "diagnostic_summary.json"


class DiagnosticsError(ValueError):
    """The OOF predictions do not match the development set."""


def aggregate_oof(oof: pd.DataFrame, id_column: str = "Id") -> pd.DataFrame:
    """DN-09: one prediction per row, the mean OOF log prediction over the repeats."""
    grouped = oof.groupby(id_column, sort=True)["log_prediction"]
    frame = grouped.agg(["mean", "size"]).reset_index()
    return frame.rename(columns={"mean": "log_prediction", "size": "n_repeats"})


def diagnostic_frame(
    oof: pd.DataFrame, dev: pd.DataFrame, *, id_column: str, target: str, group: str
) -> pd.DataFrame:
    """Per development row: actual and averaged predicted log price, signed log error."""
    predicted = aggregate_oof(oof, id_column)
    if set(predicted[id_column]) != set(dev[id_column]):
        raise DiagnosticsError("OOF predictions do not cover exactly the development rows")
    base = dev[[id_column, target, group]].copy()
    frame = base.merge(predicted, on=id_column, validate="one_to_one")
    frame["log_actual"] = np.log1p(frame[target].astype("float64"))
    frame["signed_error"] = frame["log_prediction"] - frame["log_actual"]
    frame["predicted_dollars"] = np.expm1(frame["log_prediction"])
    return frame.sort_values(id_column, ignore_index=True)


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors))))


def residual_trend(frame: pd.DataFrame, n_bins: int = TREND_BINS) -> pd.DataFrame:
    """Gate (a) trend: mean signed error within quantile bins of the prediction."""
    bins = pd.qcut(frame["log_prediction"], q=n_bins, labels=False, duplicates="drop")
    table = frame.groupby(bins).agg(
        n=("signed_error", "size"),
        predicted_median=("log_prediction", "median"),
        mean_signed_error=("signed_error", "mean"),
        sd_signed_error=("signed_error", "std"),
    )
    return table.reset_index(names="bin")


def decile_bias(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Gate (b): mean signed error per decile of actual log price, with a 95% band."""
    deciles = pd.qcut(frame["log_actual"], q=n_bins, labels=False)
    rows = []
    for decile, part in frame.groupby(deciles, sort=True):
        errors = part["signed_error"]
        se = float(errors.std(ddof=1) / math.sqrt(len(errors)))
        mean = float(errors.mean())
        rows.append(
            {
                "decile": int(decile),
                "n": len(errors),
                "actual_log_min": float(part["log_actual"].min()),
                "actual_log_max": float(part["log_actual"].max()),
                "mean_signed_error": mean,
                "se": se,
                "ci_low": mean - Z_95 * se,
                "ci_high": mean + Z_95 * se,
                "log_rmse": _rmse(errors),
            }
        )
    return pd.DataFrame(rows)


def neighborhood_error(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Gate (c): log-RMSE and mean signed error per neighborhood, largest error first."""
    rows = [
        {
            group: str(name),
            "n": len(part),
            "log_rmse": _rmse(part["signed_error"]),
            "mean_signed_error": float(part["signed_error"].mean()),
        }
        for name, part in frame.groupby(group, sort=True)
    ]
    table = pd.DataFrame(rows)
    return table.sort_values(["log_rmse", group], ascending=[False, True], ignore_index=True)


@dataclass(frozen=True)
class DiagnosticReport:
    """The gate evidence and where it was written."""

    candidate: str
    overall_log_rmse: float
    frame: pd.DataFrame
    trend: pd.DataFrame
    deciles: pd.DataFrame
    neighborhoods: pd.DataFrame
    paths: dict[str, Path]

    def summary(self) -> dict[str, Any]:
        """Facts a reviewer reads first (no pass/fail: IN-10)."""
        d, n, t = self.deciles, self.neighborhoods, self.trend
        worst = n.iloc[0]
        return {
            "candidate": self.candidate,
            "n_rows": len(self.frame),
            "repeats_per_row": sorted({int(v) for v in self.frame["n_repeats"]}),
            "overall_log_rmse": self.overall_log_rmse,
            "overall_mean_signed_error": float(self.frame["signed_error"].mean()),
            "residual_trend_mean_signed_error_range": [
                float(t["mean_signed_error"].min()),
                float(t["mean_signed_error"].max()),
            ],
            "decile_mean_signed_error": {
                int(r.decile): float(r.mean_signed_error) for r in d.itertuples()
            },
            "deciles_with_ci_excluding_zero": [
                int(r.decile) for r in d.itertuples() if r.ci_low > 0 or r.ci_high < 0
            ],
            "worst_neighborhood": {
                "name": str(worst.iloc[0]),
                "n": int(worst["n"]),
                "log_rmse": float(worst["log_rmse"]),
            },
            "neighborhoods_above_1_5x_overall": [
                {"name": str(r.iloc[0]), "n": int(r["n"]), "log_rmse": float(r["log_rmse"])}
                for _, r in n.iterrows()
                if r["log_rmse"] > 1.5 * self.overall_log_rmse
            ],
            "files": {k: v.name for k, v in self.paths.items()},
        }


def _save_table(table: pd.DataFrame, path: Path) -> Path:
    table.to_csv(path, index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    return path


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": None})
    plt.close(figure)
    return path


def _plot_residuals(frame: pd.DataFrame, trend: pd.DataFrame, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        frame["log_prediction"],
        frame["signed_error"],
        s=6,
        alpha=0.35,
        label="development rows (OOF)",
    )
    axis.plot(
        trend["predicted_median"],
        trend["mean_signed_error"],
        color="tab:red",
        marker="o",
        label=f"binned mean ({len(trend)} quantile bins)",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("predicted log1p(SalePrice) (mean OOF over repeats)")
    axis.set_ylabel("signed log error (predicted - actual)")
    axis.set_title(f"Gate (a) residuals vs predicted: {title}")
    axis.legend()
    return figure


def _plot_deciles(deciles: pd.DataFrame, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 5))
    x = deciles["decile"]
    axis.fill_between(x, deciles["ci_low"], deciles["ci_high"], alpha=0.25, label="95% CI")
    axis.plot(x, deciles["mean_signed_error"], marker="o", label="mean signed log error")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(list(x))
    axis.set_xlabel("decile of actual log price (0 = cheapest; development set, DN-01)")
    axis.set_ylabel("mean signed log error (predicted - actual)")
    axis.set_title(f"Gate (b) bias by price decile: {title}")
    axis.legend()
    return figure


def _plot_neighborhoods(table: pd.DataFrame, group: str, overall: float, title: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, max(5, 0.28 * len(table))))
    ordered = table.iloc[::-1]
    labels = [f"{name} (n={n})" for name, n in zip(ordered[group], ordered["n"], strict=True)]
    axis.barh(labels, ordered["log_rmse"])
    axis.axvline(overall, color="tab:red", linestyle="--", label=f"overall {overall:.4f}")
    axis.set_xlabel("log-RMSE (OOF, development set)")
    axis.set_title(f"Gate (c) error by neighborhood: {title}")
    axis.legend()
    return figure


def write_report(
    candidate: str,
    oof: pd.DataFrame,
    dev: pd.DataFrame,
    out_dir: Path,
    *,
    id_column: str = "Id",
    target: str = "SalePrice",
    group: str = "Neighborhood",
) -> DiagnosticReport:
    """Compute the three gate views for ``candidate`` and write plots and tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = diagnostic_frame(oof, dev, id_column=id_column, target=target, group=group)
    overall = _rmse(frame["signed_error"])
    trend, deciles = residual_trend(frame), decile_bias(frame)
    neighborhoods = neighborhood_error(frame, group)
    paths = {
        "oof_predictions": _save_table(
            frame[[id_column, "log_actual", "log_prediction", "n_repeats", "signed_error", group]],
            out_dir / OOF_TABLE,
        ),
        "residual_trend": _save_table(trend, out_dir / TREND_TABLE),
        "price_decile_bias": _save_table(deciles, out_dir / DECILE_TABLE),
        "neighborhood_log_rmse": _save_table(neighborhoods, out_dir / NEIGHBORHOOD_TABLE),
        "residual_plot": _save_figure(
            _plot_residuals(frame, trend, candidate), out_dir / RESIDUAL_PLOT
        ),
        "decile_plot": _save_figure(_plot_deciles(deciles, candidate), out_dir / DECILE_PLOT),
        "neighborhood_plot": _save_figure(
            _plot_neighborhoods(neighborhoods, group, overall, candidate),
            out_dir / NEIGHBORHOOD_PLOT,
        ),
    }
    return DiagnosticReport(candidate, overall, frame, trend, deciles, neighborhoods, paths)

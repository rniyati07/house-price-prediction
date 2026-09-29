"""Engineered feature assessment, E-27 to E-29 (DOC-02 §10, §13.7; development set only).

The features are produced by the project's own transformers (``SemanticNAFiller`` then
``FeatureEngineer``), applied to the development set's model-input columns, so the EDA
examines exactly what the pipeline will compute.

Each DOC-02 §10 "expected relationship" is a hypothesis. E-27 records it with an explicit
assessment rule and whether the development-set evidence supports it. This is evidence for
Q17 only: retention is decided in M7 by the cross-validation ablation (E-30, IN-04), never
by these rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from house_price.eda.context import EDAContext, log_price
from house_price.eda.outputs import Outputs, save_figure, save_table
from house_price.features.engineer import FeatureEngineer
from house_price.features.semantic import SemanticNAFiller

CONTINUOUS = ("TotalSF", "TotalBath", "HouseAge", "RemodAge", "TotalPorchSF", "GarageAge")
BINARY = ("IsRemodeled", "HasPool", "HasGarage", "HasBsmt", "HasFireplace", "Has2ndFlr")
TREND_BINS = 10  # E-28: decile bins for the log-price trend of continuous features
MIN_ROWS = 10  # levels/bins with fewer rows are shown but not used by the rules
CLEAR_EFFECT = 0.2  # "clearly positive": median log-price gap of at least 0.2 (about 22%)
WEAK = 0.2  # "weak": |Spearman| below 0.2
REMODEL_FLOOR = 1950  # DOC-02 §10.5

Assessment = Callable[[pd.DataFrame, pd.Series], tuple[str, bool]]


def engineered_frame(ctx: EDAContext) -> pd.DataFrame:
    """The development set's model inputs after the semantic filler and feature engineer."""
    inputs = [spec.name for spec in ctx.schema.with_role("model_input")]
    filled = SemanticNAFiller.from_config(ctx.features).fit_transform(ctx.dev[inputs])
    return FeatureEngineer.from_config(ctx.features).fit_transform(filled)


# ------------------------------------------------------------------- assessment rules


def _spearman(frame: pd.DataFrame, feature: str, y: pd.Series) -> float:
    return float(frame[feature].corr(y, method="spearman"))


def _median_gap(frame: pd.DataFrame, flag: str, y: pd.Series) -> tuple[float, int]:
    medians = y.groupby(frame[flag]).median()
    count = int((frame[flag] == 1).sum())
    if not {0, 1} <= set(medians.index):
        return float("nan"), count
    return float(medians[1] - medians[0]), count


def _total_sf(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    total, living = _spearman(frame, "TotalSF", y), _spearman(frame, "GrLivArea", y)
    return (f"Spearman {total:.3f} (GrLivArea {living:.3f})", total > 0.5 and total >= living)


def _total_bath(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    rho = _spearman(frame, "TotalBath", y)
    steps = y.groupby(frame["TotalBath"]).agg(["size", "median"])
    medians = steps.loc[steps["size"] >= MIN_ROWS, "median"]
    rising = bool(medians.is_monotonic_increasing)
    gains = np.diff(medians.to_numpy())
    diminishing = len(gains) > 2 and gains[-1] < gains.max()
    observed = (
        f"Spearman {rho:.3f}; median log price rises across values {list(medians.index)}: "
        f"{rising}; last step smaller than the largest: {diminishing}"
    )
    return observed, rho > 0 and rising


def _house_age(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    rho = _spearman(frame, "HouseAge", y)
    trend = binned_trend(frame["HouseAge"], y)
    trend = trend[trend["n"] >= MIN_ROWS]
    half = len(trend) // 2
    first = np.polyfit(trend["x_median"][:half], trend["median_log_price"][:half], 1)[0]
    second = np.polyfit(trend["x_median"][half:], trend["median_log_price"][half:], 1)[0]
    observed = (
        f"Spearman {rho:.3f}; slope of binned medians per year: younger half {first:.4f}, "
        f"older half {second:.4f}"
    )
    return observed, rho < 0 and abs(first) > abs(second)


def _negative(feature: str) -> Assessment:
    def rule(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
        rho = _spearman(frame, feature, y)
        return f"Spearman {rho:.3f}", rho < 0

    return rule


def _is_remodeled(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    rho = _spearman(frame, "IsRemodeled", y)
    gap, count = _median_gap(frame, "IsRemodeled", y)
    return (f"Spearman {rho:.3f}; median gap {gap:+.3f} ({count} remodeled)", abs(rho) < WEAK)


def _porch(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    rho = _spearman(frame, "TotalPorchSF", y)
    return f"Spearman {rho:.3f}", 0 < rho < 0.5


def _positive_flag(flag: str, threshold: float) -> Assessment:
    def rule(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
        gap, count = _median_gap(frame, flag, y)
        return f"median log price gap {gap:+.3f} ({count} of {len(frame)} with the feature)", bool(
            gap > threshold
        )

    return rule


def _has_pool(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    gap, count = _median_gap(frame, "HasPool", y)
    share = count / len(frame)
    observed = f"{count} pools ({100 * share:.2f}% of rows); median gap {gap:+.3f}"
    return observed, share < 0.01


def _second_floor(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    bands = pd.qcut(frame["TotalSF"], q=5, labels=False, duplicates="drop")
    gaps = []
    for band in sorted(bands.dropna().unique()):
        inside = bands == band
        counts = frame.loc[inside, "Has2ndFlr"].value_counts()
        if counts.get(0, 0) >= MIN_ROWS and counts.get(1, 0) >= MIN_ROWS:
            gaps.append(_median_gap(frame.loc[inside], "Has2ndFlr", y[inside])[0])
    signs = {np.sign(g) for g in gaps}
    observed = "median gap (2nd floor - none) within TotalSF quintiles: " + ", ".join(
        f"{g:+.3f}" for g in gaps
    )
    return observed, len(signs) > 1


def _garage_age(frame: pd.DataFrame, y: pd.Series) -> tuple[str, bool]:
    garage = frame[(frame["HasGarage"] == 1) & frame["GarageAge"].notna()]
    ranks = garage[["GarageAge", "HouseAge"]].rank()
    target = y[garage.index].rank()
    raw = float(ranks["GarageAge"].corr(target))

    def residual(values: pd.Series) -> np.ndarray:
        design = np.column_stack([np.ones(len(ranks)), ranks["HouseAge"]])
        coef, *_ = np.linalg.lstsq(design, values.to_numpy(), rcond=None)
        return np.asarray(values.to_numpy() - design @ coef)

    partial = float(np.corrcoef(residual(ranks["GarageAge"]), residual(target))[0, 1])
    observed = f"Spearman {raw:.3f} (garage houses); partial given HouseAge {partial:.3f}"
    return observed, raw < 0 and abs(partial) < abs(raw)


@dataclass(frozen=True)
class Hypothesis:
    feature: str
    definition: str
    expected: str
    rule: str
    assess: Assessment


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("TotalSF", "TotalBsmtSF + 1stFlrSF + 2ndFlrSF",
               "Strongly positive, near-linear; at least as strong as GrLivArea (§10.2)",
               "Spearman > 0.5 and >= GrLivArea's", _total_sf),
    Hypothesis("TotalBath", "FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath",
               "Positive and roughly stepwise, diminishing at high counts (§10.3)",
               f"Spearman > 0 and median rises across values with >= {MIN_ROWS} rows",
               _total_bath),
    Hypothesis("HouseAge", "YrSold - YearBuilt",
               "Negative, non-linear: steep drop over the first decades, then flattening (§10.4)",
               "Spearman < 0 and the younger half of the decile trend is steeper", _house_age),
    Hypothesis("RemodAge", "YrSold - YearRemodAdd", "Negative (§10.5)", "Spearman < 0",
               _negative("RemodAge")),
    Hypothesis("IsRemodeled", "1 if YearRemodAdd != YearBuilt else 0",
               "Weak on its own, possibly mixed (§10.6)", f"|Spearman| < {WEAK}", _is_remodeled),
    Hypothesis("TotalPorchSF", "OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch",
               "Mildly positive (§10.7)", "0 < Spearman < 0.5", _porch),
    Hypothesis("HasPool", "1 if PoolArea > 0 else 0",
               "Uncertain: too rare for a reliable estimate (§10.8)",
               "fewer than 1% of rows have a pool", _has_pool),
    Hypothesis("HasGarage", "1 if GarageType != 'None' else 0",
               "Clearly positive: garage-less houses are markedly cheaper (§10.8)",
               f"median log price gap > {CLEAR_EFFECT}", _positive_flag("HasGarage", CLEAR_EFFECT)),
    Hypothesis("HasBsmt", "1 if TotalBsmtSF > 0 else 0", "Positive (§10.8)",
               "median log price gap > 0", _positive_flag("HasBsmt", 0.0)),
    Hypothesis("HasFireplace", "1 if Fireplaces > 0 else 0", "Positive (§10.8)",
               "median log price gap > 0", _positive_flag("HasFireplace", 0.0)),
    Hypothesis("Has2ndFlr", "1 if 2ndFlrSF > 0 else 0",
               "Mixed: not uniformly more expensive than one-story homes of the same total "
               "size (§10.8)",
               f"the gap changes sign across TotalSF quintiles (groups of >= {MIN_ROWS})",
               _second_floor),
    Hypothesis("GarageAge", "YrSold - GarageYrBlt with a garage, else 0",
               "Negative but weak once HouseAge is known (§10.9)",
               "Spearman < 0 among garage houses and |partial given HouseAge| smaller",
               _garage_age),
)  # fmt: skip


# ------------------------------------------------------------------------ deliverables


def binned_trend(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Median log price per quantile bin of ``x`` (rows with ``x`` recorded)."""
    recorded = x.notna()
    bins = pd.qcut(x[recorded].rank(method="first"), q=TREND_BINS, labels=False)
    frame = pd.DataFrame({"bin": bins, "x": x[recorded], "y": y[recorded]})
    return (
        frame.groupby("bin", sort=True)
        .agg(n=("y", "size"), x_median=("x", "median"), median_log_price=("y", "median"))
        .reset_index()
    )


def e27_feature_table(frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """E-27: definition, expected relationship, observed correlation, assessment."""
    rows = []
    for hyp in HYPOTHESES:
        values = frame[hyp.feature]
        try:
            observed, supported = hyp.assess(frame, y)
        except (KeyError, IndexError, ValueError, TypeError, np.linalg.LinAlgError) as exc:
            observed, supported = f"not assessable: {type(exc).__name__}: {exc}", False
        rows.append({
            "feature": hyp.feature, "definition": hyp.definition,
            "expected_relationship": hyp.expected,
            "n": int(values.notna().sum()), "n_missing": int(values.isna().sum()),
            "pearson": float(values.corr(y, method="pearson")),
            "spearman": float(values.corr(y, method="spearman")),
            "assessment_rule": hyp.rule, "observed": observed,
            "hypothesis_supported": bool(supported),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e28_tables(frame: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E-28 tables: decile trends of continuous features; price by level of discrete ones."""
    trends = []
    for feature in CONTINUOUS:
        trend = binned_trend(frame[feature], y)
        trend.insert(0, "feature", feature)
        trends.append(trend)
    discrete = []
    for feature in (*BINARY, "TotalBath"):
        stats = y.groupby(frame[feature]).agg(n="size", median_log_price="median").reset_index()
        stats.columns = ["value", "n", "median_log_price"]
        stats.insert(0, "feature", feature)
        discrete.append(stats)
    return pd.concat(trends, ignore_index=True), pd.concat(discrete, ignore_index=True)


def e28_continuous_figure(frame: pd.DataFrame, y: pd.Series, trends: pd.DataFrame) -> Figure:
    """E-28: log price against each continuous engineered feature, with decile medians."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feature in zip(axes.ravel(), CONTINUOUS, strict=True):
        ax.scatter(frame[feature], y, s=4, alpha=0.3, color="tab:blue")
        trend = trends[trends["feature"] == feature]
        ax.plot(trend["x_median"], trend["median_log_price"], color="tab:red", marker="o",
                label="decile medians")  # fmt: skip
        ax.set_title(f"{feature} (Spearman {frame[feature].corr(y, method='spearman'):.2f})")
        ax.set_ylabel("log1p(SalePrice)")
        ax.legend(loc="lower right", fontsize=8)
    figure.suptitle("E-28 Engineered features against log price (development set)")
    figure.tight_layout()
    return figure


def e28_flag_figure(frame: pd.DataFrame, y: pd.Series) -> Figure:
    """E-28: presence flags as box plots of log price."""
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, feature in zip(axes.ravel(), BINARY, strict=True):
        counts = frame[feature].value_counts()
        sns.boxplot(x=frame[feature], y=y, order=[0, 1], ax=ax, color="tab:blue", fliersize=1)
        ax.set_xticks([0, 1], [f"0 (n={counts.get(0, 0)})", f"1 (n={counts.get(1, 0)})"])
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("log1p(SalePrice)")
    figure.suptitle("E-28 Presence flags against log price (development set)")
    figure.tight_layout()
    return figure


def e29_remodel_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """E-29: rows at the ``YearRemodAdd`` recording floor and the effect on the features."""
    at_floor = frame["YearRemodAdd"] == REMODEL_FLOOR
    floor_affected = at_floor & (frame["YearBuilt"] < REMODEL_FLOOR)
    old = frame["YearBuilt"] < REMODEL_FLOOR
    remodeled = frame["IsRemodeled"] == 1
    remod_age = frame.loc[floor_affected, "RemodAge"]
    gap = (frame.loc[floor_affected, "HouseAge"] - remod_age).median()
    rows = [
        ("development_rows", len(frame), "rows analysed"),
        ("remodel_year_below_floor", int((frame["YearRemodAdd"] < REMODEL_FLOOR).sum()),
         "YearRemodAdd < 1950 (0 confirms the floor)"),
        ("remodel_year_at_floor", int(at_floor.sum()), "YearRemodAdd = 1950"),
        ("floor_rows_built_before_1950", int(floor_affected.sum()),
         "remodel year may be the recording floor, not a real remodel"),
        ("floor_rows_built_in_1950", int((at_floor & (frame["YearBuilt"] == REMODEL_FLOOR)).sum()),
         "genuinely unremodeled 1950 houses"),
        ("houses_built_before_1950", int(old.sum()), ""),
        ("pct_pre1950_houses_at_floor", round(100 * floor_affected.sum() / max(old.sum(), 1), 1),
         "share of pre-1950 houses whose remodel year is 1950"),
        ("floor_rows_flagged_remodeled", int((floor_affected & remodeled).sum()),
         "IsRemodeled = 1 because 1950 != YearBuilt; may not be a real remodel"),
        ("pct_of_remodeled_flags_from_floor",
         round(100 * (floor_affected & remodeled).sum() / max(remodeled.sum(), 1), 1),
         "share of all IsRemodeled = 1 rows that sit at the floor"),
        ("floor_rows_remodage_min", float(remod_age.min()) if len(remod_age) else float("nan"),
         "RemodAge = YrSold - 1950 for every floor row"),
        ("floor_rows_remodage_max", float(remod_age.max()) if len(remod_age) else float("nan"), ""),
        ("floor_rows_median_houseage_minus_remodage", float(gap) if len(remod_age) else float("nan"),
         "years by which RemodAge understates HouseAge at the floor"),
    ]  # fmt: skip
    return pd.DataFrame(rows, columns=["metric", "value", "note"])


def e29_figure(frame: pd.DataFrame) -> Figure:
    """E-29: distribution of ``YearRemodAdd``, with the 1950 floor highlighted."""
    counts = frame["YearRemodAdd"].value_counts().sort_index()
    figure, ax = plt.subplots(figsize=(11, 4))
    colors = ["tab:red" if year == REMODEL_FLOOR else "tab:blue" for year in counts.index]
    ax.bar(counts.index, counts.to_numpy(), color=colors, width=0.8)
    ax.set_xlabel("YearRemodAdd")
    ax.set_ylabel("houses")
    ax.set_title("E-29 YearRemodAdd: the spike at the 1950 recording floor (development set)")
    figure.tight_layout()
    return figure


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-27 to E-29."""
    out = Outputs()
    frame = engineered_frame(ctx)
    y = log_price(ctx.dev, ctx.target)
    save_table(out, ctx.tables_dir, "E-27_engineered_features", e27_feature_table(frame, y))
    trends, discrete = e28_tables(frame, y)
    save_table(out, ctx.tables_dir, "E-28_binned_trends", trends)
    save_table(out, ctx.tables_dir, "E-28_discrete_price_summary", discrete)
    save_figure(out, ctx.figures_dir, "E-28_engineered_continuous",
                e28_continuous_figure(frame, y, trends))  # fmt: skip
    save_figure(out, ctx.figures_dir, "E-28_engineered_flags", e28_flag_figure(frame, y))
    save_table(out, ctx.tables_dir, "E-29_remodel_floor_audit", e29_remodel_audit(frame))
    save_figure(out, ctx.figures_dir, "E-29_yearremodadd_distribution", e29_figure(frame))
    return out

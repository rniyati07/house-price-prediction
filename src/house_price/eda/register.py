"""E-35 confirmation register (DOC-02 §13.9; DOC-05 M3 task 3).

Every documented property stated in DOC-02 is checked against the saved deliverable tables
(never against fresh data), with an explicit rule, and recorded as ``confirmed`` with the
observed value or ``not confirmed`` with the discrepancy.

Two tolerances apply to DOC-02's approximate statistics, because DOC-02 quotes raw-file
values while target statistics are measured on the development set (FR-009): location
statistics must agree within 10%, shape statistics (skewness, kurtosis) within 20%.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

Tables = Mapping[str, pd.DataFrame]
Check = Callable[[Tables], tuple[str, bool]]

LOCATION_TOLERANCE = 0.10
SHAPE_TOLERANCE = 0.20
DOMINANT_SHARE = 0.95  # "almost always" / "dominated by one value"
NEAR_CONSTANT_SHARE = 0.90  # §7.2: one value covers almost every row
ZERO_INFLATED_SHARE = 0.40  # §7.1: many exact zeros
STRONG_CORRELATION = 0.70  # §7.3 / E-16: strongly correlated pair
TOP_CORRELATES_WINDOW = 15  # §7.3 list must fall within the top 15 of E-15
MIN_LEVEL_ROWS = 10


# ------------------------------------------------------------------------ table access


def _row(t: Tables, table: str, key: str, value: object) -> pd.Series:
    frame = t[table]
    return frame.loc[frame[key].astype(str) == str(value)].iloc[0]


def _values(t: Tables, column: str) -> pd.DataFrame:
    counts = t["E-02_value_counts"]
    return counts[(counts["column"] == column) & (counts["value"] != "<missing>")]


def _count(t: Tables, column: str, value: str) -> int:
    levels = _values(t, column)
    hit = levels.loc[levels["value"].astype(str) == value, "count"]
    return int(hit.iloc[0]) if len(hit) else 0


def _share(t: Tables, column: str, value: str) -> float:
    counts = t["E-02_value_counts"]
    total = counts.loc[counts["column"] == column, "count"].sum()
    return _count(t, column, value) / total


def _integrity(t: Tables, check: str) -> str:
    return str(_row(t, "E-03_key_integrity", "check", check)["result"])


def _stat(t: Tables, scale: str, name: str) -> float:
    return float(_row(t, "E-05_target_statistics", "scale", scale)[name])


def _num(t: Tables, feature: str, name: str) -> float:
    return float(_row(t, "E-13_numeric_profile", "feature", feature)[name])


def _miss(t: Tables, column: str, name: str) -> float:
    return float(_row(t, "E-09_missing_values", "column", column)[name])


def _ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    return [int(v) for v in str(value).split("|")]


def _flag(t: Tables, rule: str) -> tuple[str, list[int]]:
    row = _row(t, "E-11_consistency_checks", "name", rule)
    return str(row["status"]), _ids(row["ids"])


def _pair(t: Tables, a: str, b: str) -> float | None:
    pairs = t["E-16_correlated_pairs"]
    forward = (pairs["feature_a"] == a) & (pairs["feature_b"] == b)
    backward = (pairs["feature_a"] == b) & (pairs["feature_b"] == a)
    hit = pairs[forward | backward]
    return float(hit["pearson"].iloc[0]) if len(hit) else None


def _spearman(t: Tables, feature: str) -> float:
    return abs(float(_row(t, "E-15_target_correlation", "feature", feature)["spearman"]))


def _rank(t: Tables, feature: str) -> int:
    return int(_row(t, "E-15_target_correlation", "feature", feature)["rank"])


def _within(observed: float, documented: float, tolerance: float) -> bool:
    return abs(observed - documented) <= tolerance * abs(documented)


def _category_median(t: Tables, feature: str, level: str) -> float:
    frame = t["E-21_price_by_category"]
    hit = frame[(frame["feature"] == feature) & (frame["level"].astype(str) == level)]
    return float(hit["median_log_price"].iloc[0])


# ------------------------------------------------------------------ check factories


def _approx(scale: str, name: str, documented: float, tolerance: float) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        observed = _stat(t, scale, name)
        return f"{observed:,.2f} (development set)", _within(observed, documented, tolerance)

    return check


def _strong_pair(a: str, b: str) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        r = _pair(t, a, b)
        if r is None:
            return "|r| < 0.5", False
        return f"Pearson r = {r:.3f}", abs(r) >= STRONG_CORRELATION

    return check


def _dominant(column: str, value: str) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        share = _share(t, column, value)
        return f"{value} share {100 * share:.1f}% (raw file)", share >= DOMINANT_SHARE

    return check


def _flag_clear(rule: str) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        status, _ = _flag(t, rule)
        return f"rule {rule}: {status}", status == "clear"

    return check


def _all_numeric(columns: tuple[str, ...], stat: str, threshold: float, percent: bool) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        values = {c: _num(t, c, stat) for c in columns}
        shown = ", ".join(
            f"{c} {100 * v:.1f}%" if percent else f"{c} {v:.1f}" for c, v in values.items()
        )
        passes = all(v >= threshold if percent else v > threshold for v in values.values())
        return shown, passes

    return check


def _ordinal(scales: tuple[str, ...], expect_monotonic: bool) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        mono = t["E-23_ordinal_monotonicity"].set_index("scale")
        quality_spread = mono.loc[["ExterQual", "KitchenQual", "BsmtQual"], "median_spread"].min()
        parts, ok = [], True
        for scale in scales:
            row = mono.loc[scale]
            monotonic = bool(row["monotonic"])
            label = "monotonic" if monotonic else "not monotonic"
            parts.append(f"{scale}: {label}, spread {row['median_spread']:.2f}")
            if expect_monotonic:
                ok &= monotonic
            else:
                ok &= (not monotonic) or row["median_spread"] < quality_spread
        return "; ".join(parts), ok

    return check


def _anomaly_ids(rules: dict[str, list[int]]) -> Check:
    def check(t: Tables) -> tuple[str, bool]:
        rows = t["E-11_anomaly_rows"]
        observed = {
            rule: sorted(int(i) for i in rows.loc[rows["rule"] == rule, "Id"]) for rule in rules
        }
        return "; ".join(f"{r}: {ids}" for r, ids in observed.items()), observed == rules

    return check


# --------------------------------------------------------------------- named checks


def _shape(t: Tables) -> tuple[str, bool]:
    rows, columns = _integrity(t, "row_count"), _integrity(t, "column_count")
    return f"{rows} x {columns}", (rows, columns) == ("2930", "82")


def _structure(t: Tables) -> tuple[str, bool]:
    schema = t["E-01_shape_schema"]
    features = int(schema["role"].isin(["model_input", "excluded"]).sum())
    ids = schema.loc[schema["role"] == "identifier", "column"].tolist()
    targets = schema.loc[schema["role"] == "target", "column"].tolist()
    unique = _integrity(t, "Id_unique") == "True" and _integrity(t, "PID_unique") == "True"
    positive = _integrity(t, "SalePrice_positive") == "True"
    observed = (
        f"{features} features; identifiers {ids} (unique: {unique}); "
        f"target {targets} (> 0: {positive})"
    )
    passes = features == 79 and ids == ["Id", "PID"] and targets == ["SalePrice"]
    return observed, passes and unique and positive


def _sale_period(t: Tables) -> tuple[str, bool]:
    years = t["E-32_sale_years"].sort_values("YrSold")
    first, last = years.iloc[0], years.iloc[-1]
    observed = (
        f"{int(first['YrSold'])}-{int(first['first_month'])} to "
        f"{int(last['YrSold'])}-{int(last['last_month'])}"
    )
    passes = (int(first["YrSold"]), int(first["first_month"])) == (2006, 1)
    return observed, passes and (int(last["YrSold"]), int(last["last_month"])) == (2010, 7)


def _scope_rows(t: Tables) -> tuple[str, bool]:
    ids = sorted(int(v) for v in t["E-25_out_of_scope_homes"]["Id"])
    in_scope = int(_integrity(t, "row_count")) - len(ids)
    observed = f"{len(ids)} removed (Ids {ids}); {in_scope} in scope"
    return observed, ids == [1499, 1761, 1768, 2181, 2182] and in_scope == 2925


def _split_sizes(t: Tables) -> tuple[str, bool]:
    balance = t["E-08_split_balance"]
    dev, hold = int(balance["dev_count"].sum()), int(balance["holdout_count"].sum())
    return f"development {dev}, holdout {hold}", (dev, hold) == (2340, 585)


def _feature_types(t: Tables) -> tuple[str, bool]:
    schema = t["E-01_shape_schema"]
    features = schema[schema["role"].isin(["model_input", "excluded"])]
    numeric = int(features["schema_dtype"].isin(["int", "float"]).sum())
    text = int((features["schema_dtype"] == "category").sum())
    observed = f"{len(features)} features: {numeric} numeric, {text} text"
    return observed, (numeric, text) == (36, 43)


def _price_range(t: Tables) -> tuple[str, bool]:
    low, high = _stat(t, "dollars", "min"), _stat(t, "dollars", "max")
    return f"${low:,.0f} to ${high:,.0f} (development set)", low >= 12789 and high <= 755000


def _right_skew(t: Tables) -> tuple[str, bool]:
    mean, median = _stat(t, "dollars", "mean"), _stat(t, "dollars", "median")
    return f"mean ${mean:,.0f} vs median ${median:,.0f}", mean > median


def _log_symmetry(t: Tables) -> tuple[str, bool]:
    raw, log = _stat(t, "dollars", "skewness"), _stat(t, "log1p", "skewness")
    return f"skewness {raw:.2f} -> {log:.2f}", abs(log) < 0.25


def _heteroscedasticity(t: Tables) -> tuple[str, bool]:
    spread = t["E-07_residual_spread"]

    def ratios(scale: str) -> tuple[float, float]:
        sd = spread.loc[spread["scale"] == scale, "residual_sd"]
        return float(sd.iloc[-1] / sd.iloc[0]), float(sd.max() / sd.min())

    top_bottom, dollar_range = ratios("dollars")
    _, log_range = ratios("log1p")
    observed = (
        f"dollar residual SD top/bottom decile ratio {top_bottom:.2f}; max/min SD across "
        f"deciles: dollars {dollar_range:.2f}, log {log_range:.2f}"
    )
    return observed, top_bottom >= 1.5 and log_range < dollar_range


def _top_two(t: Tables) -> tuple[str, bool]:
    top = t["E-15_target_correlation"].sort_values("rank")["feature"].head(2).tolist()
    observed = f"rank 1 {top[0]}, rank 2 {top[1]} (|Spearman| with log price)"
    return observed, top == ["OverallQual", "GrLivArea"]


def _convex_quality(t: Tables) -> tuple[str, bool]:
    profile = t["E-17_overallqual_price_profile"]
    used = profile[profile["n"] >= MIN_LEVEL_ROWS]
    level = used["OverallQual"].to_numpy(dtype="float64")

    def r2(values: np.ndarray) -> float:
        fitted = np.polyval(np.polyfit(level, values, 1), level)
        return float(1 - ((values - fitted) ** 2).sum() / ((values - values.mean()) ** 2).sum())

    dollars = r2(used["median_price"].to_numpy(dtype="float64"))
    logs = r2(used["median_log_price"].to_numpy(dtype="float64"))
    steps = np.diff(used["median_price"].to_numpy(dtype="float64"))
    rising = bool(steps[len(steps) // 2 :].mean() > steps[: len(steps) // 2].mean())
    observed = (
        f"linear-fit R² of median price by level: dollars {dollars:.3f}, log {logs:.3f}; "
        f"upper-half dollar steps larger: {rising}"
    )
    return observed, rising and logs > dollars


def _condition_weaker(t: Tables) -> tuple[str, bool]:
    cond, qual = _spearman(t, "OverallCond"), _spearman(t, "OverallQual")
    return f"|Spearman| OverallCond {cond:.3f}, OverallQual {qual:.3f}", cond < 0.3 and cond < qual


def _lotarea_skew(t: Tables) -> tuple[str, bool]:
    skew = _num(t, "LotArea", "skewness")
    return f"skewness {skew:.2f}", skew > 2


def _lotfrontage(t: Tables) -> tuple[str, bool]:
    row = _row(t, "E-09_missing_values", "column", "LotFrontage")
    observed = (
        f"{int(row['n_missing'])} missing ({float(row['pct_missing']):.1f}%), "
        f"{row['classification']}"
    )
    return observed, int(row["n_missing"]) == 490 and row["classification"] == "unknown"


def _alley(t: Tables) -> tuple[str, bool]:
    pct = _miss(t, "Alley", "pct_missing")
    return f"{pct:.1f}% missing", pct > 90


def _street(t: Tables) -> tuple[str, bool]:
    gravel = _count(t, "Street", "Grvl")
    return f"{gravel} gravel", gravel == 12


def _remodel_floor(t: Tables) -> tuple[str, bool]:
    minimum = _num(t, "YearRemodAdd", "min")
    _, ids = _flag(t, "remodel_year_at_floor")
    observed = f"minimum {minimum:.0f}; {len(ids)} pre-1950 houses at 1950 (raw file)"
    return observed, minimum == 1950 and len(ids) > 0


def _no_basement(t: Tables) -> tuple[str, bool]:
    absent = int(_miss(t, "BsmtQual", "n_absent"))
    status, _ = _flag(t, "basement_absent_with_area")
    return f"{absent} absent rows; basement_absent_with_area {status}", (absent, status) == (
        79,
        "clear",
    )


def _no_garage(t: Tables) -> tuple[str, bool]:
    missing, absent = (
        int(_miss(t, "GarageType", "n_missing")),
        int(_miss(t, "GarageType", "n_absent")),
    )
    status, _ = _flag(t, "garage_absent_with_area")
    observed = f"GarageType {missing} missing, {absent} absent; garage_absent_with_area {status}"
    return observed, (missing, absent, status) == (157, 157, "clear")


def _foundation_age(t: Tables) -> tuple[str, bool]:
    frame = t["E-19_foundation_year_built"]
    frame = frame[frame["n"] >= MIN_LEVEL_ROWS].sort_values("median_year_built", ascending=False)
    top = frame.iloc[0]
    observed = f"latest median YearBuilt: {top['Foundation']} ({top['median_year_built']:.0f})"
    return observed, top["Foundation"] == "PConc"


def _utilities(t: Tables) -> tuple[str, bool]:
    others = int(_values(t, "Utilities")["count"].sum()) - _count(t, "Utilities", "AllPub")
    return f"{others} non-AllPub", others == 3


def _central_air(t: Tables) -> tuple[str, bool]:
    no, yes = _category_median(t, "CentralAir", "N"), _category_median(t, "CentralAir", "Y")
    return f"median log price N {no:.3f}, Y {yes:.3f}", no < yes


def _electrical(t: Tables) -> tuple[str, bool]:
    row = _row(t, "E-09_missing_values", "column", "Electrical")
    observed = f"{int(row['n_missing'])} missing, {row['classification']}"
    return observed, int(row["n_missing"]) == 1 and row["classification"] == "unknown"


def _fireplace(t: Tables) -> tuple[str, bool]:
    missing, absent = (
        int(_miss(t, "FireplaceQu", "n_missing")),
        int(_miss(t, "FireplaceQu", "n_absent")),
    )
    return f"{missing} missing, {absent} with Fireplaces = 0", (missing, absent) == (1422, 1422)


def _pool(t: Tables) -> tuple[str, bool]:
    missing, absent = int(_miss(t, "PoolQC", "n_missing")), int(_miss(t, "PoolQC", "n_absent"))
    pct = _miss(t, "PoolQC", "pct_missing")
    observed = f"{missing} missing ({pct:.2f}%), {absent} with PoolArea = 0"
    return observed, (missing, absent) == (2917, 2917)


def _misc(t: Tables) -> tuple[str, bool]:
    pct, zero = _miss(t, "MiscFeature", "pct_missing"), _num(t, "MiscVal", "share_zero")
    observed = f"MiscFeature {pct:.1f}% missing; MiscVal zero {100 * zero:.1f}%"
    return observed, 95 <= pct <= 97 and zero >= NEAR_CONSTANT_SHARE


def _missing_table(t: Tables) -> tuple[str, bool]:
    documented = {
        "PoolQC": 2917, "MiscFeature": 2824, "Alley": 2732, "Fence": 2358, "FireplaceQu": 1422,
        "GarageType": 157, "GarageFinish": 159, "GarageQual": 159, "GarageCond": 159,
        "GarageYrBlt": 159, "BsmtQual": 80, "BsmtCond": 80, "BsmtFinType1": 80,
        "BsmtExposure": 83, "BsmtFinType2": 81, "MasVnrType": 23, "MasVnrArea": 23,
    }  # fmt: skip
    observed = {c: int(_miss(t, c, "n_missing")) for c in documented}
    wrong = {c: n for c, n in observed.items() if n != documented[c]}
    return (f"all {len(documented)} counts match" if not wrong else f"differs: {wrong}"), not wrong


def _masvnr_parsing(t: Tables) -> tuple[str, bool]:
    row = _row(t, "E-04_parsing_comparison", "column", "MasVnrType")
    explicit, default = int(row["explicit_missing"]), int(row["default_missing"])
    none_text = _count(t, "MasVnrType", "None")
    observed = f"explicit {explicit}, default {default}, 'None' {none_text}"
    return observed, (explicit, default, none_text) == (23, 1775, 1752)


def _top_correlates(t: Tables) -> tuple[str, bool]:
    listed = ("OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalBsmtSF", "1stFlrSF",
              "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd")  # fmt: skip
    ranks = {c: _rank(t, c) for c in listed}
    observed = ", ".join(f"{c} #{r}" for c, r in ranks.items())
    return observed, all(r <= TOP_CORRELATES_WINDOW for r in ranks.values())


def _cardinality(t: Tables) -> tuple[str, bool]:
    documented = {
        "Neighborhood": 28, "Exterior2nd": 17, "MSSubClass": 16, "Exterior1st": 16,
        "Condition1": 9, "SaleType": 10, "Condition2": 8, "HouseStyle": 8, "RoofMatl": 8,
    }  # fmt: skip
    observed = {c: len(_values(t, c)) for c in documented}
    wrong = {c: n for c, n in observed.items() if n != documented[c]}
    text = ", ".join(f"{c} {n}" for c, n in observed.items())
    return (text if not wrong else f"{text}; differs: {wrong}"), not wrong


def _other_cardinality(t: Tables) -> tuple[str, bool]:
    listed = {"Neighborhood", "Exterior2nd", "MSSubClass", "Exterior1st", "Condition1",
              "SaleType", "Condition2", "HouseStyle", "RoofMatl"}  # fmt: skip
    audit = t["E-02_allowed_values_audit"]
    others = audit[~audit["column"].isin(listed)]
    in_range = int(others["n_observed"].between(2, 7).sum())
    observed = f"{in_range} of {len(others)} other categoricals have 2-7 levels"
    return observed, in_range / len(others) >= 0.75


def _smallest_neighborhood(t: Tables) -> tuple[str, bool]:
    smallest = int(_values(t, "Neighborhood")["count"].min())
    return f"smallest: {smallest} sale(s)", smallest == 1


def _neighborhood_spread(t: Tables) -> tuple[str, bool]:
    medians = t["E-22_neighborhoods"]["median_log_price"]
    spread = float(medians.max() - medians.min())
    return f"range {spread:.2f} log units", spread >= 0.5


def _partial_sales(t: Tables) -> tuple[str, bool]:
    frame = t["E-25_out_of_scope_homes"].query("SaleCondition == 'Partial'")
    residuals = ", ".join(f"{v:.1f}" for v in frame["residual_in_sd"])
    observed = f"{len(frame)} partial sales, residuals {residuals} SD"
    return observed, len(frame) == 3 and bool((frame["residual_in_sd"] < -3).all())


def _expensive_homes(t: Tables) -> tuple[str, bool]:
    frame = t["E-25_out_of_scope_homes"].query("SaleCondition != 'Partial'")
    percentiles = ", ".join(f"{v:.2f}" for v in frame["price_percentile_in_raw_file"])
    observed = f"{len(frame)} homes at percentiles {percentiles}"
    return observed, len(frame) == 2 and bool((frame["price_percentile_in_raw_file"] >= 99).all())


def _leverage(t: Tables) -> tuple[str, bool]:
    frame = t["E-26_leverage_comparison"]
    all_rows, in_scope = frame.iloc[0]["slope_per_1000_sqft"], frame.iloc[1]["slope_per_1000_sqft"]
    return (
        f"slope per 1,000 sq ft: all {all_rows:.3f}, in scope {in_scope:.3f}",
        in_scope > all_rows,
    )


def _new_construction(t: Tables) -> tuple[str, bool]:
    frame = t["E-31_sale_outcome_context"]
    frame = frame[(frame["column"] == "SaleType") & (frame["n"] >= MIN_LEVEL_ROWS)]
    top = frame.sort_values("median_log_price", ascending=False).iloc[0]
    observed = (
        f"highest: {top['level']} ({top['difference_vs_overall_median']:+.2f} log vs overall)"
    )
    return observed, top["level"] == "New"


def _distressed_sales(t: Tables) -> tuple[str, bool]:
    frame = t["E-31_sale_outcome_context"]
    medians = frame[frame["column"] == "SaleCondition"].set_index("level")["median_log_price"]
    abnormal, family, normal = medians["Abnorml"], medians["Family"], medians["Normal"]
    observed = f"Abnorml {abnormal:.3f}, Family {family:.3f}, Normal {normal:.3f}"
    return observed, abnormal < normal and family < normal


def _mostly_absent(t: Tables) -> tuple[str, bool]:
    frame = t["E-09_missing_values"]
    share = frame["n_absent"].sum() / frame["n_missing"].sum()
    observed = f"{100 * share:.1f}% of {int(frame['n_missing'].sum())} missing cells absent"
    return observed, share >= 0.9


# ------------------------------------------------------------------------ properties


@dataclass(frozen=True)
class Property:
    """One documented DOC-02 property and how it is verified."""

    pid: str
    section: str
    statement: str
    evidence: str
    rule: str
    check: Check
    adr_fixed: bool = False


_P = Property
PROPERTIES: tuple[Property, ...] = (
    _P("P-01", "§1.2", "Raw file has 2,930 rows and 82 columns", "E-03", "exact counts", _shape),
    _P("P-02", "§1.2", "79 features; identifiers Id and PID (unique); target SalePrice (> 0)",
       "E-01, E-03", "exact roles; uniqueness and positivity checks pass", _structure),
    _P("P-03", "§1.2, §14.3", "Sales run January 2006 to July 2010; 2010 covers January-July only",
       "E-32", "first month of 2006 is 1; last year is 2010 and ends in month 7", _sale_period),
    _P("P-04", "§1.2, §9.2",
       "The scope rule removes 5 homes (Ids 1499, 1761, 1768, 2181, 2182), leaving 2,925 rows",
       "E-25, E-03", "exact Ids and count", _scope_rows, adr_fixed=True),
    _P("P-05", "§1.2", "Development set 2,340 rows; holdout 585 rows (80/20)", "E-08",
       "exact counts", _split_sizes, adr_fixed=True),
    _P("P-06", "§1.3", "The 79 features are 36 numeric and 43 text columns", "E-01",
       "exact counts", _feature_types),
    _P("P-07", "§1.4, §5.1", "SalePrice ranges from $12,789 to $755,000", "E-05",
       "development-set range lies within the documented raw-file range (the documented "
       "extremes are in the holdout or out of scope)", _price_range),
    _P("P-08", "§1.4, §5.1", "Median SalePrice about $160,000", "E-05", "within 10%",
       _approx("dollars", "median", 160000, LOCATION_TOLERANCE)),
    _P("P-09", "§1.4, §5.1", "Mean SalePrice about $180,800", "E-05", "within 10%",
       _approx("dollars", "mean", 180800, LOCATION_TOLERANCE)),
    _P("P-10", "§5.1", "SalePrice skewness about 1.7", "E-05", "within 20%",
       _approx("dollars", "skewness", 1.7, SHAPE_TOLERANCE)),
    _P("P-11", "§5.1", "SalePrice excess kurtosis about 5.1", "E-05", "within 20%",
       _approx("dollars", "excess_kurtosis", 5.1, SHAPE_TOLERANCE)),
    _P("P-12", "§5.1", "Mean above median: right skew", "E-05", "mean > median", _right_skew),
    _P("P-13", "§5.3", "log1p reduces skewness to about 0 (near symmetric)", "E-05, E-06",
       "|log1p skewness| < 0.25", _log_symmetry),
    _P("P-14", "§5.2",
       "Error spread grows with price on the raw scale and is much more constant on the log scale",
       "E-07", "dollar top/bottom decile SD ratio >= 1.5 and log max/min SD ratio below the "
       "dollar one", _heteroscedasticity),
    _P("P-15", "§2.2, §3.5",
       "OverallQual and GrLivArea are the two strongest correlates of price, OverallQual first",
       "E-15", "ranks 1 and 2 by |Spearman|", _top_two),
    _P("P-16", "§3.5", "OverallQual's effect is increasing and convex in dollars and closer to "
       "linear in log space", "E-17", "upper-half dollar steps larger than lower-half; log R² > "
       "dollar R² (levels with >= 10 rows)", _convex_quality),
    _P("P-17", "§3.5", "OverallCond relates to price more weakly than OverallQual", "E-15",
       "|Spearman| of OverallCond < 0.3 and below OverallQual", _condition_weaker),
    _P("P-18", "§3.1", "Condition2 is almost always Norm", "E-02", "share >= 95%",
       _dominant("Condition2", "Norm")),
    _P("P-19", "§3.2, §7.1", "LotArea is heavily right-skewed", "E-13", "skewness > 2",
       _lotarea_skew),
    _P("P-20", "§3.2, §6.3",
       "LotFrontage is missing in 490 rows (about 17%) and is genuinely unknown", "E-09",
       "exact count; classified unknown", _lotfrontage),
    _P("P-21", "§3.2", "Alley is mostly NA (no alley access)", "E-09", "> 90% missing", _alley),
    _P("P-22", "§3.2, §8.3", "Street is almost always paved (12 gravel streets)", "E-02",
       "exact gravel count", _street),
    _P("P-23", "§3.4", "GrLivArea equals 1stFlrSF + 2ndFlrSF + LowQualFinSF", "E-11",
       "consistency rule clear", _flag_clear("grlivarea_components_differ")),
    _P("P-24", "§3.4, §7.3", "GrLivArea and TotRmsAbvGrd are strongly correlated", "E-16",
       "|Pearson| >= 0.7", _strong_pair("GrLivArea", "TotRmsAbvGrd")),
    _P("P-25", "§3.6, §10.5", "YearRemodAdd has a recording floor of 1950", "E-13, E-11",
       "minimum is 1950 and pre-1950 houses sit at the floor", _remodel_floor),
    _P("P-26", "§3.7, §6.6", "79 houses without a basement have NA in every basement "
       "categorical and TotalBsmtSF = 0", "E-09, E-11",
       "79 absent rows; no absent pattern with area", _no_basement),
    _P("P-27", "§3.8, §6.6", "157 houses without a garage: garage categoricals and GarageYrBlt "
       "NA, GarageArea and GarageCars 0", "E-09, E-11",
       "157 absent GarageType rows; no absent pattern with area", _no_garage),
    _P("P-28", "§3.8, §7.3", "GarageCars and GarageArea are strongly correlated", "E-16",
       "|Pearson| >= 0.7", _strong_pair("GarageCars", "GarageArea")),
    _P("P-29", "§3.9", "RoofMatl is dominated by standard composite shingle", "E-02",
       "share >= 95%", _dominant("RoofMatl", "CompShg")),
    _P("P-30", "§3.9", "Foundation acts as an age proxy: poured concrete is common in newer "
       "homes", "E-19", "PConc has the latest median YearBuilt among types with >= 10 rows",
       _foundation_age),
    _P("P-31", "§3.10, §8.3", "Utilities: all but three houses have all public utilities",
       "E-02", "exactly 3 non-AllPub", _utilities),
    _P("P-32", "§3.10", "Heating is dominated by gas forced-air", "E-02", "share >= 95%",
       _dominant("Heating", "GasA")),
    _P("P-33", "§3.10", "Houses without central air tend to be cheaper", "E-21",
       "median log price N < Y", _central_air),
    _P("P-34", "§3.10, §6.3", "Electrical has a single genuinely unknown value", "E-09",
       "1 missing; classified unknown", _electrical),
    _P("P-35", "§3.11, §6.2", "FireplaceQu is NA for 1,422 houses without a fireplace",
       "E-09", "1,422 missing, all with Fireplaces = 0", _fireplace),
    _P("P-36", "§3.12, §7.1", "Outdoor and amenity areas are zero-inflated (porches, 2ndFlrSF, "
       "MasVnrArea, PoolArea)", "E-13", "share of exact zeros >= 40% for each",
       _all_numeric(("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "2ndFlrSF",
                     "MasVnrArea", "PoolArea"), "share_zero", ZERO_INFLATED_SHARE, percent=True)),
    _P("P-37", "§3.12", "Only 13 houses have a pool; PoolQC is about 99.6% NA", "E-09",
       "2,917 missing (= 2,930 - 13), all with PoolArea = 0", _pool),
    _P("P-38", "§3.13", "MiscFeature is about 96% NA and MiscVal is almost always zero",
       "E-09, E-13", "95-97% missing; MiscVal zero share >= 90%", _misc),
    _P("P-39", "§6.2", "Per-column missing counts of the feature-absent columns (§6.2 table)",
       "E-09", "every count matches", _missing_table),
    _P("P-40", "§6.5", "MasVnrType has 23 missing values and 1,752 literal 'None'; default "
       "parsing would report 1,775 missing", "E-04, E-02", "exact counts", _masvnr_parsing),
    _P("P-41", "§6.6", "Basement anomalies: BsmtExposure missing with a basement (Ids 67, 1797, "
       "2780); BsmtFinType2 missing with finished area (Id 445)", "E-11", "exact Ids",
       _anomaly_ids({"basement_exposure_missing_with_basement": [67, 1797, 2780],
                     "basement_fintype2_missing_with_area": [445]})),
    _P("P-42", "§6.6", "Related unrecorded rows: Id 1342 (all basement fields); Ids 1357 and "
       "2237 (garage fields)", "E-11", "exact Ids",
       _anomaly_ids({"basement_areas_unrecorded": [1342],
                     "garage_partially_recorded": [1357, 2237]})),
    _P("P-43", "§6.6", "MasVnrType and MasVnrArea are missing together", "E-11",
       "consistency rule clear", _flag_clear("masvnr_type_area_missing_mismatch")),
    _P("P-44", "§7.1", "Strong right skew in LotArea, GrLivArea, MasVnrArea, MiscVal, PoolArea, "
       "LowQualFinSF, 3SsnPorch", "E-13", "skewness > 1 for each",
       _all_numeric(("LotArea", "GrLivArea", "MasVnrArea", "MiscVal", "PoolArea",
                     "LowQualFinSF", "3SsnPorch"), "skewness", 1.0, percent=False)),
    _P("P-45", "§7.2", "Near-constant numeric features: PoolArea, 3SsnPorch, LowQualFinSF, "
       "MiscVal, KitchenAbvGr", "E-13", "top value share >= 90% for each",
       _all_numeric(("PoolArea", "3SsnPorch", "LowQualFinSF", "MiscVal", "KitchenAbvGr"),
                    "top_value_share", NEAR_CONSTANT_SHARE, percent=True)),
    _P("P-46", "§7.3", "Strongest numeric correlates: OverallQual, GrLivArea, GarageCars, "
       "GarageArea, TotalBsmtSF, 1stFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd",
       "E-15", f"each ranks within the top {TOP_CORRELATES_WINDOW} by |Spearman|",
       _top_correlates),
    _P("P-47", "§7.3", "TotalBsmtSF and 1stFlrSF are strongly correlated", "E-16",
       "|Pearson| >= 0.7", _strong_pair("TotalBsmtSF", "1stFlrSF")),
    _P("P-48", "§7.3", "YearBuilt and GarageYrBlt are strongly correlated", "E-16",
       "|Pearson| >= 0.7", _strong_pair("YearBuilt", "GarageYrBlt")),
    _P("P-49", "§8.2", "Cardinality: Neighborhood 28, Exterior2nd 17, MSSubClass 16, "
       "Exterior1st 16, Condition1 9, SaleType 10, Condition2/HouseStyle/RoofMatl 8", "E-02",
       "exact level counts (raw file)", _cardinality),
    _P("P-50", "§8.2", "Most other categoricals have between 2 and 7 levels", "E-02",
       ">= 75% of the other categoricals", _other_cardinality),
    _P("P-51", "§8.3", "The smallest neighborhood has 1 sale", "E-02", "exact minimum",
       _smallest_neighborhood),
    _P("P-52", "§8.5, §3.1", "Neighborhood levels have strongly different price levels",
       "E-22, E-21", "median log price range across neighborhoods >= 0.5", _neighborhood_spread),
    _P("P-53", "§8.6", "Quality scales (ExterQual, KitchenQual, BsmtQual) rise clearly with "
       "price", "E-23", "monotonic across levels with >= 10 rows",
       _ordinal(("ExterQual", "KitchenQual", "BsmtQual"), expect_monotonic=True)),
    _P("P-54", "§8.6", "Condition scales (ExterCond, GarageCond) show a weaker pattern", "E-23",
       "not monotonic, or spread below every quality scale's",
       _ordinal(("ExterCond", "GarageCond"), expect_monotonic=False)),
    _P("P-55", "§9.2", "3 of the out-of-scope homes are partial sales priced far below their "
       "size trend", "E-25", "SaleCondition Partial with residual < -3 SD", _partial_sales),
    _P("P-56", "§9.2", "The other 2 out-of-scope homes are among the most expensive", "E-25",
       "price percentile >= 99 in the raw file", _expensive_homes),
    _P("P-57", "§9.2", "The out-of-scope homes flatten the size-price slope", "E-26",
       "slope on in-scope rows > slope on all rows", _leverage),
    _P("P-58", "§11.4", "New-construction sales are associated with higher prices", "E-31",
       "SaleType New has the highest median among levels with >= 10 rows", _new_construction),
    _P("P-59", "§11.5", "Abnormal and family sales tend to sell below market value", "E-31",
       "Abnorml and Family medians below Normal", _distressed_sales),
    _P("P-60", "§1.5, §15.1", "Most missing values mean 'feature absent', not 'unknown'",
       "E-09", ">= 90% of missing cells are absent", _mostly_absent),
)  # fmt: skip


def build_register(tables: Tables) -> pd.DataFrame:
    """Evaluate every documented property against the saved tables."""
    rows = []
    for prop in PROPERTIES:
        try:
            observed, confirmed = prop.check(tables)
        except (KeyError, IndexError, ValueError, ZeroDivisionError) as exc:
            observed, confirmed = f"evidence unavailable: {type(exc).__name__}: {exc}", False
        rows.append(
            {
                "property_id": prop.pid,
                "doc02_section": prop.section,
                "documented_property": prop.statement,
                "evidence": prop.evidence,
                "rule": prop.rule,
                "observed": observed,
                "status": "confirmed" if confirmed else "not confirmed",
                "adr_fixed": prop.adr_fixed,
            }
        )
    return pd.DataFrame(rows)

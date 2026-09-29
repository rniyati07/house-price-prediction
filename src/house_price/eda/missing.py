"""Missing-value report, E-09 to E-12 (DOC-02 §6, §13.3).

E-09 to E-11 use the raw file without the target; E-12 uses the development set.
The per-column counts come from the M2 profile (``missing_value_profile``) and the
consistency checks from the M2 rules (``quality_flags``); this module adds the
absent-versus-unknown classification and the anomaly detail.

A missing value is *absent* when a companion column shows the feature does not exist
(for example ``GarageArea == 0`` for the garage columns), and *unknown* when the companion
shows the feature exists or is itself unrecorded. Columns without a companion are classified
by the data dictionary (DOC-02 §6.2, §6.3).
"""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from house_price.config import SchemaConfig
from house_price.data.profile import missing_value_profile, quality_flags
from house_price.eda.context import EDAContext, log_price
from house_price.eda.outputs import Outputs, save_figure, save_table

Mask = Callable[[pd.DataFrame], pd.Series]

_GARAGE = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt",
           "GarageCars", "GarageArea")  # fmt: skip
_BASEMENT = ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
             "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath",
             "BsmtHalfBath")  # fmt: skip

# Companion evidence that the feature is absent (DOC-02 §6.2, §6.6).
ABSENCE_EVIDENCE: dict[str, tuple[str, Mask]] = {
    "PoolQC": ("PoolArea == 0", lambda f: f["PoolArea"] == 0),
    "FireplaceQu": ("Fireplaces == 0", lambda f: f["Fireplaces"] == 0),
    "MiscFeature": ("MiscVal == 0", lambda f: f["MiscVal"] == 0),
    **{c: ("GarageArea == 0", lambda f: f["GarageArea"] == 0) for c in _GARAGE},
    **{c: ("TotalBsmtSF == 0", lambda f: f["TotalBsmtSF"] == 0) for c in _BASEMENT},
}
# Columns whose NA the data dictionary defines as "absent" but that have no companion.
DICTIONARY_ABSENT = ("Alley", "Fence", "MasVnrType", "MasVnrArea")
# Columns whose missing values are genuinely unknown (DOC-02 §6.3).
UNKNOWN = ("LotFrontage", "Electrical")

LAYER1_CATEGORICAL = ("PoolQC", "Alley", "Fence", "FireplaceQu", "MiscFeature", "GarageType",
                      "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
                      "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType")  # fmt: skip
LAYER1_NUMERIC = ("GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
                  "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "MasVnrArea")  # fmt: skip

# Columns shown for each E-11 anomaly rule, and how the approved rule treats the rows.
ANOMALY_RULES: dict[str, tuple[tuple[str, ...], str]] = {
    "basement_exposure_missing_with_basement": (
        ("BsmtQual", "BsmtCond", "BsmtExposure", "TotalBsmtSF"),
        ("Basement exists; missing BsmtExposure is unknown but is filled 'None' by the "
        "column-wise Layer 1 rule (known approximation, DOC-02 §6.6)")),
    "basement_fintype2_missing_with_area": (
        ("BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2"),
        ("Finished area exists; missing BsmtFinType2 is unknown but is filled 'None' by "
        "Layer 1 (known approximation, DOC-02 §6.6)")),
    "basement_areas_unrecorded": (
        ("BsmtQual", "TotalBsmtSF", "BsmtFinSF1", "BsmtUnfSF", "BsmtFullBath"),
        ("Every basement field unrecorded; Layer 1 fills 'None'/0, i.e. treats it as no "
        "basement (known approximation, DOC-02 §6.6)")),
    "garage_partially_recorded": (
        ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "GarageYrBlt",
         "GarageCars", "GarageArea"),
        ("Garage type recorded, other fields unknown; Layer 1 fills 'None'/0 and GarageAge "
        "stays missing for Layer 2 (known approximation, DOC-02 §6.6, DOC-03 §6.4)")),
}  # fmt: skip


def planned_handling(column: str) -> str:
    """How the approved design treats missing values in ``column`` (ADR-05, DOC-03 §6-7)."""
    if column == "GarageYrBlt":
        return "Not imputed: replaced by HasGarage and GarageAge (M4), then dropped"
    if column in LAYER1_CATEGORICAL:
        return "Layer 1 semantic fill with 'None' (M4)"
    if column in LAYER1_NUMERIC:
        return "Layer 1 semantic fill with 0 (M4)"
    if column in UNKNOWN:
        kind = "median + missing indicator" if column == "LotFrontage" else "most frequent"
        return f"Layer 2 fitted imputation: {kind} (M5)"
    return "Layer 2 fitted imputation (M5)"


def e09_missing_table(raw: pd.DataFrame, schema: SchemaConfig) -> pd.DataFrame:
    """E-09: count, share, absent/unknown split, classification, and planned handling."""
    present = [spec for spec in schema.columns if spec.name in raw.columns]
    profile = missing_value_profile(raw, schema.model_copy(update={"columns": present}))
    rows = []
    for rec in profile.itertuples():
        missing = raw[rec.column].isna()
        if rec.column in ABSENCE_EVIDENCE:
            basis, mask_fn = ABSENCE_EVIDENCE[rec.column]
            absent = int((missing & mask_fn(raw).fillna(False).astype(bool)).sum())
        elif rec.column in DICTIONARY_ABSENT:
            basis, absent = "data dictionary (no companion column)", int(missing.sum())
        else:
            basis, absent = "genuinely unknown (DOC-02 §6.3)", 0
        unknown = int(missing.sum()) - absent
        label = "absent" if unknown == 0 else "unknown" if absent == 0 else "mixed"
        rows.append({
            "column": rec.column, "n_missing": rec.n_missing, "pct_missing": rec.pct_missing,
            "n_absent": absent, "n_unknown": unknown, "classification": label,
            "evidence": basis, "planned_handling": planned_handling(rec.column),
        })  # fmt: skip
    return pd.DataFrame(rows)


def e10_missingness_figure(table: pd.DataFrame) -> Figure:
    """E-10: missing percentage per column, colored by classification."""
    figure, ax = plt.subplots(figsize=(9, 8))
    ordered = table.sort_values("pct_missing", ascending=True)
    palette = {"absent": "tab:blue", "unknown": "tab:orange", "mixed": "tab:purple"}
    ax.barh(ordered["column"], ordered["pct_missing"],
            color=[palette[c] for c in ordered["classification"]])  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlabel("% missing (log scale)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, list(palette), title="classification", loc="lower right")
    ax.set_title("E-10 Missing values in the raw file, by classification")
    figure.tight_layout()
    return figure


def e11_consistency(raw: pd.DataFrame, id_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """E-11: the M2 consistency rules, plus each basement/garage anomaly row by ``Id``."""
    flags = quality_flags(raw, id_column)
    checks = pd.DataFrame(
        [{**f.model_dump(exclude={"ids"}), "ids": "|".join(map(str, f.ids))} for f in flags]
    )
    rows = []
    for flag in flags:
        if flag.name not in ANOMALY_RULES:
            continue
        columns, interpretation = ANOMALY_RULES[flag.name]
        for row_id in flag.ids:
            record = raw.loc[raw[id_column] == row_id].iloc[0]
            values = "; ".join(f"{c}={'NA' if pd.isna(record[c]) else record[c]}" for c in columns)
            rows.append({id_column: row_id, "rule": flag.name, "related_values": values,
                         "interpretation": interpretation})  # fmt: skip
    anomalies = pd.DataFrame(rows, columns=[id_column, "rule", "related_values", "interpretation"])
    return checks, anomalies.sort_values([id_column, "rule"], ignore_index=True)


def e12_lotfrontage_study(dev: pd.DataFrame, target: str) -> tuple[pd.DataFrame, Figure]:
    """E-12: log price for development rows with and without ``LotFrontage``."""
    frame = pd.DataFrame({
        "LotFrontage": dev["LotFrontage"].isna().map({True: "missing", False: "recorded"}),
        "log_price": log_price(dev, target),
    })  # fmt: skip
    table = (frame.groupby("LotFrontage", sort=True)["log_price"]
             .agg(n="size", median_log_price="median", mean_log_price="mean",
                  sd_log_price="std").reset_index())  # fmt: skip
    table["share_pct"] = 100 * table["n"] / table["n"].sum()
    figure, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=frame, x="LotFrontage", y="log_price", order=["recorded", "missing"],
                ax=ax, color="tab:blue")  # fmt: skip
    ax.set_title("E-12 log1p(SalePrice) by LotFrontage missingness (development set)")
    figure.tight_layout()
    return table, figure


def run(ctx: EDAContext) -> Outputs:
    """Compute and save E-09 to E-12."""
    out = Outputs()
    raw = ctx.raw_features
    table = e09_missing_table(raw, ctx.schema)
    save_table(out, ctx.tables_dir, "E-09_missing_values", table)
    save_figure(out, ctx.figures_dir, "E-10_missingness", e10_missingness_figure(table))
    checks, anomalies = e11_consistency(raw, ctx.id_column)
    save_table(out, ctx.tables_dir, "E-11_consistency_checks", checks)
    save_table(out, ctx.tables_dir, "E-11_anomaly_rows", anomalies)
    study, figure = e12_lotfrontage_study(ctx.dev, ctx.target)
    save_table(out, ctx.tables_dir, "E-12_lotfrontage_missingness", study)
    save_figure(out, ctx.figures_dir, "E-12_lotfrontage_missingness", figure)
    return out

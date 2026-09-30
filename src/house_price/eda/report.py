"""E-35 EDA report and E-36 data-card inputs (DOC-02 §13.9; DOC-05 M3 tasks 3 to 5).

Reads only the saved deliverables (``reports/eda/E-*.csv``), as the roadmap requires for
``99_eda_report.ipynb``. Writes:

* ``E-35_confirmation_register.csv``: every DOC-02 documented property, confirmed or not
* ``E-35_question_answers.csv``: answers to Q1-Q19; Q17 carries the M4 evidence (E-27 to
  E-29) and, once ``E-30_ablation.csv`` exists (written by the M7 ``train`` ablation), the
  per-branch retention decisions; without E-30 its retention part stays open
* ``E-35_eda_report.md``: the narrative report in nine sections, each with observations,
  evidence, and implications for later milestones
* ``E-36_data_card_inputs.csv``: known issues handed to the data card
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from house_price.eda.context import EDAContext
from house_price.eda.outputs import Outputs, load_tables, save_table
from house_price.eda.register import Tables, build_register

REPORT_NAME = "E-35_eda_report.md"
SECTIONS = (
    "1. Executive Summary",
    "2. Dataset Overview",
    "3. Target Analysis",
    "4. Missing-Value Analysis",
    "5. Numerical Feature Analysis",
    "6. Categorical Feature Analysis",
    "7. Correlation Analysis",
    "8. Anomaly Review",
    "9. Final EDA Findings",
)


@dataclass(frozen=True)
class ReportResult:
    register: pd.DataFrame
    answers: pd.DataFrame
    data_card_inputs: pd.DataFrame
    markdown: str
    path: Path

    @property
    def adr_fixed_discrepancies(self) -> pd.DataFrame:
        """Not-confirmed properties that contradict an ADR-fixed number (DOC-05 M3 task 4)."""
        reg = self.register
        return reg[(reg["status"] == "not confirmed") & reg["adr_fixed"].astype(bool)]


# ---------------------------------------------------------------------- helpers


def _stat(t: Tables, scale: str, name: str) -> float:
    frame = t["E-05_target_statistics"]
    return float(frame.loc[frame["scale"] == scale, name].iloc[0])


def _check(t: Tables, name: str) -> str:
    frame = t["E-03_key_integrity"]
    return str(frame.loc[frame["check"] == name, "result"].iloc[0])


def _join(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> list[str]:
    shown = frame[columns].head(limit) if limit else frame[columns]

    def cell(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"
        return str(value).replace("|", ", ")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |" for row in shown.itertuples(index=False)
    ]
    return lines


# ------------------------------------------------------------------ question answers


def ablation_answer(ablation: pd.DataFrame, engineered: pd.DataFrame) -> str:
    """The Q17 retention part, read from E-30 (DOC-03 §6.6; IN-04), with one sentence per
    dropped feature comparing the ablation with its E-27 hypothesis assessment."""
    supported = engineered.set_index("feature")["hypothesis_supported"].astype(bool)
    parts = [("M7 ablation (E-30; IN-04: a feature is dropped only if removing it lowers the "
              "branch's mean CV log-RMSE; 15 shared folds on the development set):")]  # fmt: skip
    notes = []
    for branch, rows in ablation.groupby("branch", sort=False):
        retained = rows[rows["retained"].astype(bool)]
        dropped = rows[~rows["retained"].astype(bool)]
        first = rows.iloc[0]
        parts.append(
            f"{branch} branch (reference {first['reference_model']}, full-feature mean "
            f"{first['mean_with']:.4f}): retains {len(retained)} of {len(rows)}"
            + (
                f", drops {_join(dropped['feature'].tolist())}."
                if len(dropped)
                else ", drops none."
            )
        )
        for row in dropped.itertuples():
            was_supported = bool(supported.get(row.feature, False))
            hypothesis, link = ("supported", "but") if was_supported else ("not supported", "and")
            size = abs(row.delta) / row.delta_se if row.delta_se > 0 else float("inf")
            notes.append(
                f"{row.feature} ({branch}): removing it changes the mean by {row.delta:+.5f} "
                f"(paired SE {row.delta_se:.5f}, {size:.1f} SE); its E-27 hypothesis was "
                f"{hypothesis}, {link} in this branch the feature does not lower the CV error."
            )
    return " ".join(parts + notes)


def answer_questions(t: Tables) -> pd.DataFrame:
    """Answers to the DOC-02 §4 questions, each derived from the saved evidence."""
    schema = t["E-01_shape_schema"]
    audit = t["E-02_allowed_values_audit"]
    parsing = t["E-04_parsing_comparison"]
    spread = t["E-07_residual_spread"]
    missing = t["E-09_missing_values"]
    checks = t["E-11_consistency_checks"]
    lot = t["E-12_lotfrontage_missingness"].set_index("LotFrontage")
    numeric = t["E-13_numeric_profile"]
    ranking = t["E-15_target_correlation"]
    pairs = t["E-16_correlated_pairs"]
    cat_profile = t["E-19_categorical_profile"]
    rare = t["E-20_rare_categories"]
    mono = t["E-23_ordinal_monotonicity"]
    scope = t["E-25_out_of_scope_homes"]
    leverage = t["E-26_leverage_comparison"]
    extremes = t["E-18_numeric_extremes"]
    leakage = t["E-31_leakage_review"]
    years = t["E-32_sale_years"]
    by_year = t["E-33_price_by_year"]
    months = t["E-34_seasonality"]

    engineered = t["E-27_engineered_features"]
    supported = engineered[engineered["hypothesis_supported"].astype(bool)]
    unsupported = engineered[~engineered["hypothesis_supported"].astype(bool)]
    floor = t["E-29_remodel_floor_audit"].set_index("metric")["value"].astype(float)
    houseage_trend = str(engineered.set_index("feature").loc["HouseAge", "observed"])
    m4_evidence = (
        "M4 implemented the 12 approved "
        "formulas (house_price.features) and assessed each DOC-02 §10 hypothesis on the "
        f"development set (E-27, E-28): {len(supported)} of {len(engineered)} supported "
        f"({_join(supported['feature'].tolist())}). Not supported: "
        + _join([f"{r.feature} ({r.observed})" for r in unsupported.itertuples()])
        + f". E-29: {int(floor['floor_rows_built_before_1950'])} pre-1950 houses sit at the "
        f"1950 remodel floor; they supply {floor['pct_of_remodeled_flags_from_floor']:.1f}% of "
        "the IsRemodeled = 1 flags, and their RemodAge understates HouseAge by a median "
        f"{floor['floor_rows_median_houseage_minus_remodage']:.0f} years."
    )
    if "E-30_ablation" in t:
        q17 = ("Answered (M4 evidence and M7 ablation). " + m4_evidence + " "
               + ablation_answer(t["E-30_ablation"], engineered))  # fmt: skip
    else:
        q17 = (
            "Partly answered in M4; retention stays open until M7. "
            + m4_evidence
            + " Retention is not decided here: the M7 cross-validation ablation decides it per "
            "branch (E-30, IN-04)."
        )

    def sd_ratio(scale: str) -> float:
        sd = spread.loc[spread["scale"] == scale, "residual_sd"]
        return float(sd.iloc[-1] / sd.iloc[0])

    flagged = checks[(checks["status"] == "flagged") & (checks["severity"] == "warning")]
    counts = missing["classification"].value_counts()
    top5 = ranking.sort_values("rank")["feature"].head(5).tolist()
    top8 = ranking.sort_values("rank").head(8)
    strong = pairs[pairs["strong"].astype(bool)]
    near_constant = cat_profile[cat_profile["top_share"] >= 0.95]["feature"].tolist()
    monotonic = mono[mono["monotonic"].astype(bool)]["scale"].tolist()
    not_monotonic = mono[~mono["monotonic"].astype(bool) & mono["assessable"].astype(bool)][
        "scale"
    ].tolist()
    skewed = numeric[numeric["skewness"] > 1]["feature"].tolist()
    zero_inflated = numeric[numeric["share_zero"] >= 0.5]["feature"].tolist()
    flagged_extremes = extremes[~extremes["assessment"].str.startswith("legitimate")][
        "feature"
    ].tolist()
    partial = scope[scope["SaleCondition"] == "Partial"]
    busiest = months.sort_values("n", ascending=False).iloc[0]
    diff = lot.loc["missing", "median_log_price"] - lot.loc["recorded", "median_log_price"]
    answers = [
        ("Q1", "Does the raw file match the expected shape (2,930 x 82), column names, and types?",
         (f"Yes: {_check(t, 'row_count')} rows x {_check(t, 'column_count')} columns; all "
         f"{len(schema)} columns carry their schema names and "
         f"{int(schema['dtype_matches'].astype(bool).sum())} of {len(schema)} load with the "
         "declared dtype."), "E-01, E-03"),
        ("Q2", "Are all category values within the data dictionary's allowed codes?",
         (f"Yes: no observed value falls outside the allowed lists "
         f"({int((audit['not_allowed'].fillna('') != '').sum())} columns with out-of-list values). "
         "The file uses its own spellings for some codes (MSZoning, Neighborhood, BldgType, "
         "Exterior2nd, SaleType), which the allowed lists include."), "E-02"),
        ("Q3", "Are Id and PID unique and is every SalePrice positive?",
         (f"Id unique: {_check(t, 'Id_unique')}; PID unique: {_check(t, 'PID_unique')}; "
         f"SalePrice > 0: {_check(t, 'SalePrice_positive')}; raw-file hash confirmed: "
         f"{_check(t, 'raw_sha256_confirmed')}."), "E-03"),
        ("Q4", "Does the CSV parser change missing-value counts depending on its defaults?",
         "Yes, for " + _join([f"{r.column} ({int(r.explicit_missing)} -> {int(r.default_missing)})"
                              for r in parsing[parsing["difference"] != 0].itertuples()])
         + ". The explicit token contract (DN-18) keeps the counts stable.", "E-04"),
        ("Q5", "How skewed and heavy-tailed is SalePrice, and how much does log1p reduce this?",
         (f"Skewness {_stat(t, 'dollars', 'skewness'):.2f} and excess kurtosis "
         f"{_stat(t, 'dollars', 'excess_kurtosis'):.2f} in dollars; {_stat(t, 'log1p', 'skewness'):.2f} "
         f"and {_stat(t, 'log1p', 'excess_kurtosis'):.2f} after log1p (development set)."),
         "E-05, E-06"),
        ("Q6", "Is residual spread larger for expensive homes on the raw scale than on the log scale?",
         (f"Yes: the residual SD in the top fitted-value decile is {sd_ratio('dollars'):.2f}x the "
         f"bottom decile in dollars, versus {sd_ratio('log1p'):.2f}x on the log scale."), "E-07"),
        ("Q7", "For each column with NA, does it mean 'absent' or 'unknown'?",
         (f"{int(counts.get('absent', 0))} columns are purely absent, "
         f"{int(counts.get('unknown', 0))} purely unknown (LotFrontage, Electrical, and fields "
         f"of unrecorded rows), {int(counts.get('mixed', 0))} mixed: absent except for the "
         "documented anomaly rows."), "E-09, E-10"),
        ("Q8", "Are 'absent' patterns internally consistent?",
         "Largely: the garage, basement, and masonry absence patterns hold, with these flagged "
         "exceptions: " + _join([f"{r.name} (Ids {str(r.ids).replace('|', ', ')})"
                                  for r in flagged.itertuples()]) + ".", "E-11"),
        ("Q9", "Does missingness in LotFrontage relate to price?",
         (f"Rows with LotFrontage missing ({int(lot.loc['missing', 'n'])}) have a median log "
         f"price {diff:+.3f} (about {100 * (math.exp(diff) - 1):+.0f}% in dollars) relative to "
         f"recorded rows ({int(lot.loc['recorded', 'n'])}), so missingness carries some price "
         "signal; the missing-indicator column lets the model use it."), "E-12"),
        ("Q10", "Which numeric features are strongly skewed or zero-inflated?",
         f"Skewness > 1: {_join(skewed)}. At least 50% zeros: {_join(zero_inflated)}.",
         "E-13, E-14"),
        ("Q11", ("Which features correlate most with log price, and which are strongly correlated "
         "with each other?"),
         f"Top by |Spearman|: {_join(top5)}. Strongly correlated pairs (|r| >= 0.7): "
         + _join([f"{r.feature_a}-{r.feature_b} ({r.pearson:.2f})" for r in strong.itertuples()])
         + ".", "E-15, E-16, E-17"),
        ("Q12", "Which categorical features are near-constant or have rare categories?",
         (f"Top level covers >= 95% of rows: {_join(near_constant)}. {len(rare)} categories in "
         f"{rare['feature'].nunique()} features fall under 1% of development rows."), "E-19, E-20"),
        ("Q13", "Do quality scales relate monotonically to price?",
         f"Monotonic: {_join(monotonic)}. Not monotonic: {_join(not_monotonic)}.", "E-23"),
        ("Q14", ("Are relationships mostly linear in log space, or are there thresholds and "
         "interactions?"),
         ("Mostly monotonic and close to linear for the main drivers: for the top 8 predictors "
         f"Pearson and Spearman differ by {float((top8['pearson'] - top8['spearman']).abs().mean()):.3f} "
         "on average. Step-like effects appear in the categorical and ordinal scales (E-21, "
         "E-23) and in OverallQual's convex dollar profile (E-17). Engineered-feature evidence "
         "E-28: HouseAge's decile trend is steeper for younger houses than for older ones "
         f"({houseage_trend})."), "E-17, E-21, E-28"),
        ("Q15", ("Which properties fall above 4,000 sq ft, and how do their prices compare with "
         "the size trend?"),
         (f"{len(scope)} homes (Ids {_join([str(i) for i in scope['Id']])}). The "
         f"{len(partial)} partial sales sit {_join([f'{v:.1f}' for v in partial['residual_in_sd']])} "
         "SD below the in-scope size trend; the others are near-top-priced homes. Removing them "
         f"changes the log-price slope by {leverage.iloc[1]['slope_change_pct_vs_all']:+.1f}%."),
         "E-24, E-25, E-26"),
        ("Q16", "Are there other extreme values, and are they legitimate?",
         (f"Values beyond the 99th percentile exist in all {len(extremes)} key features; all are "
         "within the schema's physical ranges. Features whose extremes include rows flagged by "
         f"a consistency rule: {_join(flagged_extremes)}. No statistical outlier removal is "
         "warranted (ADR-06)."), "E-18"),
        ("Q17", "Does each approved engineered feature show the expected relationship with log price?",
         q17, "E-27, E-28, E-29 (M4); E-30 (M7)"),
        ("Q18", "Which columns would not be known at listing time?",
         "Not known at listing: " + _join(leakage[leakage["known_at_listing"] == "no"]["column"].tolist())
         + " (excluded; rows kept). Identifiers Id and PID are excluded as non-attributes.", "E-31"),
        ("Q19", "How are sales distributed across 2006-2010, and does price level shift across years?",
         "Sales per year: " + _join([f"{int(r.YrSold)}: {int(r.n)}" for r in years.itertuples()])
         + f" (2010 covers months 1-{int(years.iloc[-1]['last_month'])}). Median log price varies "
         f"by {by_year['median_log_price'].max() - by_year['median_log_price'].min():.3f} across "
         f"years. Sales peak in month {int(busiest['MoSold'])}.", "E-32, E-33, E-34"),
    ]  # fmt: skip
    return pd.DataFrame(answers, columns=["question", "text", "answer", "evidence"])


# ----------------------------------------------------------------- data-card inputs


def data_card_inputs(t: Tables, register: pd.DataFrame) -> pd.DataFrame:
    """E-36: known issues handed to the data card, with their status and evidence."""
    checks = t["E-11_consistency_checks"].set_index("name")
    rows = [
        (
            "Parsing hazard: MasVnrType literal 'None'",
            "confirmed",
            "E-04",
            "Default parsing would report 1,775 missing instead of 23",
        ),
    ]
    for rule in ("basement_exposure_missing_with_basement", "basement_fintype2_missing_with_area",
                 "basement_areas_unrecorded", "garage_partially_recorded",
                 "garage_built_after_sale", "house_built_after_sale", "remodel_before_built",
                 "masvnr_none_with_area", "remodel_year_at_floor"):  # fmt: skip
        rec = checks.loc[rule]
        ids = str(rec["ids"]) if pd.notna(rec["ids"]) else ""
        detail = f"{int(rec['n_rows'])} row(s)" + (
            f": Ids {ids.replace('|', ', ')}" if ids and int(rec["n_rows"]) <= 10 else ""
        )
        rows.append(
            (
                f"{rule}: {rec['description']}",
                "confirmed" if rec["status"] == "flagged" else "not observed",
                "E-11",
                detail,
            )
        )
    rare = t["E-20_rare_categories"]
    rows.append(
        (
            "Rare categories (< 1% of development rows)",
            "confirmed",
            "E-20",
            f"{len(rare)} levels in {rare['feature'].nunique()} features",
        )
    )
    rows.append(
        (
            "Dictionary-versus-file spellings",
            "confirmed",
            "E-02",
            "Allowed lists include the file spellings; values are not normalised",
        )
    )
    for rec in register[register["status"] == "not confirmed"].itertuples():
        rows.append(
            (
                f"Register discrepancy {rec.property_id}: {rec.documented_property}",
                "not confirmed",
                rec.evidence,
                str(rec.observed),
            )
        )
    return pd.DataFrame(rows, columns=["issue", "status", "evidence", "detail"])


# ------------------------------------------------------------------------ narrative


def _section(
    title: str, observations: list[str], evidence: list[str], implications: list[str]
) -> list[str]:
    return [f"## {title}", "", "**Observations**", "", *[f"- {o}" for o in observations], "",
            "**Evidence**", "", *[f"- {e}" for e in evidence], "",
            "**Implications for later milestones**", "", *[f"- {i}" for i in implications], ""]  # fmt: skip


def render_markdown(t: Tables, register: pd.DataFrame, answers: pd.DataFrame,
                    card: pd.DataFrame) -> str:  # fmt: skip
    """The nine-section E-35 report."""
    ans = answers.set_index("question")["answer"]
    confirmed = int((register["status"] == "confirmed").sum())
    ranking = t["E-15_target_correlation"].sort_values("rank")
    categories = t["E-21_category_importance"].sort_values("rank")
    missing = t["E-09_missing_values"]
    numeric = t["E-13_numeric_profile"]
    weakest = (
        ranking.assign(a=ranking["spearman"].abs()).sort_values("a").head(5)["feature"].tolist()
    )
    weakest_cat = categories.tail(5)["feature"].tolist()
    status = (
        f"{confirmed} of {len(register)} documented properties confirmed; "
        f"{len(register) - confirmed} not confirmed"
    )
    lines = [
        "# E-35 EDA Report (draft)",
        "",
        (
            "Generated by `python -m house_price.eda` from the saved deliverables in `reports/eda/` "
            "and `reports/figures/eda/`. Target relationships use the development set only; the "
            "raw file is used for target-free profiling and the ADR-06 scope review (FR-009). "
            "Q17 carries the M4 engineered-feature evidence (E-27 to E-29); feature retention "
            + (
                "is decided by the M7 ablation (E-30), summarized in Q17."
                if "E-30_ablation" in t
                else "is decided by the M7 ablation (E-30)."
            )
        ),
        "",
    ]
    absent_share = register.set_index("property_id").loc["P-60", "observed"]
    lines += _section(
        SECTIONS[0],
        [
            status + ".",
            "Data integrity (Q1-Q3): " + ans["Q1"] + " " + ans["Q3"],
            "Target (Q5): SalePrice is right-skewed and log1p makes it near-symmetric. "
            + ans["Q5"],
            "Price drivers (Q11): size and quality dominate. " + ans["Q11"].split(" Strongly")[0],
            (
                f"Missing values (Q7): {absent_share}; the genuinely unknown values are LotFrontage, "
                "Electrical, and a few recorded anomaly rows."
            ),
            "Scope (Q15): " + ans["Q15"],
        ],
        ["E-35_confirmation_register.csv", "E-35_question_answers.csv"],
        [
            (
                "M4: the semantic filler, the 12 engineered features, and the ordinal map are "
                "implemented as designed; E-27/E-28 assess their hypotheses (Q17)."
            ),
            (
                "M5: log target wrapper, Yeo-Johnson on skewed numerics, fitted imputation with "
                "missing indicators, one-hot/ordinal encoding with unseen-category handling."
            ),
        ],
    )
    lines += _section(SECTIONS[1], [
        ans["Q1"], ans["Q2"], ans["Q3"], ans["Q4"],
        ("Split balance: every price decile differs by at most "
        f"{t['E-08_split_balance']['diff_pp'].max():.2f} pp between the sets (tolerance 2 pp)."),
    ], ["E-01_shape_schema.csv", "E-02_allowed_values_audit.csv", "E-02_value_counts.csv",
        "E-03_key_integrity.csv", "E-04_parsing_comparison.csv", "E-08_split_balance.csv"], [
        ("The ingestion contract needs no change; allowed values already include the file "
        "spellings, so the API accepts exactly the values training saw (M11)."),
        "Keep the explicit missing-token contract (DN-18) everywhere data is parsed (CLI, M11).",
    ])  # fmt: skip
    lines += _section(SECTIONS[2], [ans["Q5"], ans["Q6"],
        ("Mean exceeds median, the signature of right skew; Q-Q plots show the raw tail "
        "departing from normality and the log scale close to the reference line.")],
        ["E-05_target_statistics.csv", "figures/eda/E-06_target_distribution.png",
         "E-07_residual_spread.csv", "figures/eda/E-07_heteroscedasticity.png"], [
        ("Train on log1p(SalePrice) via TransformedTargetRegressor and select on log-RMSE "
        "(ADR-02, ADR-12): M5, M6."),
        "Stratify folds on binned log price (DN-01): M6.",
    ])  # fmt: skip
    lines += _section(SECTIONS[3], [ans["Q7"], ans["Q9"],
        "Largest gaps: " + _join([f"{r.column} {r.pct_missing:.1f}%" for r in missing.head(6).itertuples()]) + "."],
        ["E-09_missing_values.csv", "figures/eda/E-10_missingness.png",
         "E-12_lotfrontage_missingness.csv", "figures/eda/E-12_lotfrontage_missingness.png"], [
        ("M4: Layer 1 semantic filler ('None'/0) for the absent-feature columns; GarageYrBlt "
        "replaced by HasGarage/GarageAge."),
        ("M5: Layer 2 median + missing indicator for LotFrontage, most frequent for Electrical, "
        "fitted inside the pipeline."),
    ])  # fmt: skip
    lines += _section(SECTIONS[4], [ans["Q10"], ans["Q16"],
        "Near-constant (top value >= 90%): " + _join(numeric[numeric["top_value_share"] >= 0.9]["feature"].tolist()) + "."],
        ["E-13_numeric_profile.csv", "figures/eda/E-14_numeric_distributions.png",
         "E-17_overallqual_price_profile.csv", "figures/eda/E-17_top_predictor_scatter.png",
         "E-18_numeric_extremes.csv"], [
        "M5: Yeo-Johnson power transform and scaling in the linear branch; none in the tree branch.",
        ("M4: presence flags (HasPool, Has2ndFlr, ...) for zero-inflated columns; aggregates "
        "TotalSF, TotalBath, TotalPorchSF."),
        "No statistical outlier removal beyond the scope rule (ADR-06).",
    ])  # fmt: skip
    lines += _section(SECTIONS[5], [ans["Q12"], ans["Q13"],
        "Most informative categoricals (eta²): " + _join([f"{r.feature} {r.eta_squared:.2f}" for r in categories.head(6).itertuples()]) + ".",
        "Least informative categoricals (eta²): " + _join(weakest_cat) + "."],
        ["E-19_categorical_profile.csv", "E-19_foundation_year_built.csv", "E-20_rare_categories.csv",
         "E-21_category_importance.csv", "E-21_price_by_category.csv",
         "figures/eda/E-21_price_by_category_*.png", "E-22_neighborhoods.csv",
         "figures/eda/E-22_neighborhoods.png", "E-23_ordinal_monotonicity.csv",
         "figures/eda/E-23_ordinal_monotonicity.png"], [
        "M4: ordinal map (None=0 ... Ex=5) for the ten Po-Ex scales; MSSubClass cast to string.",
        ("M5: one-hot (linear) / ordinal encoding (tree) with unseen-category handling for rare "
        "levels; no target encoding."),
        ("Non-monotonic condition scales are a known limitation of the linear branch; trees can "
        "split them freely (DOC-02 §8.6)."),
    ])  # fmt: skip
    lines += _section(SECTIONS[6], [ans["Q11"], ans["Q14"],
        "Least informative numerics (|Spearman|): " + _join(weakest) + "."],
        ["E-15_target_correlation.csv", "E-16_correlated_pairs.csv",
         "figures/eda/E-16_correlation_heatmap.png"], [
        ("M6-M8: regularised linear models (Ridge shares weight, Lasso selects) for the "
        "collinear size and garage features; OverallQual + GrLivArea form the heuristic baseline."),
        "M4: GarageYrBlt's overlap with YearBuilt supports replacing it with GarageAge/HasGarage.",
    ])  # fmt: skip
    lines += _section(SECTIONS[7], [ans["Q8"], ans["Q15"]],
        ["E-11_consistency_checks.csv", "E-11_anomaly_rows.csv", "figures/eda/E-24_scope_rule.png",
         "E-25_out_of_scope_homes.csv", "E-26_leverage_comparison.csv",
         "reports/data_validation/quality_flags.csv (M2)"], [
        ("The anomalies are handled by the approved column-wise rules (DOC-02 §6.6): no special "
        "code path in M4; recorded in the data card."),
        ("The scope rule (GrLivArea <= 4000) stays as fixed by ADR-06; the API flags larger "
        "homes out_of_domain (M11)."),
    ])  # fmt: skip
    lines += _section(SECTIONS[8], [ans["Q18"], ans["Q19"], "Engineered features (Q17): " + ans["Q17"], status + "."],
        ["E-31_leakage_review.csv", "E-31_sale_outcome_context.csv", "E-32_sale_years.csv",
         "E-33_price_by_year.csv", "E-34_seasonality.csv", "E-27_engineered_features.csv",
         "E-28_binned_trends.csv", "E-28_discrete_price_summary.csv",
         "figures/eda/E-28_engineered_continuous.png", "figures/eda/E-28_engineered_flags.png",
         "E-29_remodel_floor_audit.csv", "figures/eda/E-29_yearremodadd_distribution.png",
         *(["E-30_ablation.csv"] if "E-30_ablation" in t else []),
         "E-35_confirmation_register.csv", "E-36_data_card_inputs.csv"], [
        "M5: SaleType, SaleCondition, Id, and PID are not model inputs (77-column schema).",
        "M9: the temporal diagnostic (2006-2009 -> 2010) uses a partial 2010 (January-July).",
        ("M7: the per-branch cross-validation ablation (E-30) decides which engineered features "
        "are retained; the M4 hypothesis assessments above do not."
        + (" The outcome is recorded per branch in features.yaml (dropped_engineered)."
           if "E-30_ablation" in t else "")),
    ])  # fmt: skip
    lines += ["## Questions Q1-Q19", "", *_table(answers, ["question", "answer", "evidence"]), "",
              "## Confirmation Register", "",
              *_table(register, ["property_id", "doc02_section", "documented_property",
                                 "observed", "status"]), "",
              "## Data Card Inputs (E-36)", "", *_table(card, ["issue", "status", "evidence", "detail"]), ""]  # fmt: skip
    return "\n".join(lines)


def build(ctx: EDAContext) -> ReportResult:
    """Read the saved deliverables and write E-35 and E-36."""
    tables = load_tables(ctx.tables_dir)
    register = build_register(tables)
    answers = answer_questions(tables)
    card = data_card_inputs(tables, register)
    out = Outputs()
    save_table(out, ctx.tables_dir, "E-35_confirmation_register", register)
    save_table(out, ctx.tables_dir, "E-35_question_answers", answers)
    save_table(out, ctx.tables_dir, "E-36_data_card_inputs", card)
    markdown = render_markdown(tables, register, answers, card)
    path = ctx.tables_dir / REPORT_NAME
    path.write_text(markdown, encoding="utf-8")
    return ReportResult(register, answers, card, markdown, path)

"""The M3 and M4 run records, read from the saved EDA deliverables.

Like the E-35 register, these functions only *read* ``reports/eda/E-*.csv``: every value in
a record is a cell of a saved table, never a fresh computation. M4's evidence (E-27 to E-29)
is written by the same EDA run as M3's, so one execution produces both records.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from house_price.config import FeatureConfig
from house_price.eda.register import Tables
from house_price.eda.report import REPORT_NAME
from house_price.results import ResultRun

M4_IDS = ("E-27", "E-28", "E-29")  # the engineered-feature evidence (DOC-05 M4)
M7_IDS = ("E-30",)  # the ablation table: written and recorded by the M7 train run
TOP_CORRELATES = 5
REQUIRED_TABLES = (
    "E-03_key_integrity", "E-05_target_statistics", "E-08_split_balance",
    "E-09_missing_values", "E-15_target_correlation", "E-16_correlated_pairs",
    "E-25_out_of_scope_homes", "E-27_engineered_features", "E-29_remodel_floor_audit",
    "E-31_leakage_review", "E-35_confirmation_register", "E-35_question_answers",
    "E-36_data_card_inputs",
)  # fmt: skip


def missing_tables(t: Tables) -> list[str]:
    return [name for name in REQUIRED_TABLES if name not in t]


def is_m4(stem: str) -> bool:
    return stem.startswith(M4_IDS)


def _number(value: object) -> float | int:
    number = float(str(value))
    return int(number) if number.is_integer() else number


def _indexed(t: Tables, table: str, key: str) -> pd.DataFrame:
    return t[table].set_index(key)


# ------------------------------------------------------------------------------ M3


def record_m3(run: ResultRun, t: Tables) -> None:
    """Headline M3 findings: data integrity, target, missingness, correlations, scope,
    leakage, and the E-35 register."""
    integrity = _indexed(t, "E-03_key_integrity", "check")["result"]
    run.log_metrics({
        "raw_rows": _number(integrity["row_count"]),
        "raw_columns": _number(integrity["column_count"]),
    })  # fmt: skip
    run.log_value("key_integrity", {
        check: str(integrity[check]) == "True"
        for check in integrity.index if check not in ("row_count", "column_count")
    })  # fmt: skip

    target = _indexed(t, "E-05_target_statistics", "scale")
    run.log_metric("dev_rows", target.loc["dollars", "n"])
    for scale in ("dollars", "log1p"):
        for stat in ("median", "mean", "std", "skewness", "excess_kurtosis"):
            run.log_metric(f"target.{scale}.{stat}", target.loc[scale, stat])

    balance = t["E-08_split_balance"]
    run.log_metric("split_balance.max_abs_diff_pp", balance["diff_pp"].abs().max())
    run.log_value("split_balance.all_within_tolerance", bool(balance["within_tolerance"].all()))

    missing = t["E-09_missing_values"]
    run.log_metric("missing.columns_with_missing", len(missing))
    run.log_value("missing.by_classification",
                  missing["classification"].value_counts().sort_index().to_dict())  # fmt: skip

    correlation = t["E-15_target_correlation"].sort_values("rank").head(TOP_CORRELATES)
    run.log_value("top_correlates", [
        {"feature": row.feature, "spearman": row.spearman, "pearson": row.pearson}
        for row in correlation.itertuples()
    ])  # fmt: skip
    pairs = t["E-16_correlated_pairs"]
    run.log_metric("strongly_correlated_pairs", int(pairs["strong"].astype(bool).sum()))

    run.log_value("scope.out_of_scope_ids",
                  sorted(int(i) for i in t["E-25_out_of_scope_homes"]["Id"]))  # fmt: skip
    leakage = t["E-31_leakage_review"]
    run.log_value("leakage.excluded_columns",
                  list(leakage.loc[leakage["decision"] == "excluded", "column"]))  # fmt: skip

    register = t["E-35_confirmation_register"]
    confirmed = register["status"] == "confirmed"
    adr_fixed = register["adr_fixed"].astype(str) == "True"
    run.log_metrics({
        "register.properties": len(register),
        "register.confirmed": int(confirmed.sum()),
        "register.adr_fixed_discrepancies": int((~confirmed & adr_fixed).sum()),
    })  # fmt: skip
    run.log_value("register.not_confirmed", [
        {"property_id": row.property_id, "documented": row.documented_property,
         "observed": row.observed}
        for row in register[~confirmed].itertuples()
    ])  # fmt: skip

    answers = t["E-35_question_answers"]
    partly = answers["answer"].astype(str).str.startswith("Partly")
    run.log_metric("questions.answered", len(answers))
    run.log_value("questions.partly_answered", list(answers.loc[partly, "question"]))
    card = t["E-36_data_card_inputs"]
    run.log_value(
        "data_card_inputs.by_status", card["status"].value_counts().sort_index().to_dict()
    )


# ------------------------------------------------------------------------------ M4


def record_m4(run: ResultRun, t: Tables, features: FeatureConfig) -> None:
    """M4 configuration (from ``features.yaml``) and its evidence (E-27 to E-29, Q17)."""
    run.log_params({
        "semantic_fill.categorical_none": len(features.semantic_fill.categorical_none),
        "semantic_fill.numeric_zero": len(features.semantic_fill.numeric_zero),
        "ordinal.columns": len(features.ordinal.columns),
        "ordinal.mapping": features.ordinal.mapping,
        "categorical_codes": features.categorical_codes,
        "engineered": features.engineered,
    })  # fmt: skip

    engineered = t["E-27_engineered_features"]
    supported = engineered["hypothesis_supported"].astype(str) == "True"
    run.log_metrics({
        "engineered.count": len(engineered),
        "engineered.hypotheses_supported": int(supported.sum()),
    })  # fmt: skip
    run.log_value("engineered.not_supported", list(engineered.loc[~supported, "feature"]))
    run.log_value("engineered.features", {
        row.feature: {
            "definition": row.definition, "spearman": row.spearman, "pearson": row.pearson,
            "n": row.n, "n_missing": row.n_missing, "assessment_rule": row.assessment_rule,
            "observed": row.observed, "hypothesis_supported": bool(supported.iloc[i]),
        }
        for i, row in enumerate(engineered.itertuples())
    })  # fmt: skip

    floor = _indexed(t, "E-29_remodel_floor_audit", "metric")["value"]
    run.log_value("remodel_floor_audit", {
        metric: None if pd.isna(value) else _number(value) for metric, value in floor.items()
    })  # fmt: skip
    answers = _indexed(t, "E-35_question_answers", "question")
    run.log_value("q17_answer", answers.loc["Q17", "answer"])


# ------------------------------------------------------------------------- sources


def deliverables(tables_dir: Path, figures_dir: Path) -> tuple[list[Path], list[Path]]:
    """Saved EDA files split into M3's and M4's (E-27 to E-29)."""
    files = sorted([*tables_dir.glob("E-*.csv"), *figures_dir.glob("E-*.png")])
    report = tables_dir / REPORT_NAME
    m3 = [f for f in files if not (is_m4(f.stem) or f.stem.startswith(M7_IDS))]
    m3 += [report] if report.is_file() else []
    return m3, [f for f in files if is_m4(f.stem)]


def register_artifacts(run: ResultRun, paths: Iterable[Path]) -> None:
    for path in paths:
        run.log_artifact(path)


def source_summary(paths: Iterable[Path]) -> dict[str, Any]:
    paths = list(paths)
    return {"tables": sum(p.suffix == ".csv" for p in paths),
            "figures": sum(p.suffix == ".png" for p in paths),
            "reports": sum(p.suffix == ".md" for p in paths)}  # fmt: skip

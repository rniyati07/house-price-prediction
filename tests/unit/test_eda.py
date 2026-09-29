"""M3 EDA package: execution, deliverables, report structure, reproducibility, anomalies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from house_price.data import split
from house_price.eda import EDAContext, report
from house_price.eda.context import EDAError
from house_price.eda.register import PROPERTIES
from house_price.eda.runner import EXIT_ADR_DISCREPANCY, main, run_analyses
from tests.conftest import CONFIG_DIR, REPO_ROOT, SampleEnv, make_env

# Every DOC-02 deliverable produced by M3 and M4 (E-30 belongs to M7).
REQUIRED_IDS = [f"E-{n:02d}" for n in [*range(1, 30), *range(31, 37)]]


@dataclass(frozen=True)
class EDARun:
    env: SampleEnv
    ctx: EDAContext
    result: report.ReportResult


def _context(env: SampleEnv, out: Path) -> EDAContext:
    return EDAContext.load(env.config_dir, env.root, out / "tables", out / "figures")


def _run(env: SampleEnv, out: Path) -> EDARun:
    ctx = _context(env, out)
    run_analyses(ctx)
    return EDARun(env, ctx, report.build(ctx))


@pytest.fixture(scope="module")
def sample_env_with_split(tmp_path_factory: pytest.TempPathFactory) -> SampleEnv:
    env = make_env(tmp_path_factory.mktemp("eda_env"))
    assert split.main(env.cli_args()) == 0
    return env


@pytest.fixture(scope="module")
def eda_run(sample_env_with_split: SampleEnv, tmp_path_factory: pytest.TempPathFactory) -> EDARun:
    return _run(sample_env_with_split, tmp_path_factory.mktemp("eda_out"))


def _files(run: EDARun) -> list[Path]:
    return sorted([*run.ctx.tables_dir.iterdir(), *run.ctx.figures_dir.iterdir()])


# ------------------------------------------------------------------- execution


def test_eda_requires_the_persisted_split(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    with pytest.raises(EDAError, match="make split"):
        EDAContext.load(env.config_dir, env.root)


def test_raw_features_exclude_the_target(eda_run: EDARun) -> None:
    assert eda_run.ctx.target not in eda_run.ctx.raw_features.columns
    assert len(eda_run.ctx.dev) == eda_run.ctx.manifest.n_dev


def test_every_deliverable_is_generated(eda_run: EDARun) -> None:
    """M3-1: E-01 to E-26 and E-31 to E-34 exist (plus E-35 and E-36)."""
    names = [path.name for path in _files(eda_run)]
    missing = [i for i in REQUIRED_IDS if not any(n.startswith(f"{i}_") for n in names)]
    assert not missing
    assert all(re.match(r"E-\d{2}_", n) for n in names), "every file starts with its ID"


def test_visualizations_are_generated(eda_run: EDARun) -> None:
    figures = sorted(p.name for p in eda_run.ctx.figures_dir.glob("*.png"))
    for stem in ("E-06_target_distribution", "E-07_heteroscedasticity", "E-10_missingness",
                 "E-14_numeric_distributions", "E-16_correlation_heatmap",
                 "E-17_top_predictor_scatter", "E-21_price_by_category_1", "E-22_neighborhoods",
                 "E-23_ordinal_monotonicity", "E-24_scope_rule", "E-32_sale_years",
                 "E-33_price_by_year", "E-34_seasonality"):  # fmt: skip
        assert f"{stem}.png" in figures
    assert all((eda_run.ctx.figures_dir / f).stat().st_size > 1000 for f in figures)


# --------------------------------------------------------------------- report


def test_report_has_the_nine_sections(eda_run: EDARun) -> None:
    markdown = eda_run.result.path.read_text(encoding="utf-8")
    positions = [markdown.index(f"## {title}") for title in report.SECTIONS]
    assert positions == sorted(positions)
    for title in report.SECTIONS:
        body = markdown.split(f"## {title}")[1].split("\n## ")[0]
        for part in ("**Observations**", "**Evidence**", "**Implications for later milestones**"):
            assert part in body, (title, part)
    for heading in (
        "## Questions Q1-Q19",
        "## Confirmation Register",
        "## Data Card Inputs (E-36)",
    ):
        assert heading in markdown


def test_register_covers_every_documented_property(eda_run: EDARun) -> None:
    """M3-4: every property has a confirmed / not confirmed status and an observation."""
    register = eda_run.result.register
    assert list(register["property_id"]) == [p.pid for p in PROPERTIES]
    assert set(register["status"]) <= {"confirmed", "not confirmed"}
    assert register["observed"].astype(str).str.len().gt(0).all()
    saved = pd.read_csv(eda_run.ctx.tables_dir / "E-35_confirmation_register.csv")
    assert len(saved) == len(PROPERTIES)


def test_questions_answered_except_q17(eda_run: EDARun) -> None:
    answers = eda_run.result.answers.set_index("question")
    assert list(answers.index) == [f"Q{n}" for n in range(1, 20)]
    q17 = answers.loc["Q17", "answer"]
    assert q17.startswith("Partly answered in M4")  # M4 evidence, not a retention decision
    assert "E-30" in q17 and "M7" in q17
    assert all(answers.loc[q, "answer"] for q in answers.index)


def test_discrepancies_feed_the_data_card_inputs(eda_run: EDARun) -> None:
    """M3-5: every not-confirmed property is listed in E-36."""
    card = eda_run.result.data_card_inputs
    for pid in eda_run.result.register.query("status == 'not confirmed'")["property_id"]:
        assert card["issue"].str.contains(pid).any()


def test_adr_fixed_discrepancy_stops_the_run(
    sample_env_with_split: SampleEnv, tmp_path: Path
) -> None:
    """DOC-05 M3 task 4: the fixture's 100 in-scope rows contradict the ADR-fixed 2,925."""
    args = [*sample_env_with_split.cli_args(), "--tables-dir", str(tmp_path / "t"),
            "--figures-dir", str(tmp_path / "f")]  # fmt: skip
    assert main(args) == EXIT_ADR_DISCREPANCY


# ------------------------------------------------------------ reproducibility


def test_outputs_are_reproducible(
    eda_run: EDARun, sample_env_with_split: SampleEnv, tmp_path: Path
) -> None:
    second = _run(sample_env_with_split, tmp_path)
    first_files = {p.name: p.read_bytes() for p in _files(eda_run)}
    second_files = {p.name: p.read_bytes() for p in _files(second)}
    assert first_files.keys() == second_files.keys()
    different = [name for name in first_files if first_files[name] != second_files[name]]
    assert not different


# ------------------------------------------------------------------ anomalies


def test_anomaly_review_finds_the_documented_rows(eda_run: EDARun) -> None:
    rows = pd.read_csv(eda_run.ctx.tables_dir / "E-11_anomaly_rows.csv")
    by_rule = rows.groupby("rule")["Id"].apply(sorted).to_dict()
    assert by_rule == {
        "basement_areas_unrecorded": [1342],
        "basement_exposure_missing_with_basement": [67],
        "basement_fintype2_missing_with_area": [445],
        "garage_partially_recorded": [1357, 2237],
    }
    assert rows["related_values"].str.contains("=NA").all()


def test_missing_values_are_classified(eda_run: EDARun) -> None:
    table = pd.read_csv(eda_run.ctx.tables_dir / "E-09_missing_values.csv").set_index("column")
    assert table.loc["LotFrontage", "classification"] == "unknown"
    assert table.loc["GarageFinish", "classification"] == "mixed"
    assert table.loc["PoolQC", "classification"] == "absent"
    assert (table["n_absent"] + table["n_unknown"] == table["n_missing"]).all()


def test_scope_review_lists_the_out_of_scope_homes(eda_run: EDARun) -> None:
    table = pd.read_csv(eda_run.ctx.tables_dir / "E-25_out_of_scope_homes.csv")
    assert sorted(table["Id"]) == [1499, 1768]


# -------------------------------------------------------------- real dataset


@pytest.mark.data
def test_real_eda_confirms_adr_fixed_numbers(tmp_path: Path) -> None:
    ctx = EDAContext.load(CONFIG_DIR, REPO_ROOT, tmp_path / "tables", tmp_path / "figures")
    run_analyses(ctx)
    result = report.build(ctx)
    register = result.register.set_index("property_id")
    assert register.loc["P-04", "status"] == "confirmed"
    assert register.loc["P-05", "status"] == "confirmed"
    assert result.adr_fixed_discrepancies.empty

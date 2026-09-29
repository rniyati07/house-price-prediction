"""M3 EDA package: execution, deliverables, report structure, reproducibility, anomalies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from house_price.config import load_feature_config
from house_price.data import split
from house_price.data.load import sha256_file
from house_price.eda import EDAContext, report
from house_price.eda.context import EDAError
from house_price.eda.register import PROPERTIES
from house_price.eda.runner import EXIT_ADR_DISCREPANCY, main, run_analyses
from house_price.results import read_index, read_result
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


# ---------------------------------------------------------------- run records


@dataclass(frozen=True)
class RecordedRun:
    env: SampleEnv
    out: Path
    code: int
    m3: dict[str, Any]
    m4: dict[str, Any]


def _records(results: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    index = read_index(results)
    latest = {line["milestone"]: line for line in index}
    return (read_result(results / latest["M3"]["result"]),
            read_result(results / latest["M4"]["result"]))  # fmt: skip


@pytest.fixture(scope="module")
def recorded(
    sample_env_with_split: SampleEnv, tmp_path_factory: pytest.TempPathFactory
) -> RecordedRun:
    out = tmp_path_factory.mktemp("recorded")
    code = main([*sample_env_with_split.cli_args(), "--tables-dir", str(out / "tables"),
                 "--figures-dir", str(out / "figures"), "--results-dir", str(out / "results")])  # fmt: skip
    return RecordedRun(sample_env_with_split, out, code, *_records(out / "results"))


def test_full_run_writes_m3_and_m4_records(recorded: RecordedRun) -> None:
    m3, m4 = recorded.m3, recorded.m4
    assert recorded.code == EXIT_ADR_DISCREPANCY  # the 100-row fixture contradicts 2,925
    assert m3["status"] == "failed" and "P-04" in m3["error"]["message"]
    assert m4["status"] == "succeeded"
    assert m3["findings"]["execution.shared_with"] == {"M4": m4["run_id"]}
    assert m3["params"]["mode"] == m4["params"]["mode"] == "full"
    assert m3["command"]["entry_point"] == "python -m house_price.eda"


def test_m3_record_matches_the_saved_tables(recorded: RecordedRun) -> None:
    tables = recorded.out / "tables"
    m3 = recorded.m3
    register = pd.read_csv(tables / "E-35_confirmation_register.csv")
    target = pd.read_csv(tables / "E-05_target_statistics.csv").set_index("scale")
    assert m3["metrics"]["register.properties"] == len(register)
    assert m3["metrics"]["register.confirmed"] == (register["status"] == "confirmed").sum()
    assert m3["metrics"]["dev_rows"] == target.loc["dollars", "n"]
    assert m3["metrics"]["target.log1p.std"] == target.loc["log1p", "std"]
    assert m3["findings"]["scope.out_of_scope_ids"] == [1499, 1768]
    leakage = pd.read_csv(tables / "E-31_leakage_review.csv")
    excluded = leakage.loc[leakage["decision"] == "excluded", "column"]
    assert m3["findings"]["leakage.excluded_columns"] == list(excluded)
    assert "Q17" in m3["findings"]["questions.partly_answered"]


def test_m4_record_matches_the_saved_tables(recorded: RecordedRun) -> None:
    tables = recorded.out / "tables"
    m4 = recorded.m4
    e27 = pd.read_csv(tables / "E-27_engineered_features.csv").set_index("feature")
    features = load_feature_config(CONFIG_DIR)
    recorded_features = m4["findings"]["engineered.features"]
    assert list(recorded_features) == list(e27.index) == features.engineered
    for name, values in recorded_features.items():
        assert values["spearman"] == e27.loc[name, "spearman"]
        assert values["hypothesis_supported"] == bool(e27.loc[name, "hypothesis_supported"])
    assert m4["metrics"]["engineered.hypotheses_supported"] == e27["hypothesis_supported"].sum()
    assert m4["params"]["engineered"] == features.engineered
    assert m4["params"]["ordinal.mapping"] == features.ordinal.mapping
    q17 = pd.read_csv(tables / "E-35_question_answers.csv").set_index("question")
    assert m4["findings"]["q17_answer"] == q17.loc["Q17", "answer"]


def test_records_register_deliverables_by_hash(recorded: RecordedRun) -> None:
    root = recorded.env.root
    for record in (recorded.m3, recorded.m4):
        assert record["artifacts"]
        for artifact in record["artifacts"]:
            path = Path(artifact["path"])
            path = path if path.is_absolute() else root / path
            assert sha256_file(path) == artifact["sha256"]
    m4_names = {Path(a["path"]).name for a in recorded.m4["artifacts"]}
    assert "E-27_engineered_features.csv" in m4_names
    assert "E-28_engineered_flags.png" in m4_names
    assert not any(Path(a["path"]).name.startswith("E-27") for a in recorded.m3["artifacts"])


def test_record_only_reads_saved_deliverables_without_rerunning(recorded: RecordedRun) -> None:
    files = sorted([*(recorded.out / "tables").iterdir(), *(recorded.out / "figures").iterdir()])
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files}
    results = recorded.out / "results_record_only"
    code = main([*recorded.env.cli_args(), "--tables-dir", str(recorded.out / "tables"),
                 "--figures-dir", str(recorded.out / "figures"),
                 "--results-dir", str(results), "--record-only"])  # fmt: skip
    assert code == 0
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files} == before  # untouched
    m3, m4 = _records(results)
    assert m3["status"] == m4["status"] == "succeeded"
    assert m3["params"]["mode"] == "record-only"
    assert m3["metrics"] == recorded.m3["metrics"]
    assert m4["findings"]["engineered.features"] == recorded.m4["findings"]["engineered.features"]


def test_record_only_without_saved_tables_is_a_recorded_failure(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    code = main([*env.cli_args(), "--tables-dir", str(tmp_path / "none"), "--record-only"])
    assert code == 1
    index = read_index(tmp_path / "results")
    assert [line["status"] for line in index] == ["failed", "failed"]
    assert "missing" in read_result(tmp_path / "results" / index[0]["result"])["error"]["message"]


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

"""Quality gates (DOC-03 §12.4; DOC-05 M9-4; AC-039 to AC-041 logic, IN-11): each threshold
passes and fails at its boundary, on synthetic metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from house_price.evaluation import gates

RECORD = {"selected": {"mean": 0.110},
          "baseline_margins": {"dummy_median": {"baseline_mean": 0.40, "baseline_se": 0.002},
                               "linear_2feat": {"baseline_mean": 0.19, "baseline_se": 0.002}}}  # fmt: skip
REFERENCE = {"dummy_median": {"log_rmse": 0.40}, "linear_2feat": {"log_rmse": 0.19}}


def _gate(results: list[gates.GateResult], rule: str) -> gates.GateResult:
    return next(r for r in results if r.rule == rule)


@pytest.mark.parametrize(
    ("value", "passed"), [(0.12, True), (0.13, True), (0.1300001, False), (0.2, False)]
)
def test_holdout_log_rmse_boundary(value: float, passed: bool) -> None:
    results = gates.check_quality_gates({"log_rmse": value, "mape": 5.0}, REFERENCE, RECORD)
    assert _gate(results, "holdout_log_rmse").passed is passed


@pytest.mark.parametrize(("value", "passed"), [(9.99, True), (10.0, True), (10.0001, False)])
def test_holdout_mape_boundary(value: float, passed: bool) -> None:
    results = gates.check_quality_gates({"log_rmse": 0.1, "mape": value}, REFERENCE, RECORD)
    assert _gate(results, "holdout_mape").passed is passed


@pytest.mark.parametrize(
    ("selected_mean", "passed"), [(0.187, True), (0.188, False), (0.189, False)]
)
def test_cv_margin_is_strict(selected_mean: float, passed: bool) -> None:
    # linear_2feat bar = 0.19 - 0.002 = 0.188; selected must be strictly below it (IN-11)
    record = {**RECORD, "selected": {"mean": selected_mean}}
    results = gates.check_quality_gates({"log_rmse": 0.1, "mape": 5.0}, REFERENCE, record)
    assert _gate(results, "cv_margin_linear_2feat").passed is passed
    assert _gate(results, "cv_margin_dummy_median").passed


@pytest.mark.parametrize(("holdout", "passed"), [(0.189, True), (0.19, False), (0.2, False)])
def test_holdout_margin_is_strict(holdout: float, passed: bool) -> None:
    results = gates.check_quality_gates({"log_rmse": holdout, "mape": 5.0}, REFERENCE, RECORD)
    assert _gate(results, "holdout_margin_linear_2feat").passed is passed


def test_every_rule_is_recorded_with_observed_values(tmp_path: Path) -> None:
    results = gates.check_quality_gates({"log_rmse": 0.12, "mape": 8.0}, REFERENCE, RECORD)
    assert sorted(r.gate for r in results) == ["QG-10", "QG-11", "QG-12", "QG-12", "QG-12", "QG-12"]
    payload = gates.write_gates(results, tmp_path / "q.json", smoke=False, record_id="rec")
    assert payload["all_passed"] and payload["enforced"]
    stored = json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))
    assert {r["rule"] for r in stored["rules"]} >= {"holdout_log_rmse", "holdout_mape"}
    assert all("observed" in r and "threshold" in r for r in stored["rules"])


def test_smoke_gates_are_reported_not_enforced(tmp_path: Path) -> None:
    results = gates.check_quality_gates({"log_rmse": 0.5, "mape": 40.0}, REFERENCE, RECORD)
    payload = gates.write_gates(results, tmp_path / "q.json", smoke=True, record_id="rec")
    assert not payload["all_passed"] and payload["enforced"] is False

"""Project quality gates (DOC-03 §12.4, §16.1 QG-10 to QG-12; AC-039 to AC-041; IN-11).

| Rule | Condition |
|---|---|
| Holdout log-RMSE (QG-10) | selected <= 0.13 (ADR-12) |
| Holdout MAPE (QG-11) | selected <= 10% (ADR-12) |
| Clear margin, CV (QG-12) | selected mean < each baseline mean - that baseline's SE |
| Clear margin, holdout (QG-12) | selected holdout log-RMSE < each baseline's reference log-RMSE |

Every rule is recorded with its observed value and pass/fail. A failure blocks ``freeze``;
it is never answered by re-selecting on holdout results (IN-12). In smoke mode the gates are
reported but not enforced (DN-17).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

HOLDOUT_LOG_RMSE_MAX = 0.13  # ADR-12, AC-039
HOLDOUT_MAPE_MAX = 10.0  # ADR-12, AC-040 (percent)
GATES_PATH = Path("reports/evaluation/quality_gates.json")


@dataclass(frozen=True)
class GateResult:
    rule: str
    gate: str
    condition: str
    observed: float
    threshold: float
    passed: bool
    source: str


def check_quality_gates(
    holdout: Mapping[str, float],
    baseline_reference: Mapping[str, Mapping[str, float]],
    record: Mapping[str, Any],
) -> list[GateResult]:
    """Evaluate QG-10 to QG-12 from the holdout metrics, the baseline reference scores and
    the selection record's CV scores."""
    results = [
        GateResult(
            "holdout_log_rmse",
            "QG-10",
            "selected holdout log-RMSE <= 0.13",
            holdout["log_rmse"],
            HOLDOUT_LOG_RMSE_MAX,
            holdout["log_rmse"] <= HOLDOUT_LOG_RMSE_MAX,
            "ADR-12, AC-039",
        ),
        GateResult(
            "holdout_mape",
            "QG-11",
            "selected holdout MAPE <= 10%",
            holdout["mape"],
            HOLDOUT_MAPE_MAX,
            holdout["mape"] <= HOLDOUT_MAPE_MAX,
            "ADR-12, AC-040",
        ),
    ]
    selected_mean = float(record["selected"]["mean"])
    for name, margin in sorted(record["baseline_margins"].items()):
        bar = float(margin["baseline_mean"]) - float(margin["baseline_se"])
        results.append(
            GateResult(
                f"cv_margin_{name}",
                "QG-12",
                f"selected CV mean < {name} mean - {name} SE",
                selected_mean,
                bar,
                selected_mean < bar,
                "IN-11, AC-041",
            )
        )
    for name, metrics in sorted(baseline_reference.items()):
        reference = float(metrics["log_rmse"])
        results.append(
            GateResult(
                f"holdout_margin_{name}",
                "QG-12",
                f"selected holdout log-RMSE < {name} reference log-RMSE",
                holdout["log_rmse"],
                reference,
                holdout["log_rmse"] < reference,
                "IN-11, IN-12, AC-041",
            )
        )
    return results


def write_gates(
    results: list[GateResult], path: Path, *, smoke: bool, record_id: str
) -> dict[str, Any]:
    """``quality_gates.json``: every rule with observed value and pass/fail."""
    payload = {
        "selection_record_id": record_id,
        "smoke": smoke,
        "enforced": not smoke,  # DN-17: reported but not enforced in smoke mode
        "all_passed": all(r.passed for r in results),
        "rules": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload

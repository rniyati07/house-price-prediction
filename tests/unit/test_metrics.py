"""Metric formulas against hand-computed values (DOC-03 §12.1; DOC-05 M6-1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from house_price.evaluation.metrics import all_metrics, log_rmse, mae, mape, r2

# Three houses, hand-computed below.
TRUE = [100.0, 200.0, 400.0]
PRED = [110.0, 180.0, 400.0]


def test_log_rmse_by_hand() -> None:
    # log1p differences: ln(111/101), ln(181/201), 0
    expected = math.sqrt((math.log(111 / 101) ** 2 + math.log(181 / 201) ** 2 + 0.0) / 3)
    assert log_rmse(TRUE, PRED) == pytest.approx(expected, rel=1e-12)


def test_mae_by_hand() -> None:
    assert mae(TRUE, PRED) == pytest.approx((10 + 20 + 0) / 3)


def test_mape_by_hand() -> None:
    assert mape(TRUE, PRED) == pytest.approx(100 * (0.10 + 0.10 + 0.0) / 3)


def test_r2_by_hand() -> None:
    # mean 700/3; SS_tot = (400^2 + 100^2 + 500^2) / 9 = 420000/9; SS_res = 100 + 400
    assert r2(TRUE, PRED) == pytest.approx(1 - 500 / (420000 / 9), rel=1e-12)


def test_perfect_predictions() -> None:
    assert all_metrics(TRUE, TRUE) == {"log_rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 1.0}


def test_log_rmse_measures_relative_error() -> None:
    """A uniform 10% over-prediction scores about log(1.1) at any price level."""
    cheap, expensive = np.array([100_000.0]), np.array([600_000.0])
    assert log_rmse(cheap, cheap * 1.1) == pytest.approx(math.log(1.1), abs=1e-5)
    assert log_rmse(expensive, expensive * 1.1) == pytest.approx(math.log(1.1), abs=1e-5)
    assert mae(expensive, expensive * 1.1) == pytest.approx(6 * mae(cheap, cheap * 1.1))


def test_metrics_take_dollars_not_logs() -> None:
    """Passing log prices by mistake gives a very different (wrong) log-RMSE."""
    dollars_true, dollars_pred = np.array([150_000.0, 250_000.0]), np.array([165_000.0, 225_000.0])
    on_dollars = log_rmse(dollars_true, dollars_pred)
    on_logs = log_rmse(np.log1p(dollars_true), np.log1p(dollars_pred))
    assert on_dollars == pytest.approx(0.1, abs=0.01)
    assert on_logs < on_dollars / 10


@pytest.mark.parametrize(
    ("true", "pred", "message"),
    [
        ([100.0, 200.0], [100.0], "equal length"),
        ([], [], "non-empty"),
        ([0.0, 200.0], [100.0, 200.0], "positive"),
        ([100.0, np.nan], [100.0, 200.0], "finite"),
    ],
)
def test_invalid_inputs_are_rejected(true: list[float], pred: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        log_rmse(true, pred)


def test_r2_undefined_for_constant_truth() -> None:
    with pytest.raises(ValueError, match="undefined"):
        r2([100.0, 100.0], [90.0, 110.0])

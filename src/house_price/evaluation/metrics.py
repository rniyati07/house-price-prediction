"""Evaluation metrics (ADR-12, DOC-03 §12.1).

Every function takes true and predicted prices **in dollars**, which is what the pipeline's
``predict`` returns. log-RMSE moves both to the log scale itself, so it cannot be computed
on the wrong scale by accident.

=========  ==========================================  =====================================
Metric     Definition                                  Role
=========  ==========================================  =====================================
log-RMSE   sqrt(mean((log1p(pred) - log1p(true))^2))    primary: selection, gates (0.13 ~ 13%)
MAE        mean(|pred - true|), dollars                 secondary: "typically off by $X"
MAPE       100 * mean(|pred - true| / true)             secondary: average percentage miss
R^2        1 - SS_res / SS_tot, on dollars              secondary: share of variance explained
=========  ==========================================  =====================================
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _prices(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if true.shape != pred.shape or true.ndim != 1 or true.size == 0:
        raise ValueError(
            f"expected two non-empty 1-D arrays of equal length, got {true.shape} and {pred.shape}"
        )
    if not (np.isfinite(true).all() and np.isfinite(pred).all()):
        raise ValueError("prices must be finite")
    if (true <= 0).any():
        raise ValueError("true prices must be positive dollars")
    if (pred <= -1).any():
        raise ValueError("predicted prices must be greater than -1 (log1p domain)")
    return true, pred


def log_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error of ``log1p`` prices: the project's primary metric."""
    true, pred = _prices(y_true, y_pred)
    return float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error in dollars."""
    true, pred = _prices(y_true, y_pred)
    return float(np.mean(np.abs(pred - true)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error, in percent (10.0 means 10%)."""
    true, pred = _prices(y_true, y_pred)
    return float(100 * np.mean(np.abs(pred - true) / true))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination on the dollar scale."""
    true, pred = _prices(y_true, y_pred)
    total = float(np.sum((true - true.mean()) ** 2))
    if total == 0:
        raise ValueError("R^2 is undefined when all true prices are equal")
    return 1.0 - float(np.sum((true - pred) ** 2)) / total


def all_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """The four DOC-03 §12.1 metrics, keyed by their MLflow names (§14.5)."""
    return {
        "log_rmse": log_rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }

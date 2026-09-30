"""Temporal diagnostic (FR-035, AC-043, DOC-03 §13, DN-10).

The **development set** is split by sale year: training part ``YrSold`` 2006 to 2009, test
part ``YrSold`` 2010. The selected configuration (same hyperparameters, same feature sets) is
fitted on the training part and scored on the test part. The result is labelled "not used
for selection" and never changes the selected model. Holdout rows are never used (DN-10:
using the holdout's 2010 rows would give holdout information a second use).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from house_price.config import SchemaConfig
from house_price.data.schema import select_model_input
from house_price.evaluation.metrics import all_metrics

TRAIN_YEARS = (2006, 2007, 2008, 2009)
TEST_YEARS = (2010,)
YEAR_COLUMN = "YrSold"
LABEL = "not used for selection"


class TemporalError(ValueError):
    """The temporal split cannot be made from these rows."""


def temporal_split(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows (2006-2009) and test rows (2010) of the development set."""
    years = dev[YEAR_COLUMN]
    unknown = sorted(set(years) - set(TRAIN_YEARS) - set(TEST_YEARS))
    if unknown:
        raise TemporalError(f"sale years outside 2006-2010: {unknown}")
    train, test = dev[years.isin(TRAIN_YEARS)], dev[years.isin(TEST_YEARS)]
    if train.empty or test.empty:
        raise TemporalError("the temporal split needs both 2006-2009 and 2010 rows")
    return train, test


@dataclass(frozen=True)
class TemporalResult:
    metrics: dict[str, float]  # log_rmse, mae, mape, r2
    mean_signed_log_error: float
    n_train: int
    n_test: int
    train_ids: list[int]
    test_ids: list[int]
    label: str = LABEL

    def as_record(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "mean_signed_log_error": self.mean_signed_log_error,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_years": list(TRAIN_YEARS),
            "test_years": list(TEST_YEARS),
            "label": self.label,
        }


def run_temporal(
    build: Callable[[], Any], dev: pd.DataFrame, schema: SchemaConfig
) -> TemporalResult:
    """Fit ``build()`` (the selected configuration, unfitted) on 2006-2009 development rows
    and score it on the 2010 development rows."""
    train, test = temporal_split(dev)
    model = build()
    model.fit(select_model_input(train, schema), train[schema.target])
    predicted = np.asarray(model.predict(select_model_input(test, schema)), dtype="float64")
    actual = test[schema.target].to_numpy(dtype="float64")
    return TemporalResult(
        metrics=all_metrics(actual, predicted),
        mean_signed_log_error=float(np.mean(np.log1p(predicted) - np.log1p(actual))),
        n_train=len(train),
        n_test=len(test),
        train_ids=[int(i) for i in train[schema.id_column]],
        test_ids=[int(i) for i in test[schema.id_column]],
    )

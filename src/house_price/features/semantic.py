"""Layer 1 of the missing-value strategy: the stateless semantic filler (ADR-05, FR-011).

The data dictionary defines ``NA`` in some columns as "this house does not have the
feature" (no pool, no garage, no basement...). That is information, not ignorance, so it
is written down explicitly: categorical columns get the category ``"None"`` and the
related numeric columns get ``0`` (DOC-03 §6.2).

The rule is applied **by column** and learns nothing from data. A missing value in one of
these columns is filled the same way whatever its cause, including the documented anomaly
rows where the feature actually exists (DOC-02 §6.6). Genuinely unknown values in other
columns (``LotFrontage``, ``Electrical``) are left missing for the fitted imputation in the
pipeline (M5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from house_price.config import FeatureConfig

ABSENT_CATEGORY = "None"
ABSENT_NUMBER = 0


class SemanticNAFiller(TransformerMixin, BaseEstimator):
    """Fill "feature absent" missing values with ``"None"`` or ``0``.

    Parameters
    ----------
    categorical_columns:
        Columns whose missing values become ``"None"``.
    numeric_columns:
        Columns whose missing values become ``0``.

    The transformer is stateless: ``fit`` only records the input column names (for
    ``get_feature_names_out``), and ``transform`` never depends on what ``fit`` saw.
    The input frame is never modified; a copy is returned with the same index and columns.
    """

    def __init__(self, categorical_columns: Sequence[str], numeric_columns: Sequence[str]):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns

    @classmethod
    def from_config(cls, config: FeatureConfig) -> SemanticNAFiller:
        """Build the filler from ``features.yaml``."""
        return cls(
            categorical_columns=tuple(config.semantic_fill.categorical_none),
            numeric_columns=tuple(config.semantic_fill.numeric_zero),
        )

    def _check_columns(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("SemanticNAFiller expects a pandas DataFrame")
        required = [*self.categorical_columns, *self.numeric_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"SemanticNAFiller: input is missing columns {missing}")

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record the input column names; learn nothing from the values."""
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the absent-feature missing values filled."""
        self._check_columns(X)
        frame = X.copy()
        for column in self.categorical_columns:
            frame[column] = frame[column].fillna(ABSENT_CATEGORY)
        for column in self.numeric_columns:
            frame[column] = frame[column].fillna(ABSENT_NUMBER)
        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Output columns equal input columns: the filler adds and removes nothing."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.feature_names_in_, dtype=object)

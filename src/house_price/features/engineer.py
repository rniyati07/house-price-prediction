"""The stateless feature engineer (ADR-05, ADR-07, FR-012, FR-013, DOC-03 §6.4).

It runs after :class:`~house_price.features.semantic.SemanticNAFiller`, so basement and
garage numbers are already ``0`` when the feature is absent. It does three things:

1. **Engineered features**: the 12 approved formulas, computed from columns of the same row.
2. **Ordinal map**: the ten Po-Ex quality/condition scales become integers
   (``None`` = 0 ... ``Ex`` = 5); a value outside the map becomes missing, for the
   fitted imputation in the pipeline (M5).
3. **Categorical codes**: ``MSSubClass`` is cast from an integer code to text, so later
   steps treat it as a category, not a quantity (DN-12).

Every input column is passed through (``GarageYrBlt`` included: it is dropped later by the
column transformer, DOC-03 §7.5). Nothing is learned from data, nothing is clipped or
repaired: a missing input gives a missing output, and a negative house or remodel age stays
negative. A garage build year later than the valuation year (``GarageYrBlt > YrSold``, such
as the recorded 2207) is impossible and is treated as unknown for ``GarageAge``, which is
then missing like a garage without a recorded year (DOC-03 §6.4).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from house_price.config import FeatureConfig


def _flag(condition: pd.Series) -> pd.Series:
    """Boolean condition as a 0/1 integer column (missing inputs count as False)."""
    return condition.fillna(False).astype("int64")


def _has_garage(frame: pd.DataFrame) -> pd.Series:
    """A garage exists when ``GarageType`` is recorded and is not ``"None"`` (DOC-03 §6.4)."""
    garage_type = frame["GarageType"]
    return _flag(garage_type.notna() & (garage_type != "None"))


def _garage_age(frame: pd.DataFrame) -> pd.Series:
    """``YrSold - GarageYrBlt`` with a garage, ``0`` without one.

    With a garage but no recorded build year the age stays missing: it is not invented
    here (DOC-03 §6.4, defensive behavior). A build year later than ``YrSold`` is treated
    as unknown in the same way, so the age is missing rather than negative; the recorded
    ``GarageYrBlt`` value itself is passed through unchanged.
    """
    built = frame["GarageYrBlt"].astype("float64")
    impossible = (built > frame["YrSold"]).fillna(False).astype(bool)
    age = frame["YrSold"] - built.where(~impossible)
    return age.astype("float64").where(_has_garage(frame) == 1, 0.0)


# Name -> (columns used, formula). Every formula reads only its own row.
FORMULAS: dict[str, tuple[tuple[str, ...], Callable[[pd.DataFrame], pd.Series]]] = {
    "TotalSF": (
        ("TotalBsmtSF", "1stFlrSF", "2ndFlrSF"),
        lambda f: f["TotalBsmtSF"] + f["1stFlrSF"] + f["2ndFlrSF"],
    ),
    "TotalBath": (
        ("FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"),
        lambda f: f["FullBath"] + 0.5 * f["HalfBath"] + f["BsmtFullBath"] + 0.5 * f["BsmtHalfBath"],
    ),
    "HouseAge": (("YrSold", "YearBuilt"), lambda f: f["YrSold"] - f["YearBuilt"]),
    "RemodAge": (("YrSold", "YearRemodAdd"), lambda f: f["YrSold"] - f["YearRemodAdd"]),
    "IsRemodeled": (
        ("YearRemodAdd", "YearBuilt"),
        lambda f: _flag(f["YearRemodAdd"] != f["YearBuilt"]),
    ),
    "TotalPorchSF": (
        ("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"),
        lambda f: f["OpenPorchSF"] + f["EnclosedPorch"] + f["3SsnPorch"] + f["ScreenPorch"],
    ),
    "HasPool": (("PoolArea",), lambda f: _flag(f["PoolArea"] > 0)),
    "HasGarage": (("GarageType",), _has_garage),
    "HasBsmt": (("TotalBsmtSF",), lambda f: _flag(f["TotalBsmtSF"] > 0)),
    "HasFireplace": (("Fireplaces",), lambda f: _flag(f["Fireplaces"] > 0)),
    "Has2ndFlr": (("2ndFlrSF",), lambda f: _flag(f["2ndFlrSF"] > 0)),
    "GarageAge": (("YrSold", "GarageYrBlt", "GarageType"), _garage_age),
}


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Add the approved engineered features, apply the ordinal map, cast code columns.

    Parameters
    ----------
    engineered:
        Names of the engineered features to add; each must have a formula in ``FORMULAS``.
    ordinal_columns:
        Columns mapped with ``ordinal_mapping``.
    ordinal_mapping:
        Category -> integer code (``None`` = 0 ... ``Ex`` = 5).
    categorical_codes:
        Numeric code columns cast to text (``MSSubClass``).

    The transformer is stateless: ``fit`` only records the input column names. The output
    keeps the input's index and row order: every input column (recoded where configured)
    followed by the engineered features.
    """

    def __init__(
        self,
        engineered: Sequence[str],
        ordinal_columns: Sequence[str],
        ordinal_mapping: Mapping[str, int],
        categorical_codes: Sequence[str],
    ):
        self.engineered = engineered
        self.ordinal_columns = ordinal_columns
        self.ordinal_mapping = ordinal_mapping
        self.categorical_codes = categorical_codes

    @classmethod
    def from_config(cls, config: FeatureConfig) -> FeatureEngineer:
        """Build the engineer from ``features.yaml``."""
        return cls(
            engineered=tuple(config.engineered),
            ordinal_columns=tuple(config.ordinal.columns),
            ordinal_mapping=dict(config.ordinal.mapping),
            categorical_codes=tuple(config.categorical_codes),
        )

    def _required_columns(self) -> list[str]:
        unknown = [name for name in self.engineered if name not in FORMULAS]
        if unknown:
            raise ValueError(f"FeatureEngineer: no formula for {unknown}")
        used = [column for name in self.engineered for column in FORMULAS[name][0]]
        return list(dict.fromkeys([*used, *self.ordinal_columns, *self.categorical_codes]))

    def _check_columns(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing = [column for column in self._required_columns() if column not in frame.columns]
        if missing:
            raise ValueError(f"FeatureEngineer: input is missing columns {missing}")
        clash = [name for name in self.engineered if name in frame.columns]
        if clash:
            raise ValueError(f"FeatureEngineer: input already has columns {clash}")

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        """Record the input column names; learn nothing from the values."""
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with the engineered features added and the recodings applied."""
        self._check_columns(X)
        # Formulas read the input as given (the ordinal text is not needed by any formula).
        engineered = {name: FORMULAS[name][1](X) for name in self.engineered}
        frame = X.copy()
        for column in self.ordinal_columns:
            codes = frame[column].astype("object").map(dict(self.ordinal_mapping))
            frame[column] = codes.astype("float64")
        for column in self.categorical_codes:
            frame[column] = frame[column].astype("str")
        for name, values in engineered.items():
            frame[name] = values
        return frame

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Input columns followed by the engineered features."""
        names = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray([*names, *self.engineered], dtype=object)

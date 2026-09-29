"""Repeated stratified cross-validation on the development set (ADR-09, FR-024, FR-025,
DOC-03 §9.2, §9.3, §11.3).

**Folds (DN-01, DN-03).** Deciles of ``log1p(SalePrice)`` are computed on the development
set, and ``RepeatedStratifiedKFold(5 folds, 3 repeats, seed)`` stratifies on them. The 15
(train, validation) pairs are stored by ``Id`` in ``artifacts/cv/folds.json`` together with
the development-set hash, and every candidate is scored on exactly these folds.

**Runner.** Each fold gets a fresh ``clone`` of the candidate's unfitted pipeline, so every
fitted statistic comes from that fold's training rows only. The runner returns the 15 fold
log-RMSE scores, their mean and standard error (DN-04: ``std(ddof=1) / sqrt(15)``), and the
out-of-fold log predictions of every development ``Id`` in every repeat.

Only the development set is ever passed in; the holdout is never read here (AC-030).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RepeatedStratifiedKFold

from house_price.config import SchemaConfig
from house_price.data.schema import select_model_input
from house_price.evaluation.metrics import log_rmse, mae, mape, r2


class CVError(ValueError):
    """The fold plan does not match the development set."""


class Fold(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repeat: int
    fold: int
    train_ids: list[int]
    valid_ids: list[int]


class FoldPlan(BaseModel):
    """The shared folds of one training run (DN-03), stored as ``folds.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int
    n_splits: int
    n_repeats: int
    n_bins: int
    stratify_on: str
    dev_sha256: str
    folds: list[Fold]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json() + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, expected_dev_sha256: str) -> FoldPlan:
        """Read a saved plan; refuse one made for a different development set."""
        plan = cls.model_validate_json(path.read_text(encoding="utf-8"))
        if plan.dev_sha256 != expected_dev_sha256:
            raise CVError(f"{path} was generated for a different development set")
        return plan


def generate_folds(
    dev: pd.DataFrame,
    *,
    id_column: str,
    target: str,
    n_bins: int,
    n_splits: int,
    n_repeats: int,
    seed: int,
    dev_sha256: str,
) -> FoldPlan:
    """Repeated stratified folds on development-set deciles of ``log1p(target)``."""
    bins = pd.qcut(np.log1p(dev[target].astype("float64")), q=n_bins, labels=False)
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    ids = dev[id_column].to_numpy()
    folds = [
        Fold(
            repeat=position // n_splits + 1,
            fold=position % n_splits + 1,
            train_ids=[int(i) for i in ids[train]],
            valid_ids=[int(i) for i in ids[valid]],
        )
        for position, (train, valid) in enumerate(splitter.split(dev, bins))
    ]
    return FoldPlan(
        seed=seed, n_splits=n_splits, n_repeats=n_repeats, n_bins=n_bins,
        stratify_on=f"log1p({target}) deciles of the development set", dev_sha256=dev_sha256,
        folds=folds,
    )  # fmt: skip


def standard_error(scores: Sequence[float]) -> float:
    """DN-04: sample standard deviation of the fold scores divided by sqrt(n)."""
    values = np.asarray(scores, dtype="float64")
    if values.size < 2:
        raise ValueError("the standard error needs at least two scores")
    return float(values.std(ddof=1) / math.sqrt(values.size))


@dataclass(frozen=True)
class CVResult:
    """What one candidate's cross-validation produced."""

    fold_scores: list[float]
    mean: float
    se: float
    oof: pd.DataFrame  # columns: Id, repeat, fold, log_prediction
    secondary: dict[str, float]  # cv_mae, cv_mape, cv_r2 from the OOF dollar predictions
    duration_seconds: float


def _check_plan(dev_ids: pd.Index, plan: FoldPlan) -> None:
    expected = set(dev_ids)
    for repeat in range(1, plan.n_repeats + 1):
        valid = [i for f in plan.folds if f.repeat == repeat for i in f.valid_ids]
        if len(valid) != len(set(valid)) or set(valid) != expected:
            raise CVError(
                f"repeat {repeat} of the fold plan does not partition the development set"
            )


def run_cv(
    template: BaseEstimator, dev: pd.DataFrame, plan: FoldPlan, schema: SchemaConfig
) -> CVResult:
    """Score an unfitted pipeline on every fold of ``plan`` (DOC-03 §9.3).

    ``template`` is never fitted itself: each fold fits a fresh ``clone`` of it.
    """
    started = time.perf_counter()
    id_column, target = schema.id_column, schema.target
    by_id = dev.set_index(id_column, drop=False)
    if not by_id.index.is_unique:
        raise CVError("development Ids are not unique")
    _check_plan(by_id.index, plan)

    scores: list[float] = []
    oof_parts: list[pd.DataFrame] = []
    for fold in plan.folds:
        train, valid = by_id.loc[fold.train_ids], by_id.loc[fold.valid_ids]
        model = clone(template)
        model.fit(select_model_input(train, schema), train[target])
        prediction = np.asarray(model.predict(select_model_input(valid, schema)), dtype="float64")
        scores.append(log_rmse(valid[target], prediction))
        oof_parts.append(pd.DataFrame({
            id_column: fold.valid_ids, "repeat": fold.repeat, "fold": fold.fold,
            "log_prediction": np.log1p(prediction),
        }))  # fmt: skip
    oof = pd.concat(oof_parts, ignore_index=True)

    per_repeat: dict[str, list[float]] = {"cv_mae": [], "cv_mape": [], "cv_r2": []}
    for _, group in oof.groupby("repeat", sort=True):
        true = by_id.loc[group[id_column], target].to_numpy(dtype="float64")
        dollars = np.expm1(group["log_prediction"].to_numpy())
        per_repeat["cv_mae"].append(mae(true, dollars))
        per_repeat["cv_mape"].append(mape(true, dollars))
        per_repeat["cv_r2"].append(r2(true, dollars))
    return CVResult(
        fold_scores=scores,
        mean=float(np.mean(scores)),
        se=standard_error(scores),
        oof=oof,
        secondary={name: float(np.mean(values)) for name, values in per_repeat.items()},
        duration_seconds=time.perf_counter() - started,
    )

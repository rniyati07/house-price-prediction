"""The scope rule (ADR-06, FR-006, DOC-03 §5.5).

Rows with ``GrLivArea`` above the configured threshold are outside the population the
model is built for. They are removed once, on the validated raw dataset, before the split.
This is a scope decision, not cleaning (DOC-02 §9.4).
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

from house_price.config import ScopeConfig
from house_price.data.errors import ScopeRuleError


class ScopeRecord(BaseModel):
    """What the scope rule did; logged and later written into artifact metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule: str
    column: str
    threshold: float
    rows_before: int
    rows_after: int
    removed_ids: list[int]


def apply_scope_rule(
    frame: pd.DataFrame, scope: ScopeConfig, id_column: str
) -> tuple[pd.DataFrame, ScopeRecord]:
    """Remove rows where ``scope.column > scope.threshold``.

    Raises ``ScopeRuleError`` if the in-scope row count differs from
    ``scope.expected_rows_after``: training never continues on an unexpected population
    (AC-006). The discrepancy must be resolved through the ADR supersession process.
    """
    if scope.column not in frame.columns:
        raise ScopeRuleError(f"scope column {scope.column!r} is not in the dataset")
    out_of_scope = frame[scope.column] > scope.threshold
    in_scope = frame.loc[~out_of_scope].reset_index(drop=True)
    record = ScopeRecord(
        rule=f"{scope.column} <= {scope.threshold:g}",
        column=scope.column,
        threshold=scope.threshold,
        rows_before=len(frame),
        rows_after=len(in_scope),
        removed_ids=sorted(int(value) for value in frame.loc[out_of_scope, id_column]),
    )
    if record.rows_after != scope.expected_rows_after:
        raise ScopeRuleError(
            f"scope rule {record.rule!r} left {record.rows_after} rows, expected "
            f"{scope.expected_rows_after} (removed ids: {record.removed_ids}). "
            "Record the discrepancy and follow the ADR supersession process (DOC-02 §9.2)."
        )
    return in_scope, record

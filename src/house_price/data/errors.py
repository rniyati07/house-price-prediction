"""Exceptions raised by the data layer. Each one stops the calling command (AC-002)."""

from __future__ import annotations

import pandas as pd

FAILURE_COLUMNS = ["column", "check", "index", "failure_case"]


class DataError(Exception):
    """Base class for every data-layer failure."""


class RawFileError(DataError):
    """The raw dataset is missing, is not a regular file, or is empty."""


class DataIntegrityError(DataError):
    """The raw file's SHA-256 differs from the committed hash (FR-002, AC-002)."""

    def __init__(self, path: str, expected: str, actual: str) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"SHA-256 mismatch for {path}: expected {expected} (configs/data.yaml), "
            f"found {actual}. The raw file has changed; refusing to continue."
        )


class DataValidationError(DataError):
    """The dataset violates the data contract. Carries every failure (lazy validation)."""

    def __init__(self, failure_cases: pd.DataFrame, max_lines: int = 25) -> None:
        self.failure_cases = failure_cases.reset_index(drop=True)
        summary = self.summary()
        lines = [
            f"  column={row.column!r} check={row.check!r} failures={row.n_failures} "
            f"examples(index)={row.example_indices}"
            for row in summary.head(max_lines).itertuples()
        ]
        if len(summary) > max_lines:
            lines.append(f"  ... and {len(summary) - max_lines} more column/check pairs")
        super().__init__(
            f"schema validation failed: {len(self.failure_cases)} failure case(s) in "
            f"{len(summary)} column/check pair(s)\n" + "\n".join(lines)
        )

    def summary(self) -> pd.DataFrame:
        """One row per failing (column, check) with a count and up to 5 example indices."""
        if self.failure_cases.empty:
            return pd.DataFrame(columns=["column", "check", "n_failures", "example_indices"])
        grouped = self.failure_cases.assign(
            column=self.failure_cases["column"].astype(str),
            check=self.failure_cases["check"].astype(str),
        ).groupby(["column", "check"], sort=True, dropna=False)
        return (
            grouped["index"]
            .agg(
                n_failures="size",
                example_indices=lambda s: [v for v in s.dropna().tolist()][:5],
            )
            .reset_index()
        )


class ScopeRuleError(DataError):
    """The scope rule did not produce the verified in-scope row count (AC-006)."""


class SplitIntegrityError(DataError):
    """Persisted split files are incomplete, altered, or inconsistent (AC-008)."""


class SplitBalanceError(DataError):
    """A price bin's share differs between the sets by more than the tolerance (AC-009)."""

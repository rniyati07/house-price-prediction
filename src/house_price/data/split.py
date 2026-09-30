"""The one permanent development/holdout split (ADR-09, FR-007, DOC-03 §5.7).

The split is stratified on deciles of ``log1p(SalePrice)`` (DN-01), created once with the
global seed, persisted, and never regenerated. Later runs load the development set and
verify the holdout only through its hash and the manifest's ``Id`` list; the holdout rows
are not parsed here (DOC-03 §5.7, holdout isolation).

Run ``python -m house_price.data.split`` to create the split (first run) or verify it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import StratifiedShuffleSplit

from house_price.config import DEFAULT_CONFIG_DIR, ConfigError, ProjectConfig, load_project_config
from house_price.data.errors import DataError, SplitBalanceError, SplitIntegrityError
from house_price.data.load import load_raw, read_typed_csv, sha256_file
from house_price.data.schema import validate_raw
from house_price.data.scope import ScopeRecord, apply_scope_rule

STRATIFY_ON = "log1p(SalePrice) deciles"


class BinBalance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bin: int
    lower_edge: float
    upper_edge: float
    dev_count: int
    holdout_count: int
    dev_share_pct: float
    holdout_share_pct: float
    diff_pp: float


class SplitManifest(BaseModel):
    """Committed record of the split: IDs, seed, bins, file hashes, and balance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_sha256: str
    seed: int
    holdout_fraction: float
    n_bins: int
    stratify_on: str
    bin_edges: list[float]
    n_in_scope: int
    n_dev: int
    n_holdout: int
    dev_sha256: str
    holdout_sha256: str
    max_bin_diff_pp: float
    balance: list[BinBalance]
    scope: ScopeRecord
    dev_ids: list[int]
    holdout_ids: list[int]


@dataclass(frozen=True)
class SplitResult:
    """The development set, the holdout's IDs (never its rows), and the manifest."""

    dev: pd.DataFrame
    holdout_ids: list[int]
    manifest: SplitManifest
    created: bool


def price_bins(prices: pd.Series, n_bins: int) -> tuple[pd.Series, list[float]]:
    """Assign each price to one of ``n_bins`` quantile bins of ``log1p(price)`` (DN-01)."""
    codes, edges = pd.qcut(np.log1p(prices), q=n_bins, labels=False, retbins=True)
    return codes.astype("int64"), [float(edge) for edge in edges]


def split_balance(
    dev_bins: pd.Series, holdout_bins: pd.Series, edges: Sequence[float]
) -> list[BinBalance]:
    """Per-bin share of each set, in percent, and their difference in percentage points."""
    n_bins = len(edges) - 1
    dev_counts = dev_bins.value_counts().reindex(range(n_bins), fill_value=0)
    holdout_counts = holdout_bins.value_counts().reindex(range(n_bins), fill_value=0)
    rows = []
    for index in range(n_bins):
        dev_share = 100.0 * int(dev_counts[index]) / len(dev_bins)
        holdout_share = 100.0 * int(holdout_counts[index]) / len(holdout_bins)
        rows.append(
            BinBalance(
                bin=index,
                lower_edge=float(edges[index]),
                upper_edge=float(edges[index + 1]),
                dev_count=int(dev_counts[index]),
                holdout_count=int(holdout_counts[index]),
                dev_share_pct=round(dev_share, 4),
                holdout_share_pct=round(holdout_share, 4),
                diff_pp=round(abs(dev_share - holdout_share), 4),
            )
        )
    return rows


def check_split_balance(balance: Sequence[BinBalance], tolerance_pp: float) -> float:
    """Raise ``SplitBalanceError`` if any bin's shares differ by more than the tolerance."""
    worst = max(row.diff_pp for row in balance)
    if worst > tolerance_pp:
        failing = [row.bin for row in balance if row.diff_pp > tolerance_pp]
        raise SplitBalanceError(
            f"split balance check failed: bins {failing} differ by up to {worst:.2f} pp "
            f"(tolerance {tolerance_pp} pp, AC-009)"
        )
    return worst


def expected_holdout_sizes(n_rows: int, fraction: float) -> set[int]:
    """Accepted holdout sizes: the fraction rounded down or up (IN-06)."""
    return {math.floor(n_rows * fraction), math.ceil(n_rows * fraction)}


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="", lineterminator="\n")


def _create_split(
    in_scope: pd.DataFrame, scope_record: ScopeRecord, config: ProjectConfig
) -> SplitResult:
    schema, validation = config.schema, config.validation
    id_col, target = schema.id_column, schema.target
    bins, edges = price_bins(in_scope[target], validation.n_bins)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=validation.holdout_fraction, random_state=validation.seed
    )
    dev_index, holdout_index = next(splitter.split(in_scope, bins))
    dev_index, holdout_index = np.sort(dev_index), np.sort(holdout_index)
    dev = in_scope.iloc[dev_index].reset_index(drop=True)
    holdout = in_scope.iloc[holdout_index].reset_index(drop=True)

    if len(holdout) not in expected_holdout_sizes(len(in_scope), validation.holdout_fraction):
        raise SplitIntegrityError(f"unexpected holdout size {len(holdout)}")
    balance = split_balance(bins.iloc[dev_index], bins.iloc[holdout_index], edges)
    worst = check_split_balance(balance, validation.split_balance_tolerance_pp)

    _write_csv(dev, config.dev_path)
    _write_csv(holdout, config.holdout_path)
    manifest = SplitManifest(
        raw_sha256=config.data.raw_sha256,
        seed=validation.seed,
        holdout_fraction=validation.holdout_fraction,
        n_bins=validation.n_bins,
        stratify_on=STRATIFY_ON,
        bin_edges=edges,
        n_in_scope=len(in_scope),
        n_dev=len(dev),
        n_holdout=len(holdout),
        dev_sha256=sha256_file(config.dev_path),
        holdout_sha256=sha256_file(config.holdout_path),
        max_bin_diff_pp=worst,
        balance=balance,
        scope=scope_record,
        dev_ids=[int(value) for value in dev[id_col]],
        holdout_ids=[int(value) for value in holdout[id_col]],
    )
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return SplitResult(dev=dev, holdout_ids=manifest.holdout_ids, manifest=manifest, created=True)


def _load_split(
    in_scope: pd.DataFrame, config: ProjectConfig, verify_holdout_file: bool = True
) -> SplitResult:
    try:
        manifest = SplitManifest.model_validate_json(
            config.manifest_path.read_text(encoding="utf-8")
        )
    except ValueError as exc:
        raise SplitIntegrityError(
            f"unreadable split manifest {config.manifest_path}: {exc}"
        ) from exc

    problems: list[str] = []
    if manifest.raw_sha256 != config.data.raw_sha256:
        problems.append("manifest was created from a different raw file")
    if sha256_file(config.dev_path) != manifest.dev_sha256:
        problems.append(f"{config.dev_path} does not match its manifest hash")
    if verify_holdout_file and sha256_file(config.holdout_path) != manifest.holdout_sha256:
        problems.append(f"{config.holdout_path} does not match its manifest hash")
    dev_ids, holdout_ids = set(manifest.dev_ids), set(manifest.holdout_ids)
    if dev_ids & holdout_ids:
        problems.append(f"{len(dev_ids & holdout_ids)} ids are in both sets")
    in_scope_ids = {int(value) for value in in_scope[config.schema.id_column]}
    if dev_ids | holdout_ids != in_scope_ids:
        problems.append("dev + holdout ids do not equal the in-scope ids")
    if problems:
        raise SplitIntegrityError("persisted split failed verification: " + "; ".join(problems))

    cast = read_typed_csv(
        config.dev_path, config.schema, config.data.missing_tokens, source_headers=False
    )
    dev = validate_raw(cast.frame, config.schema, cast_failures=cast.failures)
    if [int(value) for value in dev[config.schema.id_column]] != manifest.dev_ids:
        raise SplitIntegrityError("development set ids differ from the manifest")
    return SplitResult(dev=dev, holdout_ids=manifest.holdout_ids, manifest=manifest, created=False)


def create_or_load_split(
    in_scope: pd.DataFrame,
    scope_record: ScopeRecord,
    config: ProjectConfig,
    *,
    verify_holdout_file: bool = True,
) -> SplitResult:
    """Create the split on first use; afterwards load and verify it (never regenerate).

    ``verify_holdout_file=False`` is the training path (AC-030, FR-029): the persisted
    split is verified through the manifest, the raw-file hash, and the development file's
    hash, and the holdout file is never opened, not even to hash it. It also never creates
    a split. ``make split`` (and the M9 holdout evaluation) keep verifying the holdout hash.
    """
    if not verify_holdout_file:
        required = [config.dev_path, config.manifest_path]
        absent = [str(path) for path in required if not path.exists()]
        if absent:
            raise SplitIntegrityError(
                f"persisted split not found ({absent}); training never creates the split"
            )
        return _load_split(in_scope, config, verify_holdout_file=False)
    paths = [config.dev_path, config.holdout_path, config.manifest_path]
    present = [path.exists() for path in paths]
    if all(present):
        return _load_split(in_scope, config)
    if any(present):
        existing = [str(path) for path, exists in zip(paths, present, strict=True) if exists]
        raise SplitIntegrityError(
            f"incomplete persisted split (found only {existing}). The split is never "
            "regenerated automatically; restore the missing files or delete all three "
            "deliberately."
        )
    return _create_split(in_scope, scope_record, config)


def run(config: ProjectConfig) -> SplitResult:
    """Load and validate the raw data, apply the scope rule, then create or load the split."""
    raw = load_raw(config)
    in_scope, scope_record = apply_scope_rule(raw, config.data.scope, config.schema.id_column)
    return create_or_load_split(in_scope, scope_record, config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m house_price.data.split",
        description="Create the development/holdout split once, or verify the persisted one.",
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--root", type=Path, default=None, help="base for relative paths")
    args = parser.parse_args(argv)
    try:
        config = load_project_config(args.config_dir, args.root)
        result = run(config)
    except (ConfigError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    manifest = result.manifest
    action = "Created" if result.created else "Loaded existing"
    print(
        f"{action} split: dev={manifest.n_dev} holdout={manifest.n_holdout} "
        f"(in-scope={manifest.n_in_scope}, removed ids={manifest.scope.removed_ids})"
    )
    print(f"Max per-bin share difference: {manifest.max_bin_diff_pp:.2f} pp")
    print(f"Manifest: {config.manifest_path}")
    print(
        json.dumps(
            {"dev_sha256": manifest.dev_sha256, "holdout_sha256": manifest.holdout_sha256}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

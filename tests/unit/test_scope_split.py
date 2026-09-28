"""Scope rule and persisted stratified split (AC-006 to AC-009)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price.config import ProjectConfig
from house_price.data import split
from house_price.data.errors import ScopeRuleError, SplitBalanceError, SplitIntegrityError
from house_price.data.load import load_raw, sha256_file
from house_price.data.scope import ScopeRecord, apply_scope_rule
from house_price.data.split import (
    SplitManifest,
    check_split_balance,
    create_or_load_split,
    expected_holdout_sizes,
    price_bins,
    split_balance,
)
from tests.conftest import (
    FIXTURE_IN_SCOPE,
    FIXTURE_OUT_OF_SCOPE_IDS,
    REPO_ROOT,
    SampleEnv,
    make_env,
)


def _in_scope(config: ProjectConfig) -> tuple[pd.DataFrame, ScopeRecord]:
    return apply_scope_rule(load_raw(config), config.data.scope, config.schema.id_column)


# -------------------------------------------------------------------- scope (AC-006)


def test_scope_rule_removes_rows_above_threshold(sample_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(sample_config)
    assert len(in_scope) == FIXTURE_IN_SCOPE
    assert (in_scope["GrLivArea"] <= 4000).all()
    assert record.removed_ids == FIXTURE_OUT_OF_SCOPE_IDS
    assert record.rows_before == FIXTURE_IN_SCOPE + len(FIXTURE_OUT_OF_SCOPE_IDS)
    assert record.rows_after == FIXTURE_IN_SCOPE
    assert record.rule == "GrLivArea <= 4000"
    assert not set(in_scope["Id"]) & set(record.removed_ids)


def test_scope_rule_refuses_unexpected_row_count(tmp_path: Path) -> None:
    config = make_env(tmp_path, expected_rows_after=FIXTURE_IN_SCOPE + 1).load()
    with pytest.raises(ScopeRuleError, match="supersession"):
        _in_scope(config)


# ------------------------------------------------------------ split (AC-007 to AC-009)


def test_split_sizes_and_disjointness(sample_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(sample_config)
    result = create_or_load_split(in_scope, record, sample_config)
    manifest = result.manifest
    assert result.created
    assert manifest.n_holdout in expected_holdout_sizes(FIXTURE_IN_SCOPE, 0.2)
    assert manifest.n_dev + manifest.n_holdout == FIXTURE_IN_SCOPE
    assert len(result.dev) == manifest.n_dev
    assert not set(manifest.dev_ids) & set(manifest.holdout_ids)
    assert set(manifest.dev_ids) | set(manifest.holdout_ids) == set(in_scope["Id"])
    assert result.dev["Id"].tolist() == manifest.dev_ids


def test_split_is_persisted_with_manifest(sample_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(sample_config)
    create_or_load_split(in_scope, record, sample_config)
    manifest = SplitManifest.model_validate_json(
        sample_config.manifest_path.read_text(encoding="utf-8")
    )
    assert manifest.seed == sample_config.validation.seed
    assert manifest.n_bins == 10 and len(manifest.bin_edges) == 11
    assert manifest.raw_sha256 == sample_config.data.raw_sha256
    assert manifest.dev_sha256 == sha256_file(sample_config.dev_path)
    assert manifest.holdout_sha256 == sha256_file(sample_config.holdout_path)
    assert manifest.scope == record
    holdout = pd.read_csv(sample_config.holdout_path)
    assert {"Id", "SalePrice"} <= set(holdout.columns)
    assert holdout["Id"].tolist() == manifest.holdout_ids


def test_second_run_loads_instead_of_regenerating(sample_config: ProjectConfig) -> None:
    """AC-008: identical Id sets; persisted files are not rewritten."""
    in_scope, record = _in_scope(sample_config)
    first = create_or_load_split(in_scope, record, sample_config)
    paths = [sample_config.dev_path, sample_config.holdout_path, sample_config.manifest_path]
    stamps = [path.stat().st_mtime_ns for path in paths]
    second = create_or_load_split(in_scope, record, sample_config)
    assert not second.created
    assert second.manifest == first.manifest
    assert second.dev["Id"].tolist() == first.dev["Id"].tolist()
    assert [path.stat().st_mtime_ns for path in paths] == stamps
    pd.testing.assert_frame_equal(second.dev, first.dev)


def test_split_is_deterministic_across_fresh_environments(tmp_path: Path) -> None:
    ids = []
    for name in ("a", "b"):
        config = make_env(tmp_path / name).load()
        in_scope, record = _in_scope(config)
        ids.append(create_or_load_split(in_scope, record, config).manifest.holdout_ids)
    assert ids[0] == ids[1]


def test_split_balance_within_tolerance(sample_config: ProjectConfig) -> None:
    """AC-009: each bin's share differs by at most 2 percentage points."""
    in_scope, record = _in_scope(sample_config)
    manifest = create_or_load_split(in_scope, record, sample_config).manifest
    assert len(manifest.balance) == 10
    assert manifest.max_bin_diff_pp <= sample_config.validation.split_balance_tolerance_pp
    assert sum(row.dev_count for row in manifest.balance) == manifest.n_dev
    assert sum(row.holdout_count for row in manifest.balance) == manifest.n_holdout


def test_balance_check_rejects_an_imbalanced_split() -> None:
    prices = pd.Series(np.linspace(50_000, 500_000, 100))
    bins, edges = price_bins(prices, 10)
    dev_bins, holdout_bins = bins.iloc[:80], bins.iloc[80:]  # holdout = top prices only
    balance = split_balance(dev_bins, holdout_bins, edges)
    with pytest.raises(SplitBalanceError, match="AC-009"):
        check_split_balance(balance, 2.0)


def test_tampered_development_file_is_detected(sample_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(sample_config)
    create_or_load_split(in_scope, record, sample_config)
    with sample_config.dev_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(SplitIntegrityError, match="does not match its manifest hash"):
        create_or_load_split(in_scope, record, sample_config)


def test_incomplete_split_is_never_regenerated(sample_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(sample_config)
    create_or_load_split(in_scope, record, sample_config)
    sample_config.holdout_path.unlink()
    with pytest.raises(SplitIntegrityError, match="incomplete persisted split"):
        create_or_load_split(in_scope, record, sample_config)


def test_split_command_creates_then_loads(
    sample_env: SampleEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    assert split.main(sample_env.cli_args()) == 0
    assert "Created split" in capsys.readouterr().out
    assert split.main(sample_env.cli_args()) == 0
    assert "Loaded existing split" in capsys.readouterr().out


# ------------------------------------------------------------------- real dataset


@pytest.mark.data
def test_real_scope_rule(real_config: ProjectConfig) -> None:
    in_scope, record = _in_scope(real_config)
    assert record.rows_before == 2930
    assert len(in_scope) == 2925
    assert record.removed_ids == [1499, 1761, 1768, 2181, 2182]


@pytest.mark.data
def test_real_split_matches_committed_manifest(real_config: ProjectConfig, tmp_path: Path) -> None:
    """Recreate the split in a temporary directory and compare with the committed one."""
    processed = real_config.data.processed.model_copy(
        update={
            "dev_path": tmp_path / "dev.csv",
            "holdout_path": tmp_path / "holdout.csv",
            "manifest_path": tmp_path / "split_manifest.json",
        }
    )
    config = ProjectConfig(
        root=real_config.root,
        data=real_config.data.model_copy(update={"processed": processed}),
        validation=real_config.validation,
        schema=real_config.schema,
    )
    in_scope, record = _in_scope(config)
    manifest = create_or_load_split(in_scope, record, config).manifest
    assert (manifest.n_dev, manifest.n_holdout) in {(2340, 585), (2341, 584)}
    assert manifest.max_bin_diff_pp <= 2.0
    committed = REPO_ROOT / "data" / "processed" / "split_manifest.json"
    if committed.exists():
        stored = json.loads(committed.read_text(encoding="utf-8"))
        assert manifest.holdout_ids == stored["holdout_ids"]
        assert manifest.dev_ids == stored["dev_ids"]
        assert manifest.dev_sha256 == stored["dev_sha256"]
        assert manifest.holdout_sha256 == stored["holdout_sha256"]

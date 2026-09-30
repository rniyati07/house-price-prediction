"""Holdout isolation (DOC-01 AC-030, FR-029; DOC-03 §9.4; DOC-05 M8-3) and the isolation of
reduced runs from the real project.

Training verifies the persisted split through the manifest, the raw-file hash, and the
development file only; the holdout file is never opened. ``make split`` still verifies the
holdout hash. Only ``config.py`` (the path) and ``data/split.py`` (which writes it) may
reference the holdout file; M9's ``evaluation/holdout.py`` will be its only reader.
"""

from __future__ import annotations

import builtins
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from house_price.config import load_models_config
from house_price.data import split
from house_price.data.errors import DataError
from house_price.data.load import load_raw
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from tests.conftest import CONFIG_DIR, REPO_ROOT, M7Train, make_env

HOLDOUT_REFERENCE = re.compile(r"holdout_path|holdout\.csv")
ALLOWED = {"config.py", "data/split.py"}


def _loaded(env_root: Path) -> tuple[Any, Any, Any]:
    env = make_env(env_root)
    assert split.main(env.cli_args()) == 0
    config = env.load()
    in_scope, record = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    return config, in_scope, record


def test_training_and_tuning_run_without_the_holdout_file(m7_train: M7Train) -> None:
    """The fixture deleted the holdout right after the split: both train runs (ablation,
    RC-02, development checks, tuning, blend, comparison) completed without it."""
    assert not m7_train.env.load().holdout_path.exists()
    assert m7_train.first.stopped and not m7_train.second.stopped
    assert set(m7_train.second.studies) == {"ridge", "lasso", "random_forest", "lightgbm"}
    assert len(m7_train.second.comparison) == 7


def test_training_path_never_opens_the_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, in_scope, record = _loaded(tmp_path)
    holdout = config.holdout_path.resolve()
    real_open = builtins.open

    def guarded(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, Path)) and Path(file).resolve() == holdout:
            raise PermissionError(f"test: the holdout must not be opened ({file})")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(io, "open", guarded)
    loaded = create_or_load_split(in_scope, record, config, verify_holdout_file=False)
    assert len(loaded.dev) == loaded.manifest.n_dev
    with pytest.raises(PermissionError, match="must not be opened"):
        create_or_load_split(in_scope, record, config)  # the default path does hash it


def test_make_split_still_verifies_the_holdout(tmp_path: Path) -> None:
    config, in_scope, record = _loaded(tmp_path)
    config.holdout_path.write_bytes(config.holdout_path.read_bytes() + b"tampered\n")
    create_or_load_split(in_scope, record, config, verify_holdout_file=False)  # training: fine
    with pytest.raises(DataError, match="does not match its manifest hash"):
        create_or_load_split(in_scope, record, config)
    config.holdout_path.unlink()
    with pytest.raises(DataError, match="incomplete persisted split"):
        create_or_load_split(in_scope, record, config)


def test_training_path_never_creates_a_split(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    config = env.load()
    in_scope, record = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    with pytest.raises(DataError, match="never creates the split"):
        create_or_load_split(in_scope, record, config, verify_holdout_file=False)
    assert not config.manifest_path.exists() and not config.holdout_path.exists()


def test_only_config_and_split_reference_the_holdout_path() -> None:
    package = REPO_ROOT / "src" / "house_price"
    referencing = {
        p.relative_to(package).as_posix()
        for p in package.rglob("*.py")
        if HOLDOUT_REFERENCE.search(p.read_text(encoding="utf-8"))
    }
    assert referencing == ALLOWED


def test_no_holdout_rows_reach_tuning_or_the_comparison(m7_train: M7Train) -> None:
    env = m7_train.env
    manifest = json.loads(env.load().manifest_path.read_text(encoding="utf-8"))
    dev_ids, holdout_ids = set(manifest["dev_ids"]), set(manifest["holdout_ids"])
    for name, result in m7_train.second.comparison.items():
        assert set(result.oof["Id"]) == dev_ids, name
        oof = pd.read_csv(env.root / "artifacts" / "cv_comparison" / f"{name}_oof.csv")
        assert not set(oof["Id"]) & holdout_ids
    for study in m7_train.second.studies.values():
        for trial in study.trials:
            assert set(trial.result.oof["Id"]) <= dev_ids


def test_reduced_run_leaves_the_real_project_untouched(m7_train: M7Train) -> None:
    """The reduced configuration lives in the isolated project; the real folds, tuning
    artifacts, MLflow store, and canonical configuration are unchanged."""
    assert m7_train.repo_after == m7_train.repo_before
    assert m7_train.env.root != REPO_ROOT and REPO_ROOT not in m7_train.env.root.parents
    reduced = load_models_config(m7_train.env.config_dir).candidates
    canonical = load_models_config(CONFIG_DIR).candidates
    assert (reduced.random_forest.n_trials, reduced.lightgbm.n_trials) != (30, 100)
    assert (canonical.random_forest.n_trials, canonical.lightgbm.n_trials) == (30, 100)
    assert canonical.ridge.grid.n == canonical.lasso.grid.n == 25  # type: ignore[union-attr]
    for path in m7_train.second.tuning_paths.values():
        assert all(m7_train.env.root in p.parents for p in path)

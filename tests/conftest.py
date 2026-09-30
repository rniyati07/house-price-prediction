"""Shared test fixtures.

Fixture-based tests run everywhere: they use ``tests/fixtures/raw_sample.csv`` (102 real
rows in raw-file format: 100 in scope with distinct prices, 2 above the scope threshold)
with the real ``schema.yaml`` and ``validation.yaml`` and a temporary ``data.yaml``.

Tests marked ``@pytest.mark.data`` need ``data/raw/train.csv`` and are skipped when it is
absent (as in CI, which never has the dataset).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
import yaml

from house_price.config import (
    FeatureConfig,
    ProjectConfig,
    SchemaConfig,
    load_feature_config,
    load_project_config,
)
from house_price.data.load import cast_to_schema, load_raw, read_text_csv, sha256_file
from house_price.data.schema import select_model_input
from house_price.models.train import TrainResult, run_train

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "raw_sample.csv"
FEATURE_ROWS_CSV = REPO_ROOT / "tests" / "fixtures" / "feature_rows.csv"
REAL_RAW = REPO_ROOT / "data" / "raw" / "train.csv"

FIXTURE_ROWS = 102
FIXTURE_IN_SCOPE = 100
FIXTURE_OUT_OF_SCOPE_IDS = [1499, 1768]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if REAL_RAW.is_file():
        return
    skip = pytest.mark.skip(reason=f"real dataset not present at {REAL_RAW}")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@dataclass(frozen=True)
class SampleEnv:
    """A self-contained project layout in a temporary directory."""

    root: Path
    config_dir: Path
    raw_path: Path

    def load(self) -> ProjectConfig:
        return load_project_config(self.config_dir, self.root)

    def cli_args(self) -> list[str]:
        return ["--config-dir", str(self.config_dir), "--root", str(self.root)]


def make_env(
    root: Path,
    raw_bytes: bytes | None = None,
    *,
    expected_rows_after: int = FIXTURE_IN_SCOPE,
    committed_sha256: str | None = None,
) -> SampleEnv:
    """Create ``root/configs`` and ``root/data/raw/train.csv``.

    ``raw_bytes`` defaults to the fixture file. The committed hash defaults to the hash of
    the written file; pass ``committed_sha256`` to simulate a mismatch.
    """
    config_dir = root / "configs"
    raw_path = root / "data" / "raw" / "train.csv"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(FIXTURE_CSV.read_bytes() if raw_bytes is None else raw_bytes)
    for name in ("schema.yaml", "validation.yaml", "features.yaml", "models.yaml"):
        shutil.copyfile(CONFIG_DIR / name, config_dir / name)

    data = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    data["raw_sha256"] = committed_sha256 or sha256_file(raw_path)
    data["scope"]["expected_rows_after"] = expected_rows_after
    (config_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return SampleEnv(root=root, config_dir=config_dir, raw_path=raw_path)


FAST_FOLDS, FAST_REPEATS = 3, 1
FAST_GRID, FAST_RF_TRIALS, FAST_LGBM_TRIALS = 3, 2, 2


def use_fast_models(env: SampleEnv) -> None:
    """Shrink the M7/M8 training effort in ``env`` for tests: 3-point grids, 30 LightGBM
    reference trees, 2 Random Forest and 2 LightGBM tuning trials, and 3 folds x 1 repeat
    (every linear fit refits Yeo-Johnson on 48 columns).

    Only the test copies of ``models.yaml`` and ``validation.yaml`` change; the code paths
    are the real ones. The real 5 x 3 folds are covered by ``test_cv.py``.
    """
    validation_path = env.config_dir / "validation.yaml"
    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8"))
    validation["cv_folds"], validation["cv_repeats"] = FAST_FOLDS, FAST_REPEATS
    validation_path.write_text(yaml.safe_dump(validation, sort_keys=False), encoding="utf-8")
    path = env.config_dir / "models.yaml"
    models = yaml.safe_load(path.read_text(encoding="utf-8"))
    for name in ("ridge", "lasso"):
        models["candidates"][name]["grid"]["n"] = FAST_GRID
    models["candidates"]["lightgbm"]["reference"]["n_estimators"] = 30
    models["candidates"]["random_forest"]["n_trials"] = FAST_RF_TRIALS
    models["candidates"]["lightgbm"]["n_trials"] = FAST_LGBM_TRIALS
    path.write_text(yaml.safe_dump(models, sort_keys=False), encoding="utf-8")


def set_ablation_outcome(env: SampleEnv, outcome: Mapping[str, list[str]] | None) -> None:
    """Write (or, with ``None``, remove) ``dropped_engineered`` in ``env``'s features.yaml,
    as the reviewed commit would; used only on temporary test environments."""
    path = env.config_dir / "features.yaml"
    features = yaml.safe_load(path.read_text(encoding="utf-8"))
    for branch in ("linear", "tree"):
        features[branch].pop("dropped_engineered", None)
        if outcome is not None:
            features[branch]["dropped_engineered"] = list(outcome[branch])
    path.write_text(yaml.safe_dump(features, sort_keys=False), encoding="utf-8")


@pytest.fixture
def sample_env(tmp_path: Path) -> SampleEnv:
    return make_env(tmp_path)


@pytest.fixture
def sample_config(sample_env: SampleEnv) -> ProjectConfig:
    return sample_env.load()


@dataclass(frozen=True)
class ModelData:
    """Validated fixture rows as the pipeline sees them (DN-11): 77 inputs plus the target."""

    X: pd.DataFrame
    y: pd.Series
    schema: SchemaConfig
    features: FeatureConfig


@pytest.fixture(scope="session")
def model_data(tmp_path_factory: pytest.TempPathFactory) -> ModelData:
    config = make_env(tmp_path_factory.mktemp("model_data")).load()
    raw = load_raw(config)
    return ModelData(
        X=select_model_input(raw, config.schema),
        y=raw[config.schema.target],
        schema=config.schema,
        features=load_feature_config(CONFIG_DIR),
    )


@pytest.fixture
def feature_config() -> FeatureConfig:
    return load_feature_config(CONFIG_DIR)


@pytest.fixture
def feature_rows() -> pd.DataFrame:
    """The hand-built M4 fixture, parsed and cast exactly like the raw file (DN-18)."""
    schema = load_project_config(CONFIG_DIR, REPO_ROOT).schema
    text = read_text_csv(FEATURE_ROWS_CSV, ["NA", ""])
    return cast_to_schema(text, schema).frame.set_index("Id", drop=False)


@pytest.fixture(scope="session")
def real_config() -> ProjectConfig:
    if not REAL_RAW.is_file():
        pytest.skip(f"real dataset not present at {REAL_RAW}")
    return load_project_config(CONFIG_DIR, REPO_ROOT)


@dataclass(frozen=True)
class M7Train:
    """The RC-02 workflow on the fixture: a first ``train`` with no committed outcome (stops),
    then the proposed outcome written as the reviewed commit would, then a second ``train``
    (development checks, M8 tuning, blend, 7-candidate comparison).

    The holdout file is deleted right after the split is created, so both runs prove that
    training never needs it (AC-030). ``repo_before`` / ``repo_after`` fingerprint the real
    project's folds, tuning artifacts and MLflow store, which the fixture must not touch."""

    env: SampleEnv
    uri: str
    first: TrainResult
    second: TrainResult
    repo_before: dict[str, object]
    repo_after: dict[str, object]


def repo_fingerprint() -> dict[str, object]:
    """Hashes / file lists of the real project's training outputs (never written by tests)."""
    folds = REPO_ROOT / "artifacts" / "cv" / "folds.json"
    tuning = REPO_ROOT / "artifacts" / "tuning"
    mlruns = REPO_ROOT / "mlruns"
    return {
        "folds": sha256_file(folds) if folds.is_file() else None,
        "tuning": sorted((p.name, sha256_file(p)) for p in tuning.glob("*")) if tuning.is_dir() else [],
        "mlruns_entries": sorted(str(p.relative_to(mlruns)) for p in mlruns.rglob("meta.yaml"))
        if mlruns.is_dir() else [],
    }  # fmt: skip


@pytest.fixture(scope="session")
def m7_train(tmp_path_factory: pytest.TempPathFactory) -> M7Train:
    from house_price.data import split

    repo_before = repo_fingerprint()
    env = make_env(tmp_path_factory.mktemp("m7_train"))
    use_fast_models(env)
    set_ablation_outcome(env, None)
    assert split.main(env.cli_args()) == 0
    env.load().holdout_path.unlink()  # training must never need the holdout file
    uri = (env.root / "mlruns").as_uri()
    first = run_train(env.config_dir, env.root, uri)
    assert first.outcome_check is not None
    set_ablation_outcome(env, first.outcome_check.proposed)
    second = run_train(env.config_dir, env.root, uri)
    return M7Train(env, uri, first, second, repo_before, repo_fingerprint())

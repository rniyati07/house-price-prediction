"""Shared test fixtures.

Fixture-based tests run everywhere: they use ``tests/fixtures/raw_sample.csv`` (102 real
rows in raw-file format: 100 in scope with distinct prices, 2 above the scope threshold)
with the real ``schema.yaml`` and ``validation.yaml`` and a temporary ``data.yaml``.

Tests marked ``@pytest.mark.data`` need ``data/raw/train.csv`` and are skipped when it is
absent (as in CI, which never has the dataset).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from house_price.config import ProjectConfig, load_project_config
from house_price.data.load import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
FIXTURE_CSV = REPO_ROOT / "tests" / "fixtures" / "raw_sample.csv"
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
    for name in ("schema.yaml", "validation.yaml"):
        shutil.copyfile(CONFIG_DIR / name, config_dir / name)

    data = yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))
    data["raw_sha256"] = committed_sha256 or sha256_file(raw_path)
    data["scope"]["expected_rows_after"] = expected_rows_after
    (config_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return SampleEnv(root=root, config_dir=config_dir, raw_path=raw_path)


@pytest.fixture
def sample_env(tmp_path: Path) -> SampleEnv:
    return make_env(tmp_path)


@pytest.fixture
def sample_config(sample_env: SampleEnv) -> ProjectConfig:
    return sample_env.load()


@pytest.fixture(scope="session")
def real_config() -> ProjectConfig:
    if not REAL_RAW.is_file():
        pytest.skip(f"real dataset not present at {REAL_RAW}")
    return load_project_config(CONFIG_DIR, REPO_ROOT)

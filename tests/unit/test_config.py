"""Configuration models, loaders, and config_hash (NFR-019, NFR-020)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from house_price.config import (
    ConfigError,
    DataConfig,
    SchemaConfig,
    ValidationConfig,
    config_hash,
    load_model,
    load_project_config,
)
from tests.conftest import CONFIG_DIR, REPO_ROOT


def _write(path: Path, content: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def _data_yaml() -> dict[str, object]:
    return yaml.safe_load((CONFIG_DIR / "data.yaml").read_text(encoding="utf-8"))


def test_committed_configs_load() -> None:
    config = load_project_config(CONFIG_DIR, REPO_ROOT)
    assert config.data.missing_tokens == ["NA", ""]
    assert config.data.scope.column == "GrLivArea"
    assert config.data.scope.threshold == 4000
    assert config.validation.seed == 42
    assert config.validation.holdout_fraction == 0.2
    assert config.validation.n_bins == 10


def test_schema_declares_the_full_contract() -> None:
    schema = load_model(SchemaConfig, CONFIG_DIR / "schema.yaml")
    assert len(schema.columns) == 82
    assert schema.id_column == "Id"
    assert schema.target == "SalePrice"
    assert [c.name for c in schema.with_role("identifier")] == ["Id", "PID"]
    assert [c.name for c in schema.with_role("excluded")] == ["SaleType", "SaleCondition"]
    assert len(schema.with_role("model_input")) == 77
    assert schema.by_name("MasVnrType").allowed_values is not None
    assert "None" in schema.by_name("MasVnrType").allowed_values


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    content = _data_yaml() | {"unexpected_key": 1}
    with pytest.raises(ConfigError, match="unexpected_key"):
        load_model(DataConfig, _write(tmp_path / "data.yaml", content))


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    content = _data_yaml()
    del content["raw_sha256"]
    with pytest.raises(ConfigError, match="raw_sha256"):
        load_model(DataConfig, _write(tmp_path / "data.yaml", content))


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    content = yaml.safe_load((CONFIG_DIR / "validation.yaml").read_text(encoding="utf-8"))
    content["n_bins"] = "ten"
    with pytest.raises(ConfigError, match="n_bins"):
        load_model(ValidationConfig, _write(tmp_path / "validation.yaml", content))


def test_malformed_hash_is_rejected(tmp_path: Path) -> None:
    content = _data_yaml() | {"raw_sha256": "not-a-hash"}
    with pytest.raises(ConfigError, match="raw_sha256"):
        load_model(DataConfig, _write(tmp_path / "data.yaml", content))


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_model(DataConfig, tmp_path / "absent.yaml")


def test_category_column_requires_allowed_values() -> None:
    column = {
        "name": "X",
        "source_name": "X",
        "dtype": "category",
        "nullable": False,
        "role": "model_input",
    }
    target = {"name": "Y", "source_name": "Y", "dtype": "int", "nullable": False, "role": "target"}
    ident = {
        "name": "Id",
        "source_name": "Id",
        "dtype": "int",
        "nullable": False,
        "role": "identifier",
    }
    with pytest.raises(ValueError, match="allowed_values"):
        SchemaConfig.model_validate({"columns": [ident, column, target]})


def test_duplicate_column_names_are_rejected() -> None:
    ident = {
        "name": "Id",
        "source_name": "Id",
        "dtype": "int",
        "nullable": False,
        "role": "identifier",
    }
    target = {
        "name": "Id",
        "source_name": "Price",
        "dtype": "int",
        "nullable": False,
        "role": "target",
    }
    with pytest.raises(ValueError, match="duplicate"):
        SchemaConfig.model_validate({"columns": [ident, target]})


def test_config_hash_is_stable_and_content_sensitive() -> None:
    first = load_project_config(CONFIG_DIR, REPO_ROOT)
    second = load_project_config(CONFIG_DIR, REPO_ROOT)
    assert first.config_hash == second.config_hash
    changed = first.validation.model_copy(update={"seed": 43})
    assert config_hash(first.data, changed, first.schema) != first.config_hash

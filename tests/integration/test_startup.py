"""Startup verification (DOC-04 §5, §14.4, §16.2, §16.4; SD-13; DOC-05 M11-7): the service
starts only after every check passes and refuses bad artifacts, environments and contracts.

Each refusal runs on a temporary copy of the smoke artifact (or of ``configs/``); the
original smoke artifact is never modified."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from house_price.api.app import create_app
from house_price.api.predict import StartupError
from house_price.persistence import artifact
from tests.conftest import API_EXAMPLE, CONFIG_DIR, REPO_ROOT, Serving

STARTUP_EVENTS = ["service.starting", "artifact.verified", "environment.verified",
                  "schema.verified", "model.loaded", "warmup.completed", "service.ready"]  # fmt: skip


def lines(captured: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in captured.splitlines() if line.startswith("{")]


def copy_model(serving: Serving, tmp_path: Path, **metadata_changes: object) -> Path:
    target = tmp_path / "model"
    shutil.copytree(serving.model_dir, target)
    if metadata_changes:
        path = target / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(metadata_changes)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target


def refused(app_settings: object, step: str, capfd: pytest.CaptureFixture[str]) -> str:
    """Startup raises; ``startup.failed`` names ``step``; the service never became ready."""
    app = create_app(app_settings)  # type: ignore[arg-type]
    capfd.readouterr()
    with pytest.raises(StartupError) as caught, TestClient(app):
        pass
    assert caught.value.step == step, caught.value
    out = lines(capfd.readouterr().out)
    (failure,) = [line for line in out if line["event"] == "startup.failed"]
    assert failure["level"] == "CRITICAL" and failure["step"] == step
    assert "service.ready" not in [line["event"] for line in out]
    assert app.state.ready is False and not hasattr(app.state, "model_context")
    return str(failure["reason"])


@pytest.fixture
def no_unpickling(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Fails the test if ``joblib.load`` is reached (records the attempt)."""
    attempts: list[object] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        attempts.append(args)
        raise AssertionError("joblib.load must not be called")

    monkeypatch.setattr(artifact.joblib, "load", forbidden)
    return attempts


# ------------------------------------------------------------------- the happy path


def test_service_starts_from_the_environment(
    serving: Serving, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HPP_MODEL_DIR", str(serving.model_dir))
    monkeypatch.setenv("HPP_CONFIG_DIR", str(serving.config_dir))
    monkeypatch.setenv("HPP_ALLOW_NON_RELEASE", "true")
    monkeypatch.setenv("HPP_LOG_LEVEL", "info")
    app = create_app()
    capfd.readouterr()
    with TestClient(app) as client:
        assert app.state.ready is True
        assert client.get("/health").status_code == 200
    out = lines(capfd.readouterr().out)
    assert [line["event"] for line in out] == [*STARTUP_EVENTS, "service.stopping"]
    ready = out[STARTUP_EVENTS.index("service.ready")]
    assert ready["model_version"] == "unreleased" and ready["startup_ms"] > 0
    assert len(ready["schema_hash"]) == 64


def test_a_released_artifact_starts_without_the_override(serving: Serving, tmp_path: Path) -> None:
    model_dir = copy_model(serving, tmp_path, is_release=True, model_version="1.0.0")
    settings = serving.settings(model_dir=model_dir, allow_non_release=False)
    with TestClient(create_app(settings)) as client:  # type: ignore[arg-type]
        assert client.get("/health").json() == {"status": "ok", "model_version": "1.0.0"}
        assert client.post("/predict", json=json.loads(API_EXAMPLE.read_text()))\
            .json()["model_version"] == "1.0.0"  # fmt: skip


# ----------------------------------------------------------------------- refusals


def test_startup_rejects_non_release(
    serving: Serving, capfd: pytest.CaptureFixture[str], no_unpickling: list[object]
) -> None:
    reason = refused(serving.settings(allow_non_release=False), "release_check", capfd)
    assert "is_release=false" in reason and not no_unpickling


def test_startup_rejects_a_candidate_artifact_even_with_the_override(
    serving: Serving, capfd: pytest.CaptureFixture[str], no_unpickling: list[object]
) -> None:
    candidate = serving.model_dir.parent / "candidates" / "ridge"
    reason = refused(serving.settings(model_dir=candidate), "release_check", capfd)
    assert "candidate artifact" in reason and not no_unpickling


def test_startup_rejects_hash_mismatch(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str],
    no_unpickling: list[object],
) -> None:  # fmt: skip
    model_dir = copy_model(serving, tmp_path)
    with (model_dir / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    reason = refused(serving.settings(model_dir=model_dir), "artifact_integrity", capfd)
    assert "SHA-256" in reason
    assert not no_unpickling  # the file is not loaded


def test_startup_rejects_version_mismatch(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str],
    no_unpickling: list[object],
) -> None:  # fmt: skip
    versions = {**artifact.library_versions(), "scikit-learn": "0.0.1"}
    model_dir = copy_model(serving, tmp_path, library_versions=versions)
    reason = refused(serving.settings(model_dir=model_dir), "environment_check", capfd)
    assert "scikit-learn" in reason and not no_unpickling


def test_schema_hash_mismatch_fails_startup(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str],
    no_unpickling: list[object],
) -> None:  # fmt: skip
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    schema = config_dir / "schema.yaml"
    text = schema.read_text(encoding="utf-8")
    assert text.count("allowed_values: [20, 30, 40, 45, 50, 60") == 1
    schema.write_text(text.replace("allowed_values: [20, 30, 40, 45, 50, 60",
                                   "allowed_values: [20, 30, 40, 45, 50"), encoding="utf-8")  # fmt: skip
    reason = refused(serving.settings(config_dir=config_dir), "contract_check", capfd)
    assert "schema hash" in reason and not no_unpickling


def test_startup_rejects_missing_metadata(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    model_dir = copy_model(serving, tmp_path)
    (model_dir / "metadata.json").unlink()
    refused(serving.settings(model_dir=model_dir), "load_metadata", capfd)


def test_startup_rejects_invalid_metadata(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    model_dir = copy_model(serving, tmp_path, unexpected_field=1)  # extra="forbid"
    refused(serving.settings(model_dir=model_dir), "load_metadata", capfd)


def test_startup_rejects_invalid_settings(
    serving: Serving, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PORT", "eighty")
    reason = refused(None, "read_settings", capfd)
    assert "PORT" in reason


def test_startup_rejects_an_invalid_override_value(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HPP_ALLOW_NON_RELEASE", "yes-please")
    refused(None, "read_settings", capfd)


def test_startup_rejects_a_missing_example(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    (config_dir / "api_example.json").unlink()
    refused(serving.settings(config_dir=config_dir), "build_request_models", capfd)


def test_startup_rejects_an_invalid_example(
    serving: Serving, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    config_dir = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, config_dir)
    example = json.loads((config_dir / "api_example.json").read_text(encoding="utf-8"))
    example.pop("PoolQC")
    (config_dir / "api_example.json").write_text(json.dumps(example), encoding="utf-8")
    refused(serving.settings(config_dir=config_dir), "warm_up", capfd)


class _Stub:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: object) -> object:
        return np.full(len(frame), self.value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), -1.0])
def test_startup_rejects_a_guard_violation(
    serving: Serving, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str],
    value: float,
) -> None:  # fmt: skip
    monkeypatch.setattr(artifact.joblib, "load", lambda *a, **k: _Stub(value))
    reason = refused(serving.settings(), "warm_up", capfd)
    assert "guard" in reason


def test_startup_contains_an_unexpected_exception(
    serving: Serving, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    def broken(*args: object, **kwargs: object) -> object:
        raise MemoryError("simulated failure while unpickling")

    monkeypatch.setattr(artifact.joblib, "load", broken)
    reason = refused(serving.settings(), "load_model", capfd)
    assert "simulated failure" in reason


def test_a_failed_startup_exits_uvicorn_non_zero(serving: Serving, tmp_path: Path) -> None:
    """DOC-04 §5.1: the process exits non-zero and never listens."""
    model_dir = copy_model(serving, tmp_path)
    with (model_dir / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "HPP_MODEL_DIR": str(model_dir), "HPP_CONFIG_DIR": str(CONFIG_DIR),
           "HPP_ALLOW_NON_RELEASE": "true"}  # fmt: skip
    result = subprocess.run(
        [sys.executable, "-m", "uvicorn", "house_price.api.app:app", "--host", "127.0.0.1",
         "--port", str(port), "--workers", "1", "--no-access-log"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert result.returncode != 0
    assert '"event": "startup.failed"' in result.stdout and "artifact_integrity" in result.stdout
    assert "service.ready" not in result.stdout


def test_original_smoke_artifact_is_untouched(serving: Serving) -> None:
    meta = artifact.read_metadata(serving.model_dir)
    assert meta.model_sha256 == artifact.verify(serving.model_dir).model_sha256
    assert joblib.load is not None  # monkeypatches are undone

"""The local run-record layer (house_price.results): one result.json per execution plus an
append-only runs.jsonl index."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price.data.load import sha256_file
from house_price.results import (
    INDEX_NAME,
    RESULT_NAME,
    ResultError,
    read_index,
    read_result,
    start_run,
)
from house_price.tracking import TrackingError, git_state

RECORD_KEYS = {
    "record_version", "milestone", "run_id", "status", "started_at_utc", "finished_at_utc",
    "duration_seconds", "command", "git", "environment", "lineage", "params", "metrics",
    "findings", "artifacts", "stop_reason", "error",
}  # fmt: skip


def test_run_directory_and_valid_record(tmp_path: Path) -> None:
    with start_run("M4", root=tmp_path, entry_point="demo", argv=["--x"]) as run:
        assert run.path.is_file()  # written as "running" at the start
        assert read_result(run.path)["status"] == "running"
    assert run.directory.parent == tmp_path / "results" / "M4"
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{8}", run.directory.name)
    record = json.loads(run.path.read_text(encoding="utf-8"))  # strict, valid JSON
    assert set(record) == RECORD_KEYS
    assert record["milestone"] == "M4" and record["record_version"] == 1
    assert record["status"] == "succeeded" and record["error"] is None
    assert record["command"] == {"entry_point": "demo", "argv": ["--x"]}
    assert record["duration_seconds"] >= 0 and record["finished_at_utc"]


def test_run_ids_are_unique_and_records_never_overwritten(tmp_path: Path) -> None:
    paths = []
    for _ in range(3):
        with start_run("M3", root=tmp_path) as run:
            paths.append(run.path)
    assert len({p.parent for p in paths}) == 3
    assert len({read_result(p)["run_id"] for p in paths}) == 3
    before = paths[-1].read_bytes()
    try:  # the same id again: refused within the same second, a new directory otherwise
        with start_run("M3", root=tmp_path, run_id=run.run_id) as again:
            pass
        assert again.directory != run.directory
    except FileExistsError:
        pass
    assert paths[-1].read_bytes() == before


def test_given_run_id_is_used(tmp_path: Path) -> None:
    with start_run("M6", root=tmp_path, run_id="1f0e2d3c-aaaa-bbbb-cccc-000000000000") as run:
        pass
    assert read_result(run.path)["run_id"] == "1f0e2d3c-aaaa-bbbb-cccc-000000000000"
    assert run.directory.name.endswith("_1f0e2d3c")


def test_metadata_is_the_real_execution_environment(tmp_path: Path) -> None:
    with start_run("M5", root=tmp_path) as run:
        pass
    record = read_result(run.path)
    commit, dirty = git_state()
    assert record["git"] == {"commit": commit, "dirty": dirty}
    env = record["environment"]
    assert {"python", "pandas", "numpy", "scikit-learn", "platform"} <= set(env)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00", record["started_at_utc"])


def test_unreadable_git_is_recorded_as_null_not_guessed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken() -> tuple[str, bool]:
        raise TrackingError("git not available")

    monkeypatch.setattr("house_price.tracking.git_state", broken)
    with start_run("M3", root=tmp_path) as run:
        pass
    assert read_result(run.path)["git"] == {
        "commit": None,
        "dirty": None,
        "error": "git not available",
    }


def test_params_metrics_findings_and_lineage_persist(tmp_path: Path) -> None:
    with start_run("M6", root=tmp_path) as run:
        run.log_params({"cv_folds": 5, "candidates": ("a", "b"), "where": Path("x/y")})
        run.log_metric("a.cv_mean", np.float64(0.25))
        run.log_metrics({"rows": np.int64(2340)})
        run.log_value("nested", {"k": [1, np.int32(2)], "missing": float("nan")})
        run.log_lineage(data_sha256="0" * 64, seed=42)
    record = read_result(run.path)
    assert record["params"] == {"cv_folds": 5, "candidates": ["a", "b"], "where": "x/y"}
    assert record["metrics"] == {"a.cv_mean": 0.25, "rows": 2340.0}
    assert record["findings"] == {"nested": {"k": [1, 2], "missing": None}}
    assert record["lineage"] == {"data_sha256": "0" * 64, "seed": 42}


def test_non_finite_metric_and_unknown_type_are_refused(tmp_path: Path) -> None:
    with start_run("M6", root=tmp_path) as run:
        with pytest.raises(ResultError, match="not finite"):
            run.log_metric("bad", float("inf"))
        with pytest.raises(ResultError, match="cannot record"):
            run.log_value("bad", object())
    assert read_result(run.path)["metrics"] == {}


def test_artifacts_are_registered_by_path_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "reports" / "table.csv"
    source.parent.mkdir()
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    with start_run("M3", root=tmp_path) as run:
        run.log_artifact(source, description="a table")
        saved = run.log_table("summary", pd.DataFrame({"x": [1, 2]}))
        with pytest.raises(ResultError, match="not found"):
            run.log_artifact(tmp_path / "absent.csv")
    artifacts = read_result(run.path)["artifacts"]
    assert artifacts[0] == {
        "path": "reports/table.csv", "sha256": sha256_file(source), "size_bytes": source.stat().st_size,
        "description": "a table",
    }  # fmt: skip
    assert saved == run.directory / "artifacts" / "summary.csv"
    assert artifacts[1]["sha256"] == sha256_file(saved)
    assert len(artifacts) == 2  # the missing file was not registered


def test_a_failing_block_is_recorded_and_the_error_propagates(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), start_run("M6", root=tmp_path) as run:
        run.log_metric("partial", 1.0)
        raise RuntimeError("boom")
    record = read_result(run.path)
    assert record["status"] == "failed"
    assert record["error"] == {"type": "RuntimeError", "message": "boom"}
    assert record["metrics"] == {"partial": 1.0}  # what was logged before the failure is kept
    assert read_index(tmp_path / "results")[-1]["status"] == "failed"


def test_mark_failed_without_exception(tmp_path: Path) -> None:
    with start_run("M3", root=tmp_path) as run:
        run.mark_failed("ADR-fixed number contradicted (P-04)")
    record = read_result(run.path)
    assert record["status"] == "failed"
    assert record["error"]["message"] == "ADR-fixed number contradicted (P-04)"


def test_mark_stopped_is_not_a_failure(tmp_path: Path) -> None:
    with start_run("M7", root=tmp_path) as run:
        run.mark_stopped("RC-02: no committed ablation outcome")
    record = read_result(run.path)
    assert record["status"] == "stopped" and record["error"] is None
    assert record["stop_reason"] == "RC-02: no committed ablation outcome"
    assert read_index(tmp_path / "results")[-1]["status"] == "stopped"


def test_index_is_append_only(tmp_path: Path) -> None:
    results = tmp_path / "results"
    with start_run("M3", root=tmp_path) as first:
        pass
    before = (results / INDEX_NAME).read_bytes()
    with start_run("M4", root=tmp_path) as second:
        pass
    after = (results / INDEX_NAME).read_bytes()
    assert after.startswith(before)  # earlier lines untouched
    index = read_index(results)
    assert [line["run_id"] for line in index] == [first.run_id, second.run_id]
    assert index[1]["result"] == f"M4/{second.directory.name}/{RESULT_NAME}"
    assert index[1]["status"] == read_result(second.path)["status"] == "succeeded"


def test_custom_results_dir_and_bad_milestone(tmp_path: Path) -> None:
    with start_run("M5", root=tmp_path, results_dir=tmp_path / "elsewhere") as run:
        pass
    assert run.directory.parents[1] == tmp_path / "elsewhere"
    with pytest.raises(ResultError, match="milestone"), start_run("eda", root=tmp_path):
        pass

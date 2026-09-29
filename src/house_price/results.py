"""Local run records: one structured ``result.json`` per execution of a milestone command.

Layout (``results/`` is gitignored, like ``mlruns/``)::

    results/
    ├── runs.jsonl                               # append-only index, one line per run
    └── <milestone>/<UTC-timestamp>_<id8>/
        ├── result.json                          # the run record
        └── artifacts/                           # only when log_table is used

Usage::

    with start_run("M6", root=root, run_id=pipeline_run_id, entry_point="house-price train") as run:
        run.log_params({...})
        run.log_metrics({...})
        run.log_value("mlflow", {...})
        run.log_artifact(root / "artifacts/cv/folds.json")

The run ends ``succeeded``; if the block raises, it ends ``failed`` with the error and the
exception propagates. ``mark_failed`` records a failure the command reports through its exit
code instead of an exception. Every value is supplied by the caller from actual execution or
from saved deliverables: this module computes nothing about the data and never guesses
lineage (a git commit that cannot be read is recorded as ``null`` with the reason).

This layer complements MLflow and never calls it: from M6 on, MLflow keeps the detailed
per-candidate runs and the local record summarizes and links to them.
"""

from __future__ import annotations

import json
import math
import platform
import re
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, SupportsFloat

import numpy as np
import pandas as pd

from house_price.data.load import sha256_file
from house_price.data.profile import library_versions, utc_now

RESULTS_DIR = Path("results")
INDEX_NAME = "runs.jsonl"
RESULT_NAME = "result.json"
RECORD_VERSION = 1

Status = Literal["running", "succeeded", "failed"]
_MILESTONE = re.compile(r"^M\d{1,2}$")


class ResultError(ValueError):
    """A run record cannot be written truthfully."""


def default_results_dir(root: Path) -> Path:
    return root / RESULTS_DIR


def jsonable(value: Any) -> Any:
    """Plain JSON types for ``value``: numpy scalars unwrapped, NaN/inf as ``null``."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        return [jsonable(v) for v in items]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ResultError(f"cannot record a value of type {type(value).__name__}: {value!r}")


def _git() -> dict[str, Any]:
    from house_price.tracking import TrackingError, git_state  # imports MLflow: load lazily

    try:
        commit, dirty = git_state()
    except TrackingError as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}
    return {"commit": commit, "dirty": dirty}


def _environment() -> dict[str, Any]:
    return {**library_versions(), "platform": platform.platform()}


@dataclass
class ResultRun:
    """One open run record. Create it with :func:`start_run`."""

    milestone: str
    run_id: str
    directory: Path
    root: Path
    record: dict[str, Any] = field(repr=False)
    started: float = field(repr=False)

    @property
    def path(self) -> Path:
        return self.directory / RESULT_NAME

    @property
    def status(self) -> Status:
        status: Status = self.record["status"]
        return status

    # ------------------------------------------------------------------ logging

    def log_lineage(self, **values: Any) -> None:
        """Data and configuration identity (hashes, seed) under ``lineage``."""
        self.record["lineage"].update(jsonable(values))

    def log_param(self, key: str, value: Any) -> None:
        self.record["params"][key] = jsonable(value)

    def log_params(self, params: Mapping[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_metric(self, key: str, value: SupportsFloat) -> None:
        number = float(value)
        if not math.isfinite(number):
            raise ResultError(f"metric {key!r} is not finite: {value!r}")
        self.record["metrics"][key] = number

    def log_metrics(self, metrics: Mapping[str, SupportsFloat]) -> None:
        for key, value in metrics.items():
            self.log_metric(key, value)

    def log_value(self, key: str, value: Any) -> None:
        """A structured finding (text, a count, a nested mapping...) under ``findings``."""
        self.record["findings"][key] = jsonable(value)

    def log_artifact(self, path: Path, *, description: str | None = None) -> dict[str, Any]:
        """Reference an existing file by path and SHA-256; the file is not copied."""
        path = Path(path)
        if not path.is_file():
            raise ResultError(f"artifact not found: {path}")
        entry: dict[str, Any] = {
            "path": _relative(path, self.root),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if description:
            entry["description"] = description
        self.record["artifacts"].append(entry)
        return entry

    def log_table(self, name: str, table: pd.DataFrame) -> Path:
        """Save ``table`` as ``artifacts/<name>.csv`` in the run directory and register it."""
        target = self.directory / "artifacts" / f"{name}.csv"
        target.parent.mkdir(exist_ok=True)
        table.to_csv(target, index=False, lineterminator="\n")
        self.log_artifact(target)
        return target

    def mark_failed(self, reason: str) -> None:
        """End as ``failed`` without an exception (e.g. a blocking check reported by exit code)."""
        self.record["status"] = "failed"
        self.record["error"] = {"type": "check_failed", "message": reason}

    # ---------------------------------------------------------------- persistence

    def _write(self) -> None:
        text = json.dumps(self.record, indent=2, allow_nan=False, ensure_ascii=False)
        self.path.write_text(text + "\n", encoding="utf-8")

    def _finish(self, error: BaseException | None) -> None:
        if error is not None:
            self.record["status"] = "failed"
            self.record["error"] = {"type": type(error).__name__, "message": str(error)}
        elif self.record["status"] == "running":
            self.record["status"] = "succeeded"
        self.record["finished_at_utc"] = utc_now()
        self.record["duration_seconds"] = round(time.perf_counter() - self.started, 3)
        self._write()
        _append_index(self)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _append_index(run: ResultRun) -> None:
    """Add one line to ``runs.jsonl``; earlier lines are never rewritten."""
    record = run.record
    line = {
        "milestone": run.milestone,
        "run_id": run.run_id,
        "status": record["status"],
        "started_at_utc": record["started_at_utc"],
        "finished_at_utc": record["finished_at_utc"],
        "duration_seconds": record["duration_seconds"],
        "git_commit": record["git"]["commit"],
        "result": _relative(run.path, run.directory.parents[1]),
    }
    index = run.directory.parents[1] / INDEX_NAME
    with index.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


@contextmanager
def start_run(
    milestone: str,
    *,
    root: Path,
    results_dir: Path | None = None,
    run_id: str | None = None,
    entry_point: str | None = None,
    argv: Sequence[str] | None = None,
) -> Iterator[ResultRun]:
    """Open a run record under ``results/<milestone>/`` and finalize it on exit.

    ``run_id`` defaults to a fresh UUID (M6 passes its ``pipeline_run_id``). The run
    directory is created exclusively, so an earlier record is never overwritten.
    """
    if not _MILESTONE.match(milestone):
        raise ResultError(f"milestone must look like 'M6', got {milestone!r}")
    run_id = run_id or str(uuid.uuid4())
    base = (results_dir or default_results_dir(root)) / milestone
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    directory = base / f"{stamp}_{re.sub(r'[^0-9A-Za-z]', '', run_id)[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    record: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "milestone": milestone,
        "run_id": run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "command": {
            "entry_point": entry_point,
            "argv": list(sys.argv[1:] if argv is None else argv),
        },
        "git": _git(),
        "environment": _environment(),
        "lineage": {},
        "params": {},
        "metrics": {},
        "findings": {},
        "artifacts": [],
        "error": None,
    }
    run = ResultRun(milestone, run_id, directory, root, record, time.perf_counter())
    run._write()  # a crash that bypasses Python still leaves a "running" record behind
    try:
        yield run
    except BaseException as exc:
        run._finish(exc)
        raise
    run._finish(None)


def read_result(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return record


def read_index(results_dir: Path) -> list[dict[str, Any]]:
    index = results_dir / INDEX_NAME
    if not index.is_file():
        return []
    return [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line]

"""Reproducibility check: ``make repro-check`` (AC-033, IN-09, DOC-03 §10.7, QG-15).

Two full ``house-price train`` runs, each in its own isolated copy of the project (so the
real outputs, including the committed selection record and its review, are never
overwritten), compared on:

1. identical development and holdout ``Id`` sets (from the split manifests; the holdout
   file itself is never copied or opened);
2. the same selected model;
3. the same selected hyperparameters (exact equality of every value);
4. per-fold and mean CV log-RMSE of every candidate within the configured absolute
   tolerance (``reproducibility_tolerance``, 1e-6), read from each run's MLflow store.

The report (``reports/reproducibility/repro_check.json``) records the largest observed
metric difference. The tolerance is never loosened to hide nondeterminism.

Environment note: the project installs from an unpinned ``requirements.txt`` (there is no
uv lockfile), so AC-033's "same lockfile" condition is only met by running both trainings
in the same environment; the report says so explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from house_price.config import DEFAULT_CONFIG_DIR, load_project_config
from house_price.data.profile import library_versions, utc_now

# Outside the project: MLflow's file store rejects any run directory with a folder named
# "artifacts" in its path (a path-traversal guard), so the isolated projects cannot live
# under <root>/artifacts/.
WORK_DIR = Path(tempfile.gettempdir()) / "house-price-repro"
REPORT_PATH = Path("reports/reproducibility/repro_check.json")
LOCKFILE_NOTE = (
    "No uv lockfile exists; dependencies come from an unpinned requirements.txt. Both runs used "
    "the same interpreter and installed packages (recorded below), but AC-033's lockfile "
    "condition is NOT fully satisfied."
)


class _KeepAwake:
    """Windows: ask the OS not to idle-sleep while the check runs (process-scoped; no
    system setting is changed). A no-op elsewhere."""

    def __enter__(self) -> Self:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # type: ignore[attr-defined]
        return self

    def __exit__(self, *exc: object) -> None:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # type: ignore[attr-defined]


def prepare(root: Path, config_dir: Path, target: Path) -> None:
    """An isolated project: configuration, raw file, development file and manifest only."""
    if target.exists():
        shutil.rmtree(target)
    config = load_project_config(config_dir, root)
    shutil.copytree(config_dir, target / DEFAULT_CONFIG_DIR)
    for source in (config.raw_path, config.dev_path, config.manifest_path):
        destination = target / source.resolve().relative_to(root.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_isolated(target: Path, log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        code = subprocess.call(
            [sys.executable, "-m", "house_price.cli", "train", "--root", str(target)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return code, time.perf_counter() - started


def _comparison_metrics(target: Path, pipeline_run_id: str) -> dict[str, dict[str, float]]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    from mlflow.tracking import MlflowClient

    client = MlflowClient((target / "mlruns").resolve().as_uri())
    experiment = client.get_experiment_by_name("hpp-cv-comparison")
    if experiment is None:
        return {}
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.pipeline_run_id = '{pipeline_run_id}' and tags.stage = 'cv_comparison'"
        ),
    )
    return {
        run.data.tags["candidate"]: {
            k: v for k, v in run.data.metrics.items() if k == "cv_mean" or k.startswith("fold_")
        }
        for run in runs
    }


def _selection(target: Path) -> dict[str, Any]:
    path = target / "reports/selection/selection_record.json"
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


def compare(a: Path, b: Path, tolerance: float) -> dict[str, Any]:
    """AC-033 items 1 to 4 for two finished isolated runs."""
    manifests = [json.loads((t / "data/processed/split_manifest.json").read_text()) for t in (a, b)]
    records = [_selection(t) for t in (a, b)]
    metrics = [
        _comparison_metrics(t, r["pipeline_run_id"]) for t, r in zip((a, b), records, strict=True)
    ]
    folds = [(t / "artifacts/cv/folds.json").read_bytes() for t in (a, b)]
    differences: dict[str, dict[str, float]] = {}
    for name in sorted(set(metrics[0]) | set(metrics[1])):
        one, two = metrics[0].get(name, {}), metrics[1].get(name, {})
        keys = sorted(set(one) | set(two))
        differences[name] = {
            k: abs(one[k] - two[k]) if k in one and k in two else float("inf") for k in keys
        }
    largest = max((d for per in differences.values() for d in per.values()), default=float("inf"))
    checks = {
        "dev_ids_identical": manifests[0]["dev_ids"] == manifests[1]["dev_ids"],
        "holdout_ids_identical": manifests[0]["holdout_ids"] == manifests[1]["holdout_ids"],
        "folds_identical": folds[0] == folds[1],
        "selected_model_identical": records[0]["selected"]["name"]
        == records[1]["selected"]["name"],
        "selected_hyperparameters_identical": records[0]["selected"]["hyperparameters"]
        == records[1]["selected"]["hyperparameters"],
        "same_candidates": sorted(metrics[0]) == sorted(metrics[1]) and len(metrics[0]) == 7,
        "cv_metrics_within_tolerance": largest <= tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "tolerance": tolerance,
        "largest_metric_difference": largest,
        "per_candidate_largest_difference": {n: max(d.values()) for n, d in differences.items()},
        "selected": [r["selected"]["name"] for r in records],
        "selected_hyperparameters": records[0]["selected"]["hyperparameters"],
        "selection_record_ids": [r["selection_record_id"] for r in records],
        "pipeline_run_ids": [r["pipeline_run_id"] for r in records],
    }


def main(argv: Sequence[str] | None = None) -> int:
    from house_price.tracking import TrackingError, git_state

    parser = argparse.ArgumentParser(
        prog="python -m house_price.repro", description="AC-033: two full train runs, compared."
    )
    parser.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    parser.add_argument("--config-dir", type=Path, default=None, help="default: <root>/configs")
    parser.add_argument("--work-dir", type=Path, default=None, help=f"default: {WORK_DIR}")
    parser.add_argument("--out", type=Path, default=None, help=f"default: <root>/{REPORT_PATH}")
    args = parser.parse_args(argv)
    root = (args.root or Path.cwd()).resolve()
    config_dir = (args.config_dir or root / DEFAULT_CONFIG_DIR).resolve()
    work = (args.work_dir or WORK_DIR).resolve()
    if "artifacts" in work.parts:
        print(f"ERROR: work dir {work} is under a folder named 'artifacts'; MLflow's file store "
              "rejects runs there. Choose another --work-dir.", file=sys.stderr)  # fmt: skip
        return 1
    out = args.out or root / REPORT_PATH
    tolerance = load_project_config(config_dir, root).validation.reproducibility_tolerance
    try:
        commit, dirty = git_state()
    except TrackingError as exc:
        commit, dirty = f"unavailable: {exc}", None  # type: ignore[assignment]
    started = utc_now()
    runs = {}
    with _KeepAwake():
        for label in ("run_a", "run_b"):
            target = work / label
            prepare(root, config_dir, target)
            print(f"[repro] {label}: house-price train in {target}", file=sys.stderr, flush=True)
            code, seconds = run_isolated(target, work / f"{label}.log")
            runs[label] = {"root": str(target), "exit_code": code, "seconds": round(seconds, 1)}
            if code != 0:
                print(
                    f"ERROR: {label} failed (exit {code}); see {work / (label + '.log')}",
                    file=sys.stderr,
                )
                return 1
    result = compare(work / "run_a", work / "run_b", tolerance)
    report = {
        **result,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "runs": runs,
        "config_dir": str(config_dir),
        "git": {"commit": commit, "dirty": dirty},
        "environment": {**library_versions(), "executable": sys.executable},
        "lockfile_note": LOCKFILE_NOTE,
        "holdout_file_copied": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("passed", "checks", "largest_metric_difference", "selected")},
            indent=2,
            default=str,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

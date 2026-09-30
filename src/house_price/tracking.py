"""MLflow tracking with the standard lineage tags (ADR-16, FR-036, DOC-03 §14, DN-16).

Runs are logged to the local file store ``./mlruns`` (gitignored). The only way to open a
run is :meth:`Tracker.run`, which stamps every run with the standard tags, so no code path
can log an untagged run:

``pipeline_run_id``, ``git_commit``, ``git_dirty``, ``data_sha256``,
``split_manifest_sha256``, ``config_hash``, ``seed``, ``stage``, ``candidate``
(and ``run_kind`` for evaluation runs, from M9).

Nothing is guessed: if git cannot report the commit, tracking refuses to start.

The project uses MLflow's local file backend (FR-036, ADR-16, DOC-03 §14.1). From MLflow
3.x that backend is in maintenance mode and must be opted into with
``MLFLOW_ALLOW_FILE_STORE=true``; this module sets that default so the documented backend
keeps working. Moving to a database backend would be an ADR change.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # the documented file backend

from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

# Experiment names (DOC-03 §14.2).
EXPERIMENTS: dict[str, str] = {
    "baselines": "hpp-baselines",
    "ablation": "hpp-ablation",
    "tuning": "hpp-tuning",
    "cv_comparison": "hpp-cv-comparison",
    "selection": "hpp-selection",
    "evaluation": "hpp-evaluation",
    "release": "hpp-release",
    "smoke": "hpp-smoke",
}
STANDARD_TAGS = (
    "pipeline_run_id", "git_commit", "git_dirty", "data_sha256", "split_manifest_sha256",
    "config_hash", "seed", "stage", "candidate",
)  # fmt: skip
# DOC-03 §14.3 stages, plus ``development_check``: DOC-05 M7 task 8 tags the one-off
# reference-configuration runs of the four candidates with it (never used for selection).
STAGES = ("baselines", "ablation", "development_check", "tuning", "cv_comparison", "selection",
          "evaluation", "release", "smoke")  # fmt: skip
PARENT_TAG = "mlflow.parentRunId"  # MLflow's own tag for nested runs

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class TrackingError(RuntimeError):
    """A run cannot be tracked with complete, truthful lineage."""


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    """Commit SHA and dirty flag of the repository holding the package source."""
    location = repo or Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(location), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(location), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TrackingError(f"cannot read the git commit for {location}: {exc}") from exc  # fmt: skip
    return commit, bool(status.strip())


def new_pipeline_run_id() -> str:
    """A fresh UUID for one invocation of ``train`` (DN-16)."""
    return str(uuid.uuid4())


def default_tracking_uri(root: Path) -> str:
    """``file:./mlruns`` under the project root, as an absolute URI."""
    return (root / "mlruns").resolve().as_uri()


@dataclass(frozen=True)
class Lineage:
    """The run-independent part of the standard tags; identical for one training run."""

    pipeline_run_id: str
    git_commit: str
    git_dirty: bool
    data_sha256: str
    split_manifest_sha256: str
    config_hash: str
    seed: int

    def __post_init__(self) -> None:
        problems = [
            name for name in ("data_sha256", "split_manifest_sha256", "config_hash")
            if not _SHA256.match(getattr(self, name))
        ]  # fmt: skip
        if not _GIT_SHA.match(self.git_commit):
            problems.append("git_commit")
        if not self.pipeline_run_id:
            problems.append("pipeline_run_id")
        if problems:
            raise TrackingError(f"invalid lineage values: {problems}")

    def tags(self) -> dict[str, str]:
        return {
            "pipeline_run_id": self.pipeline_run_id,
            "git_commit": self.git_commit,
            "git_dirty": str(self.git_dirty).lower(),
            "data_sha256": self.data_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "config_hash": self.config_hash,
            "seed": str(self.seed),
        }


class TrackedRun:
    """An open MLflow run; logs parameters and metrics."""

    def __init__(self, client: MlflowClient, run_id: str):
        self._client = client
        self.run_id = run_id

    def log_params(self, params: Mapping[str, object]) -> None:
        for key, value in params.items():
            self._client.log_param(self.run_id, key, str(value))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self._client.log_metric(self.run_id, key, float(value))

    def set_tag(self, key: str, value: object) -> None:
        """An extra tag (e.g. ``edge_warning`` on a grid tuning run, DOC-03 §10.3)."""
        self._client.set_tag(
            self.run_id, key, str(value).lower() if isinstance(value, bool) else str(value)
        )

    def log_artifact(self, path: Path) -> None:
        """Attach a file (e.g. the ablation table, DOC-03 §14.6) to the run."""
        self._client.log_artifact(self.run_id, str(path))


class Tracker:
    """Opens runs in the project's experiments, always with the standard tags."""

    def __init__(self, tracking_uri: str, lineage: Lineage):
        self.tracking_uri = tracking_uri
        self.lineage = lineage
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def experiment_id(self, key: str) -> str:
        if key not in EXPERIMENTS:
            raise TrackingError(
                f"unknown experiment {key!r}; expected one of {sorted(EXPERIMENTS)}"
            )
        name = EXPERIMENTS[key]
        experiment = self.client.get_experiment_by_name(name)
        return experiment.experiment_id if experiment else self.client.create_experiment(name)

    @contextmanager
    def run(
        self,
        experiment: str,
        *,
        stage: str,
        candidate: str,
        run_name: str | None = None,
        parent_run_id: str | None = None,
    ) -> Iterator[TrackedRun]:
        """Open a tagged run; it ends FINISHED, or FAILED if the block raises.

        With ``parent_run_id`` the run is nested under that run (DOC-03 §14.2), in the
        same experiment.
        """
        if stage not in STAGES:
            raise TrackingError(f"unknown stage {stage!r}; expected one of {STAGES}")
        tags = {**self.lineage.tags(), "stage": stage, "candidate": candidate}
        if parent_run_id is not None:
            tags[PARENT_TAG] = parent_run_id
        run = self.client.create_run(self.experiment_id(experiment), tags=tags, run_name=run_name)
        tracked = TrackedRun(self.client, run.info.run_id)
        try:
            yield tracked
        except BaseException:
            self.client.set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.FAILED))
            raise
        self.client.set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.FINISHED))

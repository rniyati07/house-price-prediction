"""Reproducibility-check machinery (AC-033, IN-09; DOC-05 M9 task 7). The full real-data
check (two full trainings, about 2.5 hours) is run separately; here the machinery is
verified on the small fixture project with the fast test settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from house_price import repro
from house_price.data import split
from tests.conftest import M7Train, make_env

FOLDS = [f"fold_{i:02d}" for i in range(1, 4)]
SEVEN = ["dummy_median", "linear_2feat", "ridge", "lasso", "random_forest", "lightgbm", "blend"]


def test_prepare_copies_the_project_but_never_the_holdout(tmp_path: Path) -> None:
    env = make_env(tmp_path / "source")
    assert split.main(env.cli_args()) == 0
    target = tmp_path / "isolated"
    repro.prepare(env.root, env.config_dir, target)
    assert (target / "configs" / "models.yaml").is_file()
    assert (target / "data/raw/train.csv").is_file()
    assert (target / "data/processed/dev.csv").is_file()
    assert (target / "data/processed/split_manifest.json").is_file()
    assert not (target / "data/processed/holdout.csv").exists()


def _fake_run(
    root: Path, *, selected: str = "blend", hyper: Any = None, shift: float = 0.0
) -> None:
    (root / "data/processed").mkdir(parents=True)
    (root / "data/processed/split_manifest.json").write_text(
        json.dumps({"dev_ids": [1, 2, 3], "holdout_ids": [4]})
    )
    (root / "artifacts/cv").mkdir(parents=True)
    (root / "artifacts/cv/folds.json").write_text("{}")
    (root / "reports/selection").mkdir(parents=True)
    record = {
        "selection_record_id": root.name,
        "pipeline_run_id": root.name,
        "selected": {"name": selected, "hyperparameters": hyper or {"alpha": 1.0}},
    }
    (root / "reports/selection/selection_record.json").write_text(json.dumps(record))
    (root / "metrics.json").write_text(
        json.dumps({n: {"cv_mean": 0.1 + shift, **{f: 0.1 + shift for f in FOLDS}} for n in SEVEN})
    )


@pytest.fixture
def fake_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repro,
        "_comparison_metrics",
        lambda target, _: json.loads((target / "metrics.json").read_text()),
    )


@pytest.mark.usefixtures("fake_metrics")
def test_identical_runs_pass(tmp_path: Path) -> None:
    _fake_run(tmp_path / "a")
    _fake_run(tmp_path / "b", shift=5e-7)  # within 1e-6
    result = repro.compare(tmp_path / "a", tmp_path / "b", 1e-6)
    assert result["passed"] and result["largest_metric_difference"] == pytest.approx(5e-7)


@pytest.mark.usefixtures("fake_metrics")
@pytest.mark.parametrize(
    ("change", "check"),
    [
        ({"shift": 2e-6}, "cv_metrics_within_tolerance"),
        ({"selected": "ridge"}, "selected_model_identical"),
        ({"hyper": {"alpha": 1.0000001}}, "selected_hyperparameters_identical"),
    ],
)
def test_any_difference_fails(tmp_path: Path, change: dict[str, Any], check: str) -> None:
    _fake_run(tmp_path / "a")
    _fake_run(tmp_path / "b", **change)
    result = repro.compare(tmp_path / "a", tmp_path / "b", 1e-6)
    assert not result["passed"] and result["checks"][check] is False


def test_work_dir_under_artifacts_is_refused(tmp_path: Path) -> None:
    env = make_env(tmp_path / "source")
    code = repro.main(["--root", str(env.root), "--config-dir", str(env.config_dir),
                       "--work-dir", str(tmp_path / "artifacts" / "repro")])  # fmt: skip
    assert code == 1


def test_repro_check_end_to_end_on_the_fixture(m7_train: M7Train, tmp_path: Path) -> None:
    """Two full ``train`` runs (fast test settings) in isolated copies must agree."""
    out = tmp_path / "repro_check.json"
    code = repro.main(["--root", str(m7_train.env.root), "--config-dir",
                       str(m7_train.env.config_dir), "--work-dir", str(tmp_path / "work"),
                       "--out", str(out)])  # fmt: skip
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and report["passed"], report["checks"]
    assert report["largest_metric_difference"] <= report["tolerance"] == 1e-6
    assert (
        report["holdout_file_copied"] is False and "NOT fully satisfied" in report["lockfile_note"]
    )
    assert report["selected"][0] == report["selected"][1]

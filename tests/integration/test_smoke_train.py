"""End-to-end smoke training and evaluation (DOC-03 §16.3 DN-17; DOC-05 M9 task 6, RC-04)
on the committed synthetic fixture, in an isolated project without a holdout file."""

from __future__ import annotations

import json
import os

import yaml

from house_price.config import load_project_config
from house_price.results import read_index, read_result
from tests.conftest import SmokeProject

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
from mlflow.tracking import MlflowClient

RUN_KINDS = {"final_holdout_evaluation", "baseline_reference", "quality_gates",
             "temporal_diagnostic"}  # fmt: skip


def test_smoke_train_runs_every_stage(smoke_project: SmokeProject) -> None:
    train = smoke_project.train
    assert train.smoke and not train.stopped
    assert set(train.studies) == {"ridge", "lasso", "random_forest", "lightgbm"}
    assert len(train.comparison) == 7 and train.selection_record is not None
    assert train.selection_record["provenance"]["smoke"] is True
    config = load_project_config(smoke_project.root / "configs", smoke_project.root)
    spec = config.validation.smoke
    assert spec is not None
    assert len(train.studies["ridge"].trials) == spec.grid_points
    assert len(train.studies["random_forest"].trials) == spec.random_forest_trials
    assert len(train.studies["lightgbm"].trials) == spec.lightgbm_trials
    assert all(len(cv.fold_scores) == spec.cv_folds * spec.cv_repeats
               for cv in train.comparison.values())  # fmt: skip


def test_smoke_sample_and_substitute_are_disjoint_development_rows(
    smoke_project: SmokeProject,
) -> None:
    root = smoke_project.root
    info = json.loads((root / "artifacts/smoke/smoke_split.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/processed/split_manifest.json").read_text())
    sample, substitute = set(info["sample_ids"]), set(info["holdout_substitute_ids"])
    assert (len(sample), len(substitute)) == (200, 100) and not sample & substitute
    assert sample | substitute <= set(manifest["dev_ids"])
    assert not (sample | substitute) & set(manifest["holdout_ids"])
    assert not (root / "data/processed/holdout.csv").exists()  # never needed


def test_smoke_logs_only_to_hpp_smoke(smoke_project: SmokeProject) -> None:
    client = MlflowClient((smoke_project.root / "mlruns").resolve().as_uri())
    names = {e.name for e in client.search_experiments()} - {"Default"}
    assert names == {"hpp-smoke"}
    experiment = client.get_experiment_by_name("hpp-smoke")
    runs = client.search_runs([experiment.experiment_id], max_results=5000)  # type: ignore[union-attr]
    stages = {r.data.tags["stage"] for r in runs}
    assert {"baselines", "ablation", "tuning", "cv_comparison", "selection",
            "evaluation"} <= stages  # fmt: skip
    kinds = {r.data.tags.get("run_kind") for r in runs} - {None}
    assert kinds == RUN_KINDS
    final = [r for r in runs if r.data.tags.get("run_kind") == "final_holdout_evaluation"]
    assert len(final) == 1 and final[0].data.tags["smoke"] == "true"


def test_smoke_writes_only_under_artifacts_smoke(smoke_project: SmokeProject) -> None:
    root = smoke_project.root
    assert not (root / "reports/selection").exists()
    assert not (root / "reports/eda").exists() and not (root / "reports/evaluation").exists()
    assert not (root / "artifacts/tuning").exists() and not (root / "artifacts/cv").exists()
    smoke = root / "artifacts/smoke"
    for path in ("selection/selection_record.json", "selection/diagnostic_review.yaml",
                 "selection/residual_vs_predicted.png", "evaluation/quality_gates.json",
                 "evaluation/temporal_diagnostic.json", "ablation/E-30_ablation.csv"):  # fmt: skip
        assert (smoke / path).is_file(), path
    review = yaml.safe_load((smoke / "selection/diagnostic_review.yaml").read_text())
    assert review["smoke"] is True and review["status"] == "final"


def test_smoke_evaluation_summary(smoke_project: SmokeProject) -> None:
    summary = smoke_project.evaluation
    assert summary["smoke"] is True and summary["experiment"] == "hpp-smoke"
    assert set(summary["mlflow_run_ids"]) == RUN_KINDS  # type: ignore[call-overload]
    gates = summary["quality_gates"]
    assert gates["enforced"] is False  # type: ignore[index]  # DN-17
    temporal = summary["temporal_diagnostic"]
    assert temporal["label"] == "not used for selection"  # type: ignore[index]
    assert temporal["n_train"] + temporal["n_test"] == 200  # type: ignore[index,operator]


def test_smoke_run_records(smoke_project: SmokeProject) -> None:
    # the fixture's own two executions (train --smoke, evaluate --smoke); later tests may
    # append refused evaluations to the same shared project
    index = read_index(smoke_project.root / "results")[:2]
    assert [line["milestone"] for line in index] == ["M9", "M9"]
    assert [line["status"] for line in index] == ["succeeded", "succeeded"]
    train = read_result(smoke_project.root / "results" / index[0]["result"])
    assert train["params"]["smoke"] is True
    assert train["findings"]["smoke"]["n_sample"] == 200
    assert (
        train["findings"]["selection"]["selected"]["name"]
        == (
            smoke_project.train.selection_record["selected"]["name"]  # type: ignore[index]
        )
    )

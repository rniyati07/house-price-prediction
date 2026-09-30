"""MLflow tracking and the M6 training run (DOC-03 §14; DOC-05 M6-4 to M6-6; DOC-01 AC-032
partial). The training run uses the 102-row fixture and a temporary MLflow store."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from house_price.cli import main as cli_main
from house_price.config import load_models_config
from house_price.data import split
from house_price.data.load import sha256_file
from house_price.models.registry import PASSTHROUGH, baseline_candidates
from house_price.models.train import EXIT_RC02_STOP, TrainError, TrainResult, run_train
from house_price.results import read_index, read_result
from house_price.tracking import (
    EXPERIMENTS,
    PARENT_TAG,
    STANDARD_TAGS,
    Lineage,
    Tracker,
    TrackingError,
    git_state,
)
from tests.conftest import (
    CONFIG_DIR,
    FAST_FOLDS,
    FAST_GRID,
    FAST_LGBM_TRIALS,
    FAST_REPEATS,
    FAST_RF_TRIALS,
    M7Train,
    SampleEnv,
    make_env,
    set_ablation_outcome,
    use_fast_models,
)

HEX64 = "a" * 64
COMMIT = "b" * 40


def _lineage(**override: object) -> Lineage:
    values: dict[str, object] = {
        "pipeline_run_id": "run-1", "git_commit": COMMIT, "git_dirty": False,
        "data_sha256": HEX64, "split_manifest_sha256": HEX64, "config_hash": HEX64, "seed": 42,
    }  # fmt: skip
    values.update(override)
    return Lineage(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------- tracker


def test_every_run_carries_the_standard_tags(tmp_path: Path) -> None:
    tracker = Tracker((tmp_path / "mlruns").as_uri(), _lineage())
    with tracker.run("baselines", stage="baselines", candidate="dummy_median") as run:
        run.log_params({"strategy": "median"})
        run.log_metrics({"cv_mean": 0.4})
    client = MlflowClient((tmp_path / "mlruns").as_uri())
    stored = client.get_run(run.run_id)
    assert set(STANDARD_TAGS) <= set(stored.data.tags)
    assert stored.data.tags["candidate"] == "dummy_median"
    assert stored.data.tags["stage"] == "baselines"
    assert stored.data.tags["git_dirty"] == "false"
    assert stored.data.tags["seed"] == "42"
    assert stored.data.params == {"strategy": "median"}
    assert stored.data.metrics == {"cv_mean": 0.4}
    assert stored.info.status == "FINISHED"
    experiment = client.get_experiment(stored.info.experiment_id)
    assert experiment.name == EXPERIMENTS["baselines"] == "hpp-baselines"


def test_a_failing_block_marks_the_run_failed(tmp_path: Path) -> None:
    tracker = Tracker((tmp_path / "mlruns").as_uri(), _lineage())
    with (
        pytest.raises(RuntimeError),
        tracker.run("baselines", stage="baselines", candidate="x") as run,
    ):
        raise RuntimeError("boom")
    assert MlflowClient((tmp_path / "mlruns").as_uri()).get_run(run.run_id).info.status == "FAILED"


@pytest.mark.parametrize(
    "field", ["data_sha256", "split_manifest_sha256", "config_hash", "git_commit"]
)
def test_lineage_refuses_made_up_values(field: str) -> None:
    with pytest.raises(TrackingError, match=field):
        _lineage(**{field: "unknown"})


def test_unknown_experiment_or_stage_is_refused(tmp_path: Path) -> None:
    tracker = Tracker((tmp_path / "mlruns").as_uri(), _lineage())
    with pytest.raises(TrackingError), tracker.run("bogus", stage="baselines", candidate="x"):
        pass
    with pytest.raises(TrackingError), tracker.run("baselines", stage="bogus", candidate="x"):
        pass


def test_git_state_reads_the_repository() -> None:
    commit, dirty = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    assert isinstance(dirty, bool)


# ------------------------------------------------------ M6/M7 training run (fixture)
#
# ``m7_train`` (conftest) runs ``train`` twice on the fixture with the fast test settings
# (3 folds x 1 repeat): the first run stops at RC-02, the second passes it.


def _runs(uri: str, experiment: str, pipeline_run_id: str) -> list:  # type: ignore[type-arg]
    client = MlflowClient(uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return []
    runs = client.search_runs([found.experiment_id])
    return [run for run in runs if run.data.tags["pipeline_run_id"] == pipeline_run_id]


def _n_folds(env: SampleEnv) -> int:
    validation = env.load().validation
    return validation.cv_folds * validation.cv_repeats


def test_both_baselines_logged_with_fold_scores_mean_and_se(m7_train: M7Train) -> None:
    result, n = m7_train.first, _n_folds(m7_train.env)
    runs = {r.data.tags["candidate"]: r
            for r in _runs(m7_train.uri, "hpp-baselines", result.pipeline_run_id)}  # fmt: skip
    assert set(runs) == {"dummy_median", "linear_2feat"}
    for name, run in runs.items():
        metrics = run.data.metrics
        folds = [metrics[f"fold_{i:02d}"] for i in range(1, n + 1)]
        assert f"fold_{n + 1:02d}" not in metrics
        assert metrics["cv_mean"] == pytest.approx(sum(folds) / n)
        assert metrics["cv_se"] == pytest.approx(result.results[name].se)
        assert {"cv_mae", "cv_mape", "cv_r2", "duration_seconds"} <= set(metrics)
        assert run.data.params["n_folds"] == str(FAST_FOLDS)
        assert run.data.params["n_repeats"] == str(FAST_REPEATS)
        assert run.info.status == "FINISHED"


def _all_runs(m7: M7Train, result: TrainResult) -> list:  # type: ignore[type-arg]
    experiments = ("hpp-baselines", "hpp-ablation", "hpp-cv-comparison", "hpp-tuning")
    return [run for experiment in experiments
            for run in _runs(m7.uri, experiment, result.pipeline_run_id)]  # fmt: skip


def test_training_runs_carry_true_lineage(m7_train: M7Train) -> None:
    config = m7_train.env.load()
    commit, _ = git_state()
    for result in (m7_train.first, m7_train.second):
        runs = _all_runs(m7_train, result)
        assert runs
        for run in runs:
            tags = run.data.tags
            assert set(STANDARD_TAGS) <= set(tags)
            assert tags["pipeline_run_id"] == result.pipeline_run_id
            assert tags["git_commit"] == commit
            assert tags["data_sha256"] == config.data.raw_sha256 == sha256_file(config.raw_path)
            assert tags["split_manifest_sha256"] == sha256_file(config.manifest_path)
            assert re.fullmatch(r"[0-9a-f]{64}", tags["config_hash"])
            assert tags["seed"] == "42"
            assert tags["stage"] in {
                "baselines",
                "ablation",
                "development_check",
                "tuning",
                "cv_comparison",
            }


def test_shared_folds_are_written_for_the_development_set(m7_train: M7Train) -> None:
    env, result = m7_train.env, m7_train.first
    assert result.folds_path == env.root / "artifacts" / "cv" / "folds.json"
    stored = json.loads(result.folds_path.read_text(encoding="utf-8"))
    manifest = json.loads(env.load().manifest_path.read_text(encoding="utf-8"))
    assert stored["dev_sha256"] == manifest["dev_sha256"]
    assert len(stored["folds"]) == _n_folds(env)
    validated = {i for f in stored["folds"] for i in f["valid_ids"]}
    assert validated == set(manifest["dev_ids"])
    assert not validated & set(manifest["holdout_ids"])  # the holdout is never in any fold


def test_baseline_results_are_plausible(m7_train: M7Train) -> None:
    """M6-6: the dummy scores near the SD of log price; the heuristic is clearly better."""
    result = m7_train.first
    dummy, heuristic = result.results["dummy_median"], result.results["linear_2feat"]
    assert dummy.mean == pytest.approx(result.dev_log_price_sd, rel=0.2)
    assert heuristic.mean < dummy.mean - dummy.se


def test_ablation_parent_and_nested_runs(m7_train: M7Train) -> None:
    """DOC-03 §14.2: one parent run per branch, one nested run per removed feature."""
    result = m7_train.first
    runs = _runs(m7_train.uri, "hpp-ablation", result.pipeline_run_id)
    parents = [r for r in runs if PARENT_TAG not in r.data.tags]
    assert {r.info.run_id for r in parents} == set(result.ablation_run_ids.values())
    client = MlflowClient(m7_train.uri)
    for branch, parent_id in result.ablation_run_ids.items():
        parent = client.get_run(parent_id)
        assert parent.data.tags["stage"] == "ablation"
        assert parent.data.tags["candidate"] == result.ablation[branch].reference
        assert parent.data.metrics["mean_with"] == pytest.approx(result.ablation[branch].full.mean)
        artifacts = {a.path for a in client.list_artifacts(parent_id)}
        assert "E-30_ablation.csv" in artifacts
        children = [r for r in runs if r.data.tags.get(PARENT_TAG) == parent_id]
        assert len(children) == 12
        by_feature = {c.data.params["feature_removed"]: c for c in children}
        for row in result.ablation[branch].table().itertuples():
            child = by_feature[row.feature].data.metrics
            assert child["delta"] == pytest.approx(row.delta)
            assert child["retained"] == float(bool(row.retained))


def test_development_checks_logged_to_cv_comparison(m7_train: M7Train) -> None:
    result = m7_train.second
    runs = {r.data.tags["candidate"]: r
            for r in _runs(m7_train.uri, "hpp-cv-comparison", result.pipeline_run_id)
            if r.data.tags["stage"] == "development_check"}  # fmt: skip
    assert set(runs) == {"ridge", "lasso", "random_forest", "lightgbm"}
    assert not _runs(m7_train.uri, "hpp-cv-comparison", m7_train.first.pipeline_run_id)
    for name, run in runs.items():
        assert run.data.tags["stage"] == "development_check"
        assert run.info.run_id == result.dev_check_run_ids[name]
        assert run.data.params["used_for_selection"] == "False"
        assert run.data.metrics["cv_mean"] == pytest.approx(result.dev_checks[name].mean)
    rf = runs["random_forest"].data.params  # the complete effective configuration
    assert rf["n_estimators"] == "100" and rf["max_depth"] == "None"
    assert rf["random_state"] == "42" and rf["n_jobs"] == "-1" and rf["bootstrap"] == "True"
    ridge_alpha = result.ablation["linear"].grid.best_value  # type: ignore[union-attr]
    assert float(runs["ridge"].data.params["alpha"]) == pytest.approx(ridge_alpha)
    assert runs["lasso"].data.params["alpha"] == "0.001"


def test_m7_records_summarize_and_link_the_mlflow_runs(m7_train: M7Train) -> None:
    """One record per execution; run_id = pipeline_run_id; agrees with MLflow exactly."""
    env = m7_train.env
    index = read_index(env.root / "results")
    assert [line["run_id"] for line in index] == [
        m7_train.first.pipeline_run_id, m7_train.second.pipeline_run_id,
    ]  # fmt: skip
    assert [line["status"] for line in index] == ["stopped", "succeeded"]
    for result in (m7_train.first, m7_train.second):
        assert result.result_path is not None
        record = read_result(result.result_path)
        assert record["milestone"] == "M9" and record["run_id"] == result.pipeline_run_id
        assert result.result_path.parent.parent == env.root / "results" / "M9"
        mlflow = record["findings"]["mlflow"]
        assert mlflow["tracking_uri"] == m7_train.uri
        assert mlflow["runs"]["hpp-baselines"] == result.run_ids
        assert mlflow["runs"]["hpp-ablation"] == result.ablation_run_ids
        for run in _runs(m7_train.uri, "hpp-baselines", result.pipeline_run_id):
            name, metrics = run.data.tags["candidate"], run.data.metrics
            for key in ("cv_mean", "cv_se", "cv_mae", "cv_mape", "cv_r2"):
                assert record["metrics"][f"{name}.{key}"] == pytest.approx(metrics[key], rel=1e-12)
            for key in ("data_sha256", "split_manifest_sha256", "config_hash"):
                assert run.data.tags[key] == record["lineage"][key]
        assert not any(key.split(".")[-1].startswith("fold_") for key in record["metrics"])
        paths = {a["path"]: a for a in record["artifacts"]}
        assert paths["artifacts/cv/folds.json"]["sha256"] == sha256_file(result.folds_path)
        assert "reports/eda/E-30_ablation.csv" in paths
        assert record["findings"]["plausibility"]["heuristic_beats_dummy_by_more_than_one_se"]
    first = read_result(m7_train.first.result_path)  # type: ignore[arg-type]
    assert first["findings"]["rc02"]["status"] == "missing"
    assert "hpp-cv-comparison" not in first["findings"]["mlflow"]["runs"]
    second = read_result(m7_train.second.result_path)  # type: ignore[arg-type]
    assert second["findings"]["rc02"]["status"] == "match"
    assert second["findings"]["mlflow"]["runs"]["hpp-cv-comparison"] == (
        m7_train.second.dev_check_run_ids
    )
    for name, cv in m7_train.second.dev_checks.items():
        assert second["metrics"][f"development_check.{name}.cv_mean"] == pytest.approx(cv.mean)
    assert second["findings"]["development_check"]["used_for_selection"] is False


# ---------------------------------------------------------------- M8: tuning


TUNED = ("ridge", "lasso", "random_forest", "lightgbm")
SEVEN = ("dummy_median", "linear_2feat", "ridge", "lasso", "random_forest", "lightgbm", "blend")


def test_tuning_parent_and_nested_runs(m7_train: M7Train) -> None:
    """DOC-03 §10.1, §10.8, §14.2 / AC-032: one parent per tuned candidate in hpp-tuning and
    one nested run per grid point or trial, each with its parameters and fold metrics."""
    result, n_folds = m7_train.second, _n_folds(m7_train.env)
    runs = _runs(m7_train.uri, "hpp-tuning", result.pipeline_run_id)
    parents = {r.data.tags["candidate"]: r for r in runs if PARENT_TAG not in r.data.tags}
    assert set(parents) == set(TUNED)
    assert {r.info.run_id for r in parents.values()} == set(result.study_run_ids.values())
    budgets = {"ridge": FAST_GRID, "lasso": FAST_GRID, "random_forest": FAST_RF_TRIALS,
               "lightgbm": FAST_LGBM_TRIALS}  # fmt: skip
    for name, parent in parents.items():
        study = result.studies[name]
        children = [r for r in runs if r.data.tags.get(PARENT_TAG) == parent.info.run_id]
        assert len(children) == len(study.trials) == budgets[name]
        p, m = parent.data.params, parent.data.metrics
        assert p["tuning_method"] == study.method and p["n_trials"] == str(budgets[name])
        assert p["n_folds"] == str(n_folds) and p["direction"] == "minimize"
        assert m["best_cv_mean"] == pytest.approx(study.best.result.mean)
        assert m["best_cv_se"] == pytest.approx(study.best.result.se)
        assert {"best_cv_mae", "best_cv_mape", "best_cv_r2", "duration_seconds"} <= set(m)
        if study.method == "grid":
            assert parent.data.tags["edge_warning"] == str(study.edge_warning).lower()
            assert p["sampler"] == "grid"
        else:
            assert "edge_warning" not in parent.data.tags
            assert p["sampler"] == "TPESampler" and p["seed"] == "42"
            assert p["pruner"] == "none" and p["n_jobs"] == "1"
        for child in children:
            cm = child.data.metrics
            folds = sorted(k for k in cm if k.startswith("fold_"))
            assert folds == [f"fold_{i:02d}" for i in range(1, n_folds + 1)]
            assert {"cv_mean", "cv_se", "cv_mae", "cv_mape", "cv_r2", "duration_seconds"} <= set(cm)
            assert "trial_number" in child.data.params
            assert set(STANDARD_TAGS) <= set(child.data.tags)
            assert child.data.tags["stage"] == "tuning"


def test_best_json_is_complete_and_matches_the_study(m7_train: M7Train) -> None:
    result = m7_train.second
    folds_sha = sha256_file(result.folds_path)
    for name in TUNED:
        path = m7_train.env.root / "artifacts" / "tuning" / f"{name}_best.json"
        best = json.loads(path.read_text(encoding="utf-8"))
        study = result.studies[name]
        assert best["candidate"] == name and best["tuning_method"] == study.method
        assert best["best_params"] == study.best.params
        assert best["best_cv_mean"] == pytest.approx(study.best.result.mean)
        assert best["best_fold_scores"] == pytest.approx(study.best.result.fold_scores)
        assert best["n_trials"] == len(study.trials)
        assert best["mlflow"]["parent_run_id"] == result.study_run_ids[name]
        assert len(best["mlflow"]["child_run_ids"]) == len(study.trials)
        assert best["lineage"]["folds_sha256"] == folds_sha
        assert best["lineage"]["pipeline_run_id"] == result.pipeline_run_id
        assert re.fullmatch(r"[0-9a-f]{64}", best["search_space_sha256"])
        if study.method == "grid":
            assert best["edge_warning"] == study.edge_warning and best["params_at_bounds"] is None
            assert best["seed"] is None
        else:
            assert best["edge_warning"] is None and isinstance(best["params_at_bounds"], dict)
            assert best["seed"] == 42 and best["sampler"] == "TPESampler"
        trials = pd.read_csv(path.with_name(f"{name}_trials.csv"))
        assert list(trials["cv_mean"]) == pytest.approx([t.result.mean for t in study.trials])


def test_seven_candidate_comparison(m7_train: M7Train) -> None:
    """M8-6 / AC-028 / AC-029: exactly the 7 candidates on the shared folds; tuned candidates
    reproduce their best trial exactly (DOC-03 §11.1); evidence only, no selection."""
    result = m7_train.second
    assert tuple(result.comparison) == SEVEN
    runs = [r for r in _runs(m7_train.uri, "hpp-cv-comparison", result.pipeline_run_id)
            if r.data.tags["stage"] == "cv_comparison"]  # fmt: skip
    assert sorted(r.data.tags["candidate"] for r in runs) == sorted(SEVEN)
    for run in runs:
        name = run.data.tags["candidate"]
        assert run.info.run_id == result.comparison_run_ids[name]
        assert run.data.metrics["cv_mean"] == pytest.approx(result.comparison[name].mean)
        artifacts = {a.path for a in MlflowClient(m7_train.uri).list_artifacts(run.info.run_id)}
        assert f"{name}_oof.csv" in artifacts
    for name in TUNED:
        assert result.comparison[name].fold_scores == pytest.approx(
            result.studies[name].best.result.fold_scores, rel=1e-12
        )
    table = pd.read_csv(m7_train.env.root / "artifacts" / "cv_comparison" / "comparison.csv")
    assert list(table["candidate"]) == list(SEVEN)
    assert list(table["tier"].fillna(0).astype(int)) == [0, 0, 1, 1, 2, 3, 4]
    assert result.blend_linear in ("ridge", "lasso")
    linear = min(("ridge", "lasso"), key=lambda n: result.studies[n].best.result.mean)
    assert result.blend_linear == linear
    record = read_result(result.result_path)  # type: ignore[arg-type]
    comparison = record["findings"]["cv_comparison"]
    assert comparison["candidates"] == list(SEVEN) and "no model is selected" in comparison["note"]
    # M9: train now ends with selection; the comparison itself selects nothing
    assert record["findings"]["selection"]["selected"]["name"] in {
        "ridge",
        "lasso",
        "random_forest",
        "lightgbm",
        "blend",
    }


def test_m8_record_summarizes_tuning(m7_train: M7Train) -> None:
    result = m7_train.second
    record = read_result(result.result_path)  # type: ignore[arg-type]
    assert record["lineage"]["folds_sha256"] == sha256_file(result.folds_path)
    tuning = record["findings"]["tuning"]
    assert set(tuning) == set(TUNED)
    for name, summary in tuning.items():
        study = result.studies[name]
        assert summary["best_params"] == study.best.params
        assert summary["n_trials"] == len(study.trials)
        assert summary["edge_warning"] == study.edge_warning
        assert summary["mlflow_parent_run_id"] == result.study_run_ids[name]
        assert record["metrics"][f"tuning.{name}.best_cv_mean"] == pytest.approx(
            study.best.result.mean
        )
    assert record["findings"]["mlflow"]["runs"]["hpp-tuning"] == result.study_run_ids
    paths = {a["path"] for a in record["artifacts"]}
    assert {f"artifacts/tuning/{n}_best.json" for n in TUNED} <= paths
    assert "artifacts/cv_comparison/comparison.csv" in paths


def test_a_failed_training_run_is_recorded(tmp_path: Path) -> None:
    env = make_env(tmp_path)  # no split: training refuses to start
    with pytest.raises(TrainError):
        run_train(env.config_dir, env.root, (tmp_path / "mlruns").as_uri())
    (line,) = read_index(tmp_path / "results")
    record = read_result(tmp_path / "results" / line["result"])
    assert line["milestone"] == "M9" and record["status"] == "failed"
    assert record["error"]["type"] == "TrainError" and "never creates" in record["error"]["message"]


def test_registry_holds_exactly_the_two_ineligible_baselines() -> None:
    candidates = baseline_candidates(load_models_config(CONFIG_DIR))
    assert [c.name for c in candidates] == ["dummy_median", "linear_2feat"]
    assert not any(c.eligible for c in candidates)
    assert all(c.tuning is None and c.tier is None for c in candidates)
    assert candidates[1].branch == PASSTHROUGH
    assert candidates[1].params["features"] == ["OverallQual", "GrLivArea"]


def test_training_never_creates_the_split(tmp_path: Path) -> None:
    env = make_env(tmp_path)  # no split created
    with pytest.raises(TrainError, match="never creates"):
        run_train(env.config_dir, env.root, (tmp_path / "mlruns").as_uri())
    assert not (env.root / "data" / "processed").exists()


def test_cli_train_stops_at_rc02_and_reports(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = make_env(tmp_path)
    use_fast_models(env)
    set_ablation_outcome(env, None)
    assert split.main(env.cli_args()) == 0
    capsys.readouterr()
    code = cli_main(["train", *env.cli_args(), "--tracking-uri", (tmp_path / "mlruns").as_uri()])
    output = capsys.readouterr().out
    assert code == EXIT_RC02_STOP == 3
    assert "dummy_median" in output and "linear_2feat" in output and "hpp-baselines" in output
    assert "Feature ablation" in output and "RC-02 STOP" in output
    assert "Proposed linear.dropped_engineered" in output
    (line,) = read_index(tmp_path / "results")
    assert line["status"] == "stopped" and "Run record:" in output
    assert cli_main([]) == 0  # no subcommand: help, exit 0

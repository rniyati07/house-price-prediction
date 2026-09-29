"""MLflow tracking and the M6 training run (DOC-03 §14; DOC-05 M6-4 to M6-6; DOC-01 AC-032
partial). The training run uses the 102-row fixture and a temporary MLflow store."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from house_price.cli import main as cli_main
from house_price.config import load_models_config
from house_price.data import split
from house_price.data.load import sha256_file
from house_price.models.registry import PASSTHROUGH, baseline_candidates
from house_price.models.train import TrainError, TrainResult, run_train
from house_price.results import read_index, read_result
from house_price.tracking import (
    EXPERIMENTS,
    STANDARD_TAGS,
    Lineage,
    Tracker,
    TrackingError,
    git_state,
)
from tests.conftest import CONFIG_DIR, SampleEnv, make_env

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


# ------------------------------------------------------------- M6 training run


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[SampleEnv, TrainResult, str]:
    env = make_env(tmp_path_factory.mktemp("train"))
    assert split.main(env.cli_args()) == 0
    uri = (env.root / "mlruns").as_uri()
    return env, run_train(env.config_dir, env.root, uri), uri


def _runs(uri: str) -> list:  # type: ignore[type-arg]
    client = MlflowClient(uri)
    experiment = client.get_experiment_by_name("hpp-baselines")
    assert experiment is not None
    return client.search_runs([experiment.experiment_id])


def test_both_baselines_logged_with_fold_scores_mean_and_se(
    trained: tuple[SampleEnv, TrainResult, str],
) -> None:
    _, result, uri = trained
    runs = {run.data.tags["candidate"]: run for run in _runs(uri)}
    assert set(runs) == {"dummy_median", "linear_2feat"}
    for name, run in runs.items():
        metrics = run.data.metrics
        folds = [metrics[f"fold_{i:02d}"] for i in range(1, 16)]
        assert "fold_16" not in metrics
        assert metrics["cv_mean"] == pytest.approx(sum(folds) / 15)
        assert metrics["cv_se"] == pytest.approx(result.results[name].se)
        assert {"cv_mae", "cv_mape", "cv_r2", "duration_seconds"} <= set(metrics)
        assert run.data.params["n_folds"] == "5" and run.data.params["n_repeats"] == "3"
        assert run.info.status == "FINISHED"


def test_training_runs_carry_true_lineage(trained: tuple[SampleEnv, TrainResult, str]) -> None:
    env, result, uri = trained
    config = env.load()
    commit, _ = git_state()
    for run in _runs(uri):
        tags = run.data.tags
        assert set(STANDARD_TAGS) <= set(tags)
        assert tags["pipeline_run_id"] == result.pipeline_run_id
        assert tags["git_commit"] == commit
        assert tags["data_sha256"] == config.data.raw_sha256 == sha256_file(config.raw_path)
        assert tags["split_manifest_sha256"] == sha256_file(config.manifest_path)
        assert re.fullmatch(r"[0-9a-f]{64}", tags["config_hash"])
        assert tags["seed"] == "42" and tags["stage"] == "baselines"


def test_shared_folds_are_written_for_the_development_set(
    trained: tuple[SampleEnv, TrainResult, str],
) -> None:
    env, result, _ = trained
    assert result.folds_path == env.root / "artifacts" / "cv" / "folds.json"
    stored = json.loads(result.folds_path.read_text(encoding="utf-8"))
    manifest = json.loads(env.load().manifest_path.read_text(encoding="utf-8"))
    assert stored["dev_sha256"] == manifest["dev_sha256"]
    assert len(stored["folds"]) == 15
    validated = {i for f in stored["folds"] for i in f["valid_ids"]}
    assert validated == set(manifest["dev_ids"])
    assert not validated & set(manifest["holdout_ids"])  # the holdout is never in any fold


def test_baseline_results_are_plausible(trained: tuple[SampleEnv, TrainResult, str]) -> None:
    """M6-6: the dummy scores near the SD of log price; the heuristic is clearly better."""
    _, result, _ = trained
    dummy, heuristic = result.results["dummy_median"], result.results["linear_2feat"]
    assert dummy.mean == pytest.approx(result.dev_log_price_sd, rel=0.2)
    assert heuristic.mean < dummy.mean - dummy.se


def test_m6_record_summarizes_and_links_the_mlflow_runs(
    trained: tuple[SampleEnv, TrainResult, str],
) -> None:
    """The local record uses pipeline_run_id as run_id and agrees with MLflow exactly."""
    env, result, uri = trained
    assert result.result_path is not None
    record = read_result(result.result_path)
    assert record["milestone"] == "M6" and record["status"] == "succeeded"
    assert record["run_id"] == result.pipeline_run_id
    assert result.result_path.parent.parent == env.root / "results" / "M6"
    mlflow = record["findings"]["mlflow"]
    assert mlflow["tracking_uri"] == uri and mlflow["experiment"] == "hpp-baselines"
    assert mlflow["runs"] == result.run_ids
    for run in _runs(uri):
        name, metrics, tags = run.data.tags["candidate"], run.data.metrics, run.data.tags
        assert run.info.run_id == mlflow["runs"][name]
        for key in ("cv_mean", "cv_se", "cv_mae", "cv_mape", "cv_r2"):
            assert record["metrics"][f"{name}.{key}"] == pytest.approx(metrics[key], rel=1e-12)
        assert tags["pipeline_run_id"] == record["run_id"]
        assert tags["git_commit"] == record["git"]["commit"]
        for key in ("data_sha256", "split_manifest_sha256", "config_hash"):
            assert tags[key] == record["lineage"][key]
    assert not any(key.split(".")[-1].startswith("fold_") for key in record["metrics"])
    assert record["metrics"]["dev_log_price_sd"] == pytest.approx(result.dev_log_price_sd)
    plausibility = record["findings"]["plausibility"]
    dummy = result.results["dummy_median"]
    assert plausibility["dummy_cv_mean_over_dev_log_price_sd"] == pytest.approx(
        dummy.mean / result.dev_log_price_sd
    )
    assert plausibility["heuristic_beats_dummy_by_more_than_one_se"] is True
    folds = next(a for a in record["artifacts"] if a["path"] == "artifacts/cv/folds.json")
    assert folds["sha256"] == sha256_file(result.folds_path)
    assert read_index(env.root / "results")[-1]["run_id"] == result.pipeline_run_id


def test_a_failed_training_run_is_recorded(tmp_path: Path) -> None:
    env = make_env(tmp_path)  # no split: training refuses to start
    with pytest.raises(TrainError):
        run_train(env.config_dir, env.root, (tmp_path / "mlruns").as_uri())
    (line,) = read_index(tmp_path / "results")
    record = read_result(tmp_path / "results" / line["result"])
    assert line["milestone"] == "M6" and record["status"] == "failed"
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


def test_cli_train_runs_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = make_env(tmp_path)
    assert split.main(env.cli_args()) == 0
    capsys.readouterr()
    code = cli_main(["train", *env.cli_args(), "--tracking-uri", (tmp_path / "mlruns").as_uri()])
    output = capsys.readouterr().out
    assert code == 0
    assert "dummy_median" in output and "linear_2feat" in output and "hpp-baselines" in output
    assert "Run record:" in output and len(read_index(tmp_path / "results")) == 1
    assert cli_main([]) == 0  # no subcommand: help, exit 0

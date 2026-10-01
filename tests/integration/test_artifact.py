"""Artifact persistence, production refit and verified load (DOC-03 §15; DOC-05 M10-1 to
M10-3, M10-5; AC-042, AC-044 to AC-046) on the smoke refit of the synthetic project."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from house_price.config import load_project_config
from house_price.data.load import load_raw, sha256_file
from house_price.data.schema import select_model_input
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.persistence import artifact
from house_price.persistence.artifact import ArtifactError
from house_price.tracking import git_state
from tests.conftest import SmokeProject


def _model_dir(project: SmokeProject) -> Path:
    return project.root / "artifacts/smoke/model"


def _sample(project: SmokeProject) -> object:
    config = load_project_config(project.root / "configs", project.root)
    in_scope, scope = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    dev = create_or_load_split(in_scope, scope, config, verify_holdout_file=False).dev
    ids = json.loads((project.root / "artifacts/smoke/smoke_split.json").read_text())["sample_ids"]
    return select_model_input(dev.set_index("Id", drop=False).loc[ids], config.schema)


def test_smoke_refit_writes_one_model_and_one_metadata(smoke_project: SmokeProject) -> None:
    directory = _model_dir(smoke_project)
    assert sorted(p.name for p in directory.iterdir()) == ["metadata.json", "model.joblib"]
    meta = artifact.read_metadata(directory)
    assert meta.empty_fields() == []  # AC-044 (staging: version unreleased, release null)
    assert meta.model_sha256 == sha256_file(directory / "model.joblib")
    assert (meta.is_release, meta.model_version, meta.mlflow.release) == (False, "unreleased", None)


def test_refit_uses_the_smoke_sample_rows(smoke_project: SmokeProject) -> None:
    """AC-042 logic: the refit trains on all rows passed to it (the 200-row smoke sample;
    the 100-row holdout substitute stays separate)."""
    meta = artifact.read_metadata(_model_dir(smoke_project))
    info = json.loads((smoke_project.root / "artifacts/smoke/smoke_split.json").read_text())
    assert meta.training_rows == len(info["sample_ids"]) == 200
    refit = smoke_project.evaluation["production_refit"]
    assert refit["training_rows"] == 200 and refit["round_trip"]["passed"] is True  # type: ignore[index]
    assert refit["round_trip"]["max_abs_difference"] == 0.0  # type: ignore[index]


def test_refit_uses_the_selected_configuration(smoke_project: SmokeProject) -> None:
    meta = artifact.read_metadata(_model_dir(smoke_project))
    record = smoke_project.train.selection_record
    assert record is not None
    assert meta.selected_candidate == record["selected"]["name"]
    assert meta.hyperparameters == record["selected"]["hyperparameters"]
    assert meta.selection_record_id == record["selection_record_id"]
    assert meta.cv["mean"] == record["selected"]["mean"]
    if meta.selected_candidate == "blend":
        hp = record["selected"]["hyperparameters"]
        assert set(meta.transformed_feature_names) == {hp["linear"], hp["tree"]}
        assert set(meta.feature_sets) == {hp["linear"], hp["tree"]}


def test_metadata_provenance_is_correct(smoke_project: SmokeProject) -> None:
    """AC-045: data hash = committed raw-data hash; git SHA = the producing commit."""
    meta = artifact.read_metadata(_model_dir(smoke_project))
    config = load_project_config(smoke_project.root / "configs", smoke_project.root)
    assert meta.data_sha256 == config.data.raw_sha256 == sha256_file(config.raw_path)
    assert meta.split_manifest_sha256 == sha256_file(config.manifest_path)
    commit, _ = git_state()
    assert meta.git_commit == commit
    assert meta.library_versions == artifact.library_versions()
    assert meta.target_transform == {"func": "log1p", "inverse_func": "expm1"}
    assert len(meta.input_schema) == 77 and meta.scope_rule.max_in_domain == 4000


def test_refit_run_is_logged_as_production_refit(smoke_project: SmokeProject) -> None:
    from mlflow.tracking import MlflowClient

    meta = artifact.read_metadata(_model_dir(smoke_project))
    client = MlflowClient((smoke_project.root / "mlruns").resolve().as_uri())
    run = client.get_run(meta.mlflow.refit)
    assert run.data.tags["run_kind"] == "production_refit"
    assert client.get_experiment(run.info.experiment_id).name == "hpp-smoke"
    final = client.get_run(meta.mlflow.final_holdout_evaluation)
    assert final.data.tags["run_kind"] == "final_holdout_evaluation"


def test_verified_load_gives_identical_predictions(smoke_project: SmokeProject) -> None:
    X = _sample(smoke_project)
    first, meta = artifact.load_verified(_model_dir(smoke_project))
    second, _ = artifact.load_verified(_model_dir(smoke_project))
    assert np.array_equal(first.predict(X), second.predict(X))
    assert meta.training_rows == 200


def _copy(project: SmokeProject, tmp_path: Path) -> Path:
    target = tmp_path / "model"
    shutil.copytree(_model_dir(project), target)
    return target


def _no_unpickling(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("joblib.load must not be called before verification")

    monkeypatch.setattr(artifact.joblib, "load", refuse)


def test_tampered_artifact_is_refused_before_loading(
    smoke_project: SmokeProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _copy(smoke_project, tmp_path)
    with (target / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    _no_unpickling(monkeypatch)
    with pytest.raises(ArtifactError, match="does not match"):
        artifact.load_verified(target)


def test_library_mismatch_is_refused_before_loading(
    smoke_project: SmokeProject, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = _copy(smoke_project, tmp_path)
    meta = artifact.read_metadata(target)
    versions = {**meta.library_versions, "scikit-learn": "0.0.1"}
    artifact.write_metadata(meta.model_copy(update={"library_versions": versions}), target)
    _no_unpickling(monkeypatch)
    with pytest.raises(ArtifactError, match="library versions differ"):
        artifact.load_verified(target)


def test_release_is_required_when_asked(smoke_project: SmokeProject, tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not a released artifact"):
        artifact.load_verified(_copy(smoke_project, tmp_path), require_release=True)


def test_missing_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="metadata not found"):
        artifact.verify(tmp_path)


def test_round_trip_detects_a_different_model(tmp_path: Path) -> None:
    import pandas as pd

    X = pd.DataFrame({"x": np.arange(1.0, 11.0)})
    y = np.exp(10 + 0.1 * X["x"])
    fitted = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    fitted.fit(X, y)
    artifact.save_model(fitted, tmp_path)
    assert artifact.round_trip(fitted, tmp_path / "model.joblib", X).passed
    other = TransformedTargetRegressor(LinearRegression(), func=np.log1p, inverse_func=np.expm1)
    other.fit(X, y * 1.1)
    result = artifact.round_trip(other, tmp_path / "model.joblib", X)
    assert not result.passed and result.max_abs_difference > 0


# ------------------------------------------- candidate artifacts (additional requirement)

TUNED = {"ridge": "Ridge", "lasso": "Lasso", "random_forest": "RandomForestRegressor",
         "lightgbm": "LGBMRegressor"}  # fmt: skip


def _candidate_dir(project: SmokeProject, name: str) -> Path:
    return project.root / "artifacts/smoke/candidates" / name


def test_all_four_candidate_artifacts_are_written(smoke_project: SmokeProject) -> None:
    root = smoke_project.root / "artifacts/smoke/candidates"
    assert sorted(p.name for p in root.iterdir()) == sorted(TUNED)
    for name in TUNED:
        files = sorted(p.name for p in _candidate_dir(smoke_project, name).iterdir())
        assert files == ["metadata.json", "model.joblib"], name


@pytest.mark.parametrize("name", list(TUNED))
def test_candidate_metadata_describes_that_candidate(
    smoke_project: SmokeProject, name: str
) -> None:
    directory = _candidate_dir(smoke_project, name)
    meta = artifact.read_metadata(directory)
    assert meta.artifact_role == "candidate" and meta.empty_fields() == []
    assert meta.model_sha256 == sha256_file(directory / "model.joblib")
    assert meta.selected_candidate == name and meta.training_rows == 200
    assert (meta.holdout, meta.baseline_reference, meta.temporal_diagnostic,
            meta.quality_gates, meta.mlflow.final_holdout_evaluation) == (None,) * 5  # fmt: skip
    assert (meta.is_release, meta.model_version) == (False, "unreleased")
    best_path = smoke_project.root / f"artifacts/smoke/tuning/{name}_best.json"
    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert meta.hyperparameters["tuned_params"] == best["best_params"]
    record = smoke_project.train.selection_record
    assert record is not None
    scores = next(c for c in record["candidates"] if c["name"] == name)
    assert meta.cv["mean"] == scores["mean"] and meta.cv["se"] == scores["se"]
    assert set(meta.feature_sets) == set(meta.transformed_feature_names) == {name}
    production = artifact.read_metadata(_model_dir(smoke_project))
    for shared in ("data_sha256", "split_manifest_sha256", "config_hash", "git_commit",
                   "selection_record_id", "schema_hash", "library_versions"):  # fmt: skip
        assert getattr(meta, shared) == getattr(production, shared), shared
    assert meta.mlflow.refit == production.mlflow.refit


@pytest.mark.parametrize("name", list(TUNED))
def test_candidate_loads_verified_and_predicts(smoke_project: SmokeProject, name: str) -> None:
    model, meta = artifact.load_verified(_candidate_dir(smoke_project, name))
    estimator = model.regressor_.named_steps["model"]
    assert type(estimator).__name__ == TUNED[name]
    params = estimator.get_params()
    assert all(params[k] == v for k, v in meta.hyperparameters["tuned_params"].items())
    predictions = model.predict(_sample(smoke_project))
    assert np.isfinite(predictions).all() and (predictions > 0).all()
    refit = smoke_project.evaluation["candidate_refits"][name]  # type: ignore[index]
    assert refit["round_trip"]["passed"] is True
    assert refit["round_trip"]["max_abs_difference"] == 0


def test_candidates_are_never_releasable(smoke_project: SmokeProject) -> None:
    for name in TUNED:
        with pytest.raises(ArtifactError, match="not a released artifact"):
            artifact.load_verified(_candidate_dir(smoke_project, name), require_release=True)


def test_the_production_artifact_is_independent_of_the_candidates(
    smoke_project: SmokeProject, tmp_path: Path
) -> None:
    """The selected model loads on its own: copied alone, with no candidate files present."""
    target = _copy(smoke_project, tmp_path)
    model, meta = artifact.load_verified(target)
    assert meta.artifact_role == "production"
    assert np.isfinite(model.predict(_sample(smoke_project))).all()
    assert sorted(p.name for p in target.iterdir()) == ["metadata.json", "model.joblib"]


def test_only_final_artifacts_are_persisted(smoke_project: SmokeProject) -> None:
    """No CV-fold, Optuna-trial, tuning or ablation model is ever written."""
    pickles = sorted(
        p.relative_to(smoke_project.root).as_posix()
        for p in smoke_project.root.rglob("*.joblib")
        if "mlruns" not in p.parts
    )
    expected = ["artifacts/smoke/model/model.joblib",
                *[f"artifacts/smoke/candidates/{n}/model.joblib" for n in sorted(TUNED)]]  # fmt: skip
    assert pickles == sorted(expected)


def test_random_forest_round_trip_is_exact_and_leaves_the_model_unchanged(tmp_path: Path) -> None:
    """Parallel tree summation (``n_jobs=-1``) must not make the exact round trip flaky."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(300, 4)))
    forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    model = TransformedTargetRegressor(forest, func=np.log1p, inverse_func=np.expm1)
    model.fit(X, np.exp(10 + X[0]))
    artifact.save_model(model, tmp_path)
    assert all(artifact.round_trip(model, tmp_path / "model.joblib", X).passed for _ in range(3))
    assert model.regressor_.n_jobs == -1  # the caller's model is not modified

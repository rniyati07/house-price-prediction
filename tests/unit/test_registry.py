"""The candidate registry (DOC-03 §8.1, §8.4 to §8.7, §6.6, §10; DOC-05 M7-1).

Expected values are DOC-03's: branches and tiers (§8.1, DN-08), Lasso ``max_iter`` (§8.5),
Random Forest fixed settings (§10.5), LightGBM determinism (DN-14) and ablation reference
parameters (§6.6), and the grids (§10.3, §10.4).
"""

from __future__ import annotations

import numpy as np
import pytest
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from house_price.config import ConfigError, ModelsConfig, load_models_config
from house_price.models.registry import model_candidates, registry
from tests.conftest import CONFIG_DIR, ModelData

SEED = 42


@pytest.fixture(scope="module")
def models() -> ModelsConfig:
    return load_models_config(CONFIG_DIR)


def test_registry_holds_the_baselines_then_the_four_candidates(models: ModelsConfig) -> None:
    assert list(registry(models, SEED)) == [
        "dummy_median", "linear_2feat", "ridge", "lasso", "random_forest", "lightgbm",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "branch", "tier", "tuning"),
    [
        ("ridge", "linear", 1, "grid"),
        ("lasso", "linear", 1, "grid"),
        ("random_forest", "tree", 2, "optuna"),
        ("lightgbm", "tree", 3, "optuna"),
    ],
)
def test_branch_tier_and_tuning(
    models: ModelsConfig, name: str, branch: str, tier: int, tuning: str
) -> None:
    candidate = registry(models, SEED)[name]
    assert (candidate.branch, candidate.tier, candidate.tuning) == (branch, tier, tuning)
    assert candidate.eligible


def test_baselines_stay_ineligible(models: ModelsConfig) -> None:
    candidates = registry(models, SEED)
    assert not candidates["dummy_median"].eligible and not candidates["linear_2feat"].eligible
    assert candidates["dummy_median"].tier is None


def test_reference_configurations_match_doc03(models: ModelsConfig) -> None:
    c = registry(models, SEED)
    assert dict(c["ridge"].params) == {}  # alpha comes from the ablation grid (§6.6 step 2)
    assert dict(c["lasso"].params) == {"max_iter": 50000, "alpha": 1e-3, "random_state": SEED}
    assert dict(c["random_forest"].params) == {
        "n_jobs": -1, "bootstrap": True, "random_state": SEED,
    }  # fmt: skip
    assert dict(c["lightgbm"].params) == {
        "deterministic": True, "force_row_wise": True, "n_jobs": 1, "verbose": -1,
        "learning_rate": 0.05, "n_estimators": 600, "num_leaves": 15, "min_child_samples": 20,
        "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.7, "reg_alpha": 0.0,
        "reg_lambda": 0.0, "random_state": SEED,
    }  # fmt: skip


def test_grids_match_doc03(models: ModelsConfig) -> None:
    c = registry(models, SEED)
    ridge, lasso = c["ridge"].grid, c["lasso"].grid
    assert ridge is not None and lasso is not None
    assert (ridge.param, ridge.low, ridge.high, ridge.n) == ("alpha", 1e-3, 1e3, 25)
    assert (lasso.param, lasso.low, lasso.high, lasso.n) == ("alpha", 1e-5, 1e-1, 25)
    assert c["random_forest"].grid is None and c["lightgbm"].grid is None


def test_seed_comes_from_the_caller(models: ModelsConfig) -> None:
    for candidate in model_candidates(models, 7):
        if candidate.name != "ridge":
            assert candidate.params["random_state"] == 7


def test_ablation_references_are_ridge_and_lightgbm(models: ModelsConfig) -> None:
    assert (models.ablation.linear, models.ablation.tree) == ("ridge", "lightgbm")


@pytest.mark.parametrize(
    ("name", "estimator_class", "encoder"),
    [
        ("ridge", Ridge, OneHotEncoder),
        ("lasso", Lasso, OneHotEncoder),
        ("random_forest", RandomForestRegressor, OrdinalEncoder),
        ("lightgbm", LGBMRegressor, OrdinalEncoder),
    ],
)
def test_each_candidate_builds_its_branch_pipeline(
    models: ModelsConfig, model_data: ModelData, name: str, estimator_class: type,
    encoder: type,
) -> None:  # fmt: skip
    model = registry(models, SEED)[name].build(model_data.features, model_data.schema)
    steps = model.regressor.named_steps
    assert isinstance(steps["model"], estimator_class)
    nominal = {n: t for n, t, _ in steps["preprocess"].transformers}["nominal"]
    assert isinstance(nominal.named_steps["encode"], encoder)
    assert model.func is np.log1p and model.inverse_func is np.expm1


def test_overrides_replace_estimator_parameters(
    models: ModelsConfig, model_data: ModelData
) -> None:
    c = registry(models, SEED)
    ridge = c["ridge"].build(model_data.features, model_data.schema, {"alpha": 3.5})
    assert ridge.regressor.named_steps["model"].alpha == 3.5
    lasso = c["lasso"].build(model_data.features, model_data.schema)
    assert lasso.regressor.named_steps["model"].get_params()["max_iter"] == 50000
    with pytest.raises(ValueError, match="fixed baseline"):
        c["dummy_median"].build(model_data.features, model_data.schema, {"strategy": "mean"})


@pytest.mark.parametrize("name", ["ridge", "lasso", "random_forest", "lightgbm"])
def test_every_candidate_fits_and_predicts_finite_dollars(
    models: ModelsConfig, model_data: ModelData, name: str
) -> None:
    overrides = {"alpha": 10.0} if name == "ridge" else {}
    model = registry(models, SEED)[name].build(model_data.features, model_data.schema, overrides)
    predictions = model.fit(model_data.X, model_data.y).predict(model_data.X)
    assert np.isfinite(predictions).all() and (predictions > 0).all()


def test_registry_is_deterministic(models: ModelsConfig) -> None:
    first, second = registry(models, SEED), registry(models, SEED)
    assert list(first) == list(second)
    for name in first:
        assert dict(first[name].params) == dict(second[name].params)
        assert first[name].tier == second[name].tier and first[name].grid == second[name].grid


def test_invalid_candidate_config_is_refused(tmp_path: object) -> None:
    import yaml

    from tests.conftest import make_env

    env = make_env(tmp_path)  # type: ignore[arg-type]
    path = env.config_dir / "models.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["candidates"]["ridge"]["branch"] = "forest"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="branch"):
        load_models_config(env.config_dir)

"""Per-branch feature ablation and the RC-02 outcome check (DOC-03 §6.6, DN-05, DOC-01 IN-04,
DOC-05 M7-2 to M7-4, RC-02)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from house_price.config import (
    ConfigError,
    FeatureConfig,
    SchemaConfig,
    load_feature_config,
)
from house_price.data.load import load_raw, sha256_file
from house_price.data.scope import apply_scope_rule
from house_price.evaluation.cv import FoldPlan, generate_folds
from house_price.models import ablation
from house_price.models.ablation import check_outcome, is_retained, with_dropped
from house_price.models.registry import Candidate
from house_price.models.train import run_train
from house_price.pipelines.branches import Branch, branch_groups
from house_price.pipelines.build import wrap_estimator
from house_price.results import read_index, read_result
from tests.conftest import (
    CONFIG_DIR,
    FAST_FOLDS,
    FAST_REPEATS,
    REPO_ROOT,
    M7Train,
    make_env,
    set_ablation_outcome,
    use_fast_models,
)

ENGINEERED = [
    "TotalSF", "TotalBath", "HouseAge", "RemodAge", "IsRemodeled", "TotalPorchSF",
    "HasPool", "HasGarage", "HasBsmt", "HasFireplace", "Has2ndFlr", "GarageAge",
]  # fmt: skip


@pytest.fixture
def features() -> FeatureConfig:
    """The real features.yaml with its committed outcome removed: these tests exercise the
    behavior with and without an outcome explicitly."""
    config = load_feature_config(CONFIG_DIR)
    return config.model_copy(update={
        branch: getattr(config, branch).model_copy(update={"dropped_engineered": None})
        for branch in ("linear", "tree")
    })  # fmt: skip


def test_committed_outcome_is_the_e30_outcome() -> None:
    """DOC-05 M7-3/M7-4: features.yaml holds, per branch, exactly the features E-30 drops,
    so the committed outcome passes RC-02 against the saved ablation table."""
    committed = load_feature_config(CONFIG_DIR)
    assert committed.has_ablation_outcome
    table = pd.read_csv(REPO_ROOT / ablation.E30_PATH)
    proposed = {branch: list(rows.loc[~rows["retained"].astype(bool), "feature"])
                for branch, rows in table.groupby("branch", sort=False)}  # fmt: skip
    assert check_outcome(committed, proposed).status == "match"
    assert committed.linear.dropped_engineered == proposed["linear"]
    assert committed.tree.dropped_engineered == proposed["tree"]


# ------------------------------------------------------------------ retention rule


def test_in04_drops_only_when_removal_lowers_the_score() -> None:
    assert is_retained(0.20, 0.21)  # removal hurts -> kept
    assert is_retained(0.20, 0.20)  # tie -> kept (the burden of proof is on removal)
    assert not is_retained(0.20, 0.19)  # removal helps -> dropped


# ----------------------------------------------------- branch-specific configuration


def test_ablation_variant_changes_only_its_branch(features: FeatureConfig) -> None:
    variant = with_dropped(features, "linear", ["TotalSF"])
    assert "TotalSF" in branch_groups("linear", variant).dropped
    assert "TotalSF" not in branch_groups("linear", variant).numeric
    assert branch_groups("tree", variant) == branch_groups("tree", features)
    assert features.linear.dropped_engineered is None  # the original is never changed


def test_committed_outcome_moves_features_to_dropped(features: FeatureConfig) -> None:
    committed = features.model_copy(update={
        "linear": features.linear.model_copy(update={"dropped_engineered": ["HasPool"]}),
        "tree": features.tree.model_copy(update={"dropped_engineered": []}),
    })  # fmt: skip
    assert committed.has_ablation_outcome and not features.has_ablation_outcome
    linear = branch_groups("linear", committed)
    assert "HasPool" in linear.dropped and "HasPool" not in linear.numeric
    assert branch_groups("tree", committed) == committed.tree  # [] = outcome, nothing dropped
    assert branch_groups("tree", features) == features.tree  # absent = no outcome
    assert sorted(linear.all_columns) == sorted(features.linear.all_columns)  # coverage kept


@pytest.mark.parametrize(
    ("dropped", "message"),
    [(["NotAFeature"], "must be in an active group"), (["TotalSF", "TotalSF"], "twice"),
     (["GarageYrBlt"], "must be in an active group")],
)  # fmt: skip
def test_invalid_outcomes_are_refused(tmp_path: Path, dropped: list[str], message: str) -> None:
    env = make_env(tmp_path)
    path = env.config_dir / "features.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["linear"]["dropped_engineered"] = dropped
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_feature_config(env.config_dir)


# ------------------------------------------------ synthetic ground truth (DOC-05 M7)


def _knn_reference(branch: Branch) -> Candidate:
    """A reference model that sees only the branch's active engineered features (scaled),
    so the ablation's effect on it is fully determined by the synthetic data."""

    def factory(
        features: FeatureConfig, schema: SchemaConfig, overrides: object
    ) -> TransformedTargetRegressor:
        groups = branch_groups(branch, features)
        active = [f for f in features.engineered
                  if f in groups.numeric + groups.ordinal + groups.nominal]  # fmt: skip
        preprocess = ColumnTransformer([("engineered", StandardScaler(), active)],
                                       remainder="drop").set_output(transform="pandas")  # fmt: skip
        return wrap_estimator(preprocess, KNeighborsRegressor(n_neighbors=5), features)

    return Candidate(name=f"knn_{branch}", branch=branch, tuning=None, tier=None,
                     eligible=False, params={}, factory=factory)  # type: ignore[arg-type]  # fmt: skip


@pytest.fixture(scope="module")
def synthetic() -> tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig]:
    """Fixture rows rewritten so that only two engineered features vary: TotalSF drives the
    price, HasPool is random noise; every other engineered feature is constant."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        config = make_env(Path(tmp)).load()
        dev, _ = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    rng = np.random.default_rng(0)
    n = len(dev)
    dev = dev.copy()
    constants = {"YearBuilt": 1990, "YearRemodAdd": 1990, "YrSold": 2008, "GarageYrBlt": 1990,
                 "FullBath": 2, "HalfBath": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0,
                 "OpenPorchSF": 0, "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
                 "Fireplaces": 1, "2ndFlrSF": 0}  # fmt: skip
    for column, value in constants.items():
        dev[column] = value
    dev["GarageType"] = "Attchd"
    dev["TotalBsmtSF"] = rng.integers(400, 1500, n)
    dev["1stFlrSF"] = rng.integers(600, 1800, n)
    dev["PoolArea"] = rng.choice([0, 500], n)  # HasPool: pure noise
    total = dev["TotalBsmtSF"] + dev["1stFlrSF"]
    dev["SalePrice"] = np.exp(11.5 + 0.0008 * (total - total.mean()) + rng.normal(0, 0.01, n))
    plan = generate_folds(dev, id_column="Id", target="SalePrice", n_bins=10, n_splits=5,
                          n_repeats=3, seed=42, dev_sha256="0" * 64)  # fmt: skip
    return dev, plan, load_feature_config(CONFIG_DIR), config.schema


@pytest.mark.parametrize("branch", ["linear", "tree"])
def test_rule_keeps_the_informative_feature_and_drops_the_noise(
    synthetic: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig], branch: Branch
) -> None:
    dev, plan, features, schema = synthetic
    result = ablation.run_branch_ablation(branch, _knn_reference(branch), dev, plan, features,
                                          schema)  # fmt: skip
    table = result.table().set_index("feature")
    assert list(table.index) == ENGINEERED  # all 12, one leave-one-out run each
    assert table.loc["TotalSF", "delta"] > 0 and table.loc["TotalSF", "retained"]
    assert table.loc["HasPool", "delta"] < 0 and not table.loc["HasPool", "retained"]
    constant = [f for f in ENGINEERED if f not in ("TotalSF", "HasPool")]
    assert (table.loc[constant, "delta"] == 0).all() and table.loc[constant, "retained"].all()
    assert result.dropped == ["HasPool"]
    assert (table["mean_with"] == result.full.mean).all()
    assert np.allclose(table["delta"], table["mean_without"] - table["mean_with"])


def test_reference_must_match_the_branch(
    synthetic: tuple[pd.DataFrame, FoldPlan, FeatureConfig, SchemaConfig],
) -> None:
    dev, plan, features, schema = synthetic
    with pytest.raises(ValueError, match="not tree"):
        ablation.run_branch_ablation("tree", _knn_reference("linear"), dev, plan, features, schema)


# ------------------------------------------------------------------------- RC-02


def test_rc02_missing_match_and_differs(features: FeatureConfig) -> None:
    proposed = {"linear": ["HasPool"], "tree": []}
    missing = check_outcome(features, proposed)
    assert missing.status == "missing" and not missing.passed
    assert "no committed ablation outcome" in missing.message()

    def committed(linear: list[str] | None, tree: list[str] | None) -> FeatureConfig:
        return features.model_copy(update={
            "linear": features.linear.model_copy(update={"dropped_engineered": linear}),
            "tree": features.tree.model_copy(update={"dropped_engineered": tree}),
        })  # fmt: skip

    assert check_outcome(committed(["HasPool"], None), proposed).status == "missing"  # partial
    match = check_outcome(committed(["HasPool"], []), proposed)
    assert match.status == "match" and match.passed and match.differences == {}
    differs = check_outcome(committed(["TotalBath"], []), proposed)
    assert differs.status == "differs" and not differs.passed
    assert differs.differences == {
        "linear": {"newly_dropped": ["HasPool"], "no_longer_dropped": ["TotalBath"]}
    }
    assert "differs" in differs.message() and "TotalBath" in differs.message()


def test_first_run_without_outcome_stops_after_ablation(m7_train: M7Train) -> None:
    first = m7_train.first
    assert first.stopped and first.outcome_check is not None
    assert first.outcome_check.status == "missing"
    assert first.dev_checks == {}  # no development check before a committed outcome
    assert set(first.ablation) == {"linear", "tree"}
    record = read_result(first.result_path)  # type: ignore[arg-type]
    assert record["status"] == "stopped" and record["error"] is None
    assert "RC-02" in record["stop_reason"]


def test_second_run_with_matching_outcome_continues(m7_train: M7Train) -> None:
    second = m7_train.second
    assert not second.stopped and second.outcome_check is not None
    assert second.outcome_check.status == "match"
    assert set(second.dev_checks) == {"ridge", "lasso", "random_forest", "lightgbm"}
    for result in second.dev_checks.values():
        assert len(result.fold_scores) == FAST_FOLDS * FAST_REPEATS
        assert all(np.isfinite([result.mean, result.se, *result.secondary.values()]))
    assert read_result(second.result_path)["status"] == "succeeded"  # type: ignore[arg-type]


def test_ablation_is_deterministic_and_branch_specific(m7_train: M7Train) -> None:
    first, second = m7_train.first, m7_train.second
    for branch in ("linear", "tree"):
        pd.testing.assert_frame_equal(first.ablation[branch].table(),
                                      second.ablation[branch].table())  # fmt: skip
    assert first.ablation["linear"].reference == "ridge"
    assert first.ablation["tree"].reference == "lightgbm"
    grid = first.ablation["linear"].grid
    assert (
        grid is not None and first.ablation["linear"].reference_params["alpha"] == grid.best_value
    )
    assert first.ablation["tree"].grid is None


def test_ablation_uses_the_shared_folds(m7_train: M7Train) -> None:
    env = m7_train.env
    plan = FoldPlan.model_validate_json(
        (env.root / "artifacts" / "cv" / "folds.json").read_text(encoding="utf-8")
    )
    folds = [(f.repeat, f.fold) for f in plan.folds]
    for result in m7_train.first.ablation.values():
        for cv in (result.full, *result.without.values()):
            assert len(cv.fold_scores) == len(plan.folds)
            assert list(cv.oof.drop_duplicates(["repeat", "fold"])[["repeat", "fold"]]
                        .itertuples(index=False, name=None)) == folds  # fmt: skip


def test_e30_is_the_ablation_output(m7_train: M7Train) -> None:
    path = m7_train.env.root / ablation.E30_PATH
    saved = pd.read_csv(path)
    assert list(saved.columns) == ablation.TABLE_COLUMNS
    assert len(saved) == 2 * len(ENGINEERED)
    assert list(saved["branch"].unique()) == ["linear", "tree"]
    expected = ablation.ablation_table(m7_train.second.ablation)
    assert np.allclose(saved["delta"], expected["delta"], rtol=1e-9, atol=1e-15)
    assert list(saved["retained"]) == list(expected["retained"])
    proposed = m7_train.first.outcome_check.proposed  # type: ignore[union-attr]
    for branch in ("linear", "tree"):
        rows = saved[saved["branch"] == branch]
        assert sorted(rows.loc[~rows["retained"], "feature"]) == sorted(proposed[branch])


def test_changed_outcome_stops_and_is_never_rewritten(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    use_fast_models(env)
    from house_price.data import split

    assert split.main(env.cli_args()) == 0
    set_ablation_outcome(env, {"linear": ["TotalSF", "HouseAge"], "tree": ["TotalSF"]})
    features_yaml = env.config_dir / "features.yaml"
    before = sha256_file(features_yaml)
    result = run_train(env.config_dir, env.root, (tmp_path / "mlruns").as_uri())
    assert result.stopped and result.outcome_check is not None
    assert result.outcome_check.status == "differs"
    assert result.outcome_check.differences  # reported per branch
    assert result.dev_checks == {}
    assert sha256_file(features_yaml) == before  # never rewritten
    (line,) = read_index(tmp_path / "results")
    assert line["status"] == "stopped"


def test_training_modules_never_reference_the_holdout() -> None:
    """Only the development set reaches the ablation, tuning, and training code (AC-030)."""
    models = REPO_ROOT / "src" / "house_price" / "models"
    for path in models.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"holdout_path|holdout\.csv|\.holdout\b", text), path.name

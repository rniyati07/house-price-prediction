"""Candidate registry (ADR-10, FR-022, DOC-03 §8.1).

Each entry knows its name, preprocessing branch, how to build its unfitted pipeline,
its tuning method, its simplicity tier, and whether it may be selected. The training
orchestrator only iterates over entries; it holds no model-specific logic.

M6 registers the two baselines. They define the floor [ADR-10] and are never eligible
for selection. M7 adds Ridge, Lasso, Random Forest, and LightGBM with their branches,
tiers (DN-08), fixed settings, and reference configurations from ``models.yaml``; M8 adds
their search spaces and the DN-07 blend, built from the tuned components by
:func:`blend_candidate`. The global seed (DN-02) is passed to every seeded estimator as
``random_state``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from house_price.config import (
    CandidateConfig,
    FeatureConfig,
    GridConfig,
    ModelsConfig,
    SchemaConfig,
    SearchParam,
)
from house_price.pipelines.branches import (
    branch_groups,
    build_passthrough_transformer,
    feature_engineer_output,
)
from house_price.pipelines.build import build_pipeline, wrap_estimator

Overrides = Mapping[str, object]
PipelineFactory = Callable[[FeatureConfig, SchemaConfig, Overrides], TransformedTargetRegressor]
PASSTHROUGH = "two-column pass-through"
BOTH = "both"  # the blend: each component keeps its own branch
BLEND = "blend"
BLEND_WEIGHTS = (0.5, 0.5)  # DN-07: fixed equal weights, never fitted

# The four candidate estimators (DOC-03 §8.4 to §8.7) and which of them take the seed.
ESTIMATORS: dict[str, type[BaseEstimator]] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "random_forest": RandomForestRegressor,
    "lightgbm": LGBMRegressor,
}
SEEDED = ("lasso", "random_forest", "lightgbm")


@dataclass(frozen=True)
class Candidate:
    """One registry entry (DOC-03 §8.1)."""

    name: str
    branch: str
    tuning: str | None
    tier: int | None
    eligible: bool
    params: Mapping[str, object]
    factory: PipelineFactory = field(repr=False)
    grid: GridConfig | None = None
    search_space: Mapping[str, SearchParam] | None = None
    n_trials: int | None = None

    def build(
        self,
        feature_config: FeatureConfig,
        schema: SchemaConfig,
        overrides: Overrides | None = None,
    ) -> TransformedTargetRegressor:
        """A fresh, unfitted raw-to-dollars pipeline for this candidate.

        ``overrides`` replace estimator parameters (a grid point, or Ridge's reference
        ``alpha`` chosen by the ablation grid); baselines take none.
        """
        return self.factory(feature_config, schema, overrides or {})

    def engineered_features(self, feature_config: FeatureConfig) -> list[str]:
        """Engineered features this candidate's preprocessing consumes (for logging)."""
        if self.branch == PASSTHROUGH:
            return [f for f in feature_config.engineered if f in self.params["features"]]  # type: ignore[operator]
        branches = ("linear", "tree") if self.branch == BOTH else (self.branch,)
        active: set[str] = set()
        for branch in branches:
            groups = branch_groups(branch, feature_config)  # type: ignore[arg-type]
            active |= set(groups.numeric + groups.ordinal + groups.nominal)
        return [f for f in feature_config.engineered if f in active]


def _dummy_median(models: ModelsConfig) -> Candidate:
    spec = models.baselines.dummy_median

    def factory(
        feature_config: FeatureConfig, schema: SchemaConfig, overrides: Overrides
    ) -> TransformedTargetRegressor:
        _no_overrides("dummy_median", overrides)
        return build_pipeline(
            DummyRegressor(strategy=spec.strategy), spec.branch, feature_config, schema
        )

    return Candidate(
        name="dummy_median", branch=spec.branch, tuning=None, tier=None, eligible=False,
        params={"estimator": "DummyRegressor", "strategy": spec.strategy}, factory=factory,
    )  # fmt: skip


def _linear_2feat(models: ModelsConfig) -> Candidate:
    spec = models.baselines.linear_2feat

    def factory(
        feature_config: FeatureConfig, schema: SchemaConfig, overrides: Overrides
    ) -> TransformedTargetRegressor:
        _no_overrides("linear_2feat", overrides)
        available = set(feature_engineer_output(schema, feature_config))
        unknown = [f for f in spec.features if f not in available]
        if unknown:
            raise ValueError(f"linear_2feat: features not produced by the pipeline: {unknown}")
        preprocess = build_passthrough_transformer(list(spec.features))
        return wrap_estimator(preprocess, LinearRegression(), feature_config)

    return Candidate(
        name="linear_2feat", branch=PASSTHROUGH, tuning=None, tier=None, eligible=False,
        params={"estimator": "LinearRegression", "features": list(spec.features)}, factory=factory,
    )  # fmt: skip


def _no_overrides(name: str, overrides: Overrides) -> None:
    if overrides:
        raise ValueError(f"{name} is a fixed baseline and takes no parameters: {dict(overrides)}")


def _model(name: str, spec: CandidateConfig, seed: int) -> Candidate:
    """A tunable candidate: fixed settings + reference configuration (+ seed)."""
    params: dict[str, object] = {**spec.fixed, **spec.reference}
    if name in SEEDED:
        params["random_state"] = seed
    estimator_class = ESTIMATORS[name]

    def factory(
        feature_config: FeatureConfig, schema: SchemaConfig, overrides: Overrides
    ) -> TransformedTargetRegressor:
        estimator = estimator_class(**{**params, **overrides})
        return build_pipeline(estimator, spec.branch, feature_config, schema)

    return Candidate(
        name=name, branch=spec.branch, tuning=spec.tuning, tier=spec.tier, eligible=True,
        params=params, factory=factory, grid=spec.grid, search_space=spec.search_space,
        n_trials=spec.n_trials,
    )  # fmt: skip


def blend_candidate(
    linear: Candidate,
    linear_params: Overrides,
    tree: Candidate,
    tree_params: Overrides,
    tier: int,
) -> Candidate:
    """The DN-07 blend (DOC-03 §8.8).

    ``TransformedTargetRegressor(log1p/expm1)`` around a ``VotingRegressor`` of the two
    inner pipelines (each with its tuned parameters and its own branch preprocessing), with
    fixed weights 0.5/0.5. Each inner pipeline predicts log price, so the average is taken
    in log space (a geometric mean in dollars). Nothing about the combination is fitted.
    """

    def factory(
        feature_config: FeatureConfig, schema: SchemaConfig, overrides: Overrides
    ) -> TransformedTargetRegressor:
        _no_overrides(BLEND, overrides)
        members = [
            (linear.name, linear.build(feature_config, schema, linear_params).regressor),
            (tree.name, tree.build(feature_config, schema, tree_params).regressor),
        ]
        voting = VotingRegressor(members, weights=list(BLEND_WEIGHTS))
        return TransformedTargetRegressor(regressor=voting, func=np.log1p, inverse_func=np.expm1)

    return Candidate(
        name=BLEND, branch=BOTH, tuning=None, tier=tier, eligible=True,
        params={"linear": linear.name, "tree": tree.name, "weights": list(BLEND_WEIGHTS),
                "linear_params": dict(linear_params), "tree_params": dict(tree_params)},
        factory=factory,
    )  # fmt: skip


def baseline_candidates(models: ModelsConfig) -> tuple[Candidate, ...]:
    """The two baselines, in the order they are evaluated (DOC-03 §8.2, §8.3)."""
    return (_dummy_median(models), _linear_2feat(models))


def model_candidates(models: ModelsConfig, seed: int) -> tuple[Candidate, ...]:
    """Ridge, Lasso, Random Forest, LightGBM, in tier order (DOC-03 §8.1)."""
    specs = models.candidates
    return tuple(_model(name, getattr(specs, name), seed) for name in ESTIMATORS)


def registry(models: ModelsConfig, seed: int) -> dict[str, Candidate]:
    """Every registered candidate by name: the baselines, then the four candidates."""
    candidates = (*baseline_candidates(models), *model_candidates(models, seed))
    return {candidate.name: candidate for candidate in candidates}

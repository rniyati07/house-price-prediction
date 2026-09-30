"""Candidate registry (ADR-10, FR-022, DOC-03 §8.1).

Each entry knows its name, preprocessing branch, how to build its unfitted pipeline,
its tuning method, its simplicity tier, and whether it may be selected. The training
orchestrator only iterates over entries; it holds no model-specific logic.

M6 registers the two baselines. They define the floor [ADR-10] and are never eligible
for selection. M7 adds Ridge, Lasso, Random Forest, and LightGBM with their branches,
tiers (DN-08), fixed settings, and reference configurations from ``models.yaml``; M8 adds
the blend. The global seed (DN-02) is passed to every seeded estimator as ``random_state``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from house_price.config import (
    CandidateConfig,
    FeatureConfig,
    GridConfig,
    ModelsConfig,
    SchemaConfig,
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
        groups = branch_groups(self.branch, feature_config)  # type: ignore[arg-type]
        active = set(groups.numeric + groups.ordinal + groups.nominal)
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
        params=params, factory=factory, grid=spec.grid,
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

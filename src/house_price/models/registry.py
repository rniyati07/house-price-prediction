"""Candidate registry (ADR-10, FR-022, DOC-03 §8.1).

Each entry knows its name, preprocessing branch, how to build its unfitted pipeline,
its tuning method, its simplicity tier, and whether it may be selected. The training
orchestrator only iterates over entries; it holds no model-specific logic.

M6 registers the two baselines. They define the floor [ADR-10] and are never eligible
for selection. M7 and M8 add Ridge, Lasso, Random Forest, LightGBM, and the blend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from house_price.config import FeatureConfig, ModelsConfig, SchemaConfig
from house_price.pipelines.branches import (
    branch_groups,
    build_passthrough_transformer,
    feature_engineer_output,
)
from house_price.pipelines.build import build_pipeline, wrap_estimator

PipelineFactory = Callable[[FeatureConfig, SchemaConfig], TransformedTargetRegressor]
PASSTHROUGH = "two-column pass-through"


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

    def build(
        self, feature_config: FeatureConfig, schema: SchemaConfig
    ) -> TransformedTargetRegressor:
        """A fresh, unfitted raw-to-dollars pipeline for this candidate."""
        return self.factory(feature_config, schema)

    def engineered_features(self, feature_config: FeatureConfig) -> list[str]:
        """Engineered features this candidate's preprocessing consumes (for logging)."""
        if self.branch == PASSTHROUGH:
            return [f for f in feature_config.engineered if f in self.params["features"]]  # type: ignore[operator]
        groups = branch_groups(self.branch, feature_config)  # type: ignore[arg-type]
        active = set(groups.numeric + groups.ordinal + groups.nominal)
        return [f for f in feature_config.engineered if f in active]


def _dummy_median(models: ModelsConfig) -> Candidate:
    spec = models.baselines.dummy_median

    def factory(feature_config: FeatureConfig, schema: SchemaConfig) -> TransformedTargetRegressor:
        return build_pipeline(
            DummyRegressor(strategy=spec.strategy), spec.branch, feature_config, schema
        )

    return Candidate(
        name="dummy_median", branch=spec.branch, tuning=None, tier=None, eligible=False,
        params={"estimator": "DummyRegressor", "strategy": spec.strategy}, factory=factory,
    )  # fmt: skip


def _linear_2feat(models: ModelsConfig) -> Candidate:
    spec = models.baselines.linear_2feat

    def factory(feature_config: FeatureConfig, schema: SchemaConfig) -> TransformedTargetRegressor:
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


def baseline_candidates(models: ModelsConfig) -> tuple[Candidate, ...]:
    """The two baselines, in the order they are evaluated (DOC-03 §8.2, §8.3)."""
    return (_dummy_median(models), _linear_2feat(models))


def registry(models: ModelsConfig) -> dict[str, Candidate]:
    """Every registered candidate by name."""
    return {candidate.name: candidate for candidate in baseline_candidates(models)}

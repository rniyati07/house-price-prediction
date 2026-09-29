"""The single end-to-end pipeline builder (ADR-08, FR-017, DOC-03 §7.1).

Every model in the project is one object::

    TransformedTargetRegressor(func=log1p, inverse_func=expm1)
    └── regressor = Pipeline([
            ("semantic_na", SemanticNAFiller),    # M4, stateless
            ("features",    FeatureEngineer),     # M4, stateless
            ("preprocess",  ColumnTransformer),   # linear or tree branch, fitted
            ("model",       <estimator>),
        ])

The estimator learns ``log1p(SalePrice)`` (ADR-02); ``predict`` returns dollars.

Input contract (DN-11): fit and predict on ``select_model_input(frame, schema)``, i.e.
exactly the 77 model-input columns in schema order. Identifiers, the target, and the
excluded transaction-outcome columns are never pipeline inputs.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from house_price.config import FeatureConfig, SchemaConfig
from house_price.features.engineer import FeatureEngineer
from house_price.features.semantic import SemanticNAFiller
from house_price.pipelines.branches import Branch, build_column_transformer, check_group_coverage


def build_pipeline(
    estimator: BaseEstimator,
    branch: Branch,
    feature_config: FeatureConfig,
    schema: SchemaConfig,
) -> TransformedTargetRegressor:
    """Wrap ``estimator`` in the full raw-to-dollars pipeline for ``branch``.

    The group coverage of ``features.yaml`` is checked first, so a misconfigured column
    fails here instead of being dropped silently. The returned object is unfitted and
    ``clone``-able; each call builds fresh components.
    """
    check_group_coverage(branch, feature_config, schema)
    regressor = Pipeline(
        [
            ("semantic_na", SemanticNAFiller.from_config(feature_config)),
            ("features", FeatureEngineer.from_config(feature_config)),
            ("preprocess", build_column_transformer(branch, feature_config)),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)

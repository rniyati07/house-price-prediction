"""The two preprocessing branches (ADR-08, FR-018 to FR-020, DOC-03 §7.2 to §7.5).

``build_column_transformer`` is the only place a branch's ``ColumnTransformer`` is built.
Column membership comes from ``features.yaml`` (FR-021): nothing here names a feature.

=========  ================================================  ==================================
Group      Linear branch (Ridge, Lasso)                      Tree branch (Random Forest, LightGBM)
=========  ================================================  ==================================
numeric    median impute + indicator -> Yeo-Johnson -> scale  median impute + indicator
ordinal    most-frequent impute -> scale                     most-frequent impute
nominal    most-frequent impute -> one-hot (ignore unknown)  most-frequent impute -> ordinal code
                                                             (unknown -> -1)
dropped    removed (``remainder="drop"``)                    removed
=========  ================================================  ==================================

Every step is fitted inside the pipeline, so its statistics come only from the rows the
pipeline is fitted on (the leakage defense of ADR-08).
"""

from __future__ import annotations

from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler

from house_price.config import BranchGroups, FeatureConfig, SchemaConfig
from house_price.data.schema import model_input_columns

Branch = Literal["linear", "tree"]
BRANCHES: tuple[Branch, ...] = ("linear", "tree")


def branch_groups(branch: Branch, feature_config: FeatureConfig) -> BranchGroups:
    """The configured column groups of ``branch``."""
    if branch not in BRANCHES:
        raise ValueError(f"unknown branch {branch!r}; expected one of {BRANCHES}")
    groups: BranchGroups = getattr(feature_config, branch)
    return groups


def feature_engineer_output(schema: SchemaConfig, feature_config: FeatureConfig) -> list[str]:
    """Columns leaving ``FeatureEngineer``: the model inputs, then the engineered features."""
    return [*model_input_columns(schema), *feature_config.engineered]


def check_group_coverage(
    branch: Branch, feature_config: FeatureConfig, schema: SchemaConfig
) -> None:
    """The four groups must cover every ``FeatureEngineer`` output column exactly once.

    ``remainder="drop"`` removes any column outside the three active groups, so a column
    missing from the configuration would disappear silently; this check makes it an error
    instead (DOC-03 §7.5). Duplicates across groups are already rejected by the config model.
    """
    groups = branch_groups(branch, feature_config)
    expected = set(feature_engineer_output(schema, feature_config))
    configured = set(groups.all_columns)
    missing, unknown = sorted(expected - configured), sorted(configured - expected)
    if missing or unknown:
        raise ValueError(
            f"features.yaml branch {branch!r} does not match the FeatureEngineer output: "
            f"not in any group {missing}; not produced {unknown}"
        )


def _numeric(branch: Branch) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if branch == "linear":
        steps += [
            ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
            ("scale", StandardScaler()),
        ]
    return Pipeline(steps)


def _ordinal(branch: Branch) -> Pipeline:
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="most_frequent"))]
    if branch == "linear":
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _nominal(branch: Branch) -> Pipeline:
    encoder: OneHotEncoder | OrdinalEncoder
    if branch == "linear":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", encoder)])


def build_column_transformer(branch: Branch, feature_config: FeatureConfig) -> ColumnTransformer:
    """Build the linear or tree ``ColumnTransformer`` (DOC-03 §7.5), with pandas output."""
    groups = branch_groups(branch, feature_config)
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", _numeric(branch), list(groups.numeric)),
            ("ordinal", _ordinal(branch), list(groups.ordinal)),
            ("nominal", _nominal(branch), list(groups.nominal)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return transformer.set_output(transform="pandas")

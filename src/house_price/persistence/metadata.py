"""``metadata.json`` of a model artifact (FR-038, AC-044, AC-045; DOC-03 §15.3 to §15.5).

One strict Pydantic model holds every field of DOC-03 §15.3 and nothing else; M11's service
parses the file with the same model (DOC-04 §5.2 step 3). All fields are required and
non-empty (AC-044), with the two staging values DOC-03 documents: ``model_version`` is
``"unreleased"`` and ``mlflow.release`` is ``null`` until ``freeze`` assigns them.

**Candidate artifacts (additional project requirement, M10).** ``artifact_role`` is
``"production"`` for the selected model (every rule above unchanged) or ``"candidate"`` for
the post-selection reference refits of Ridge, Lasso, Random Forest and LightGBM. A candidate
was never evaluated on the holdout, so its ``holdout``, ``baseline_reference``,
``temporal_diagnostic``, ``quality_gates`` and ``mlflow.final_holdout_evaluation`` must be
``null`` (no holdout-derived value is copied or invented); every other field describes that
candidate. Candidates are never released or served.

``input_schema`` is generated from ``schema.yaml`` (model-input roles only, in order), so it
is exactly the contract the API enforces; ``schema_hash`` is the SHA-256 of its normalized
JSON, which the service recomputes at startup (DOC-03 §15.4, FR-005).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from house_price.config import SchemaConfig

UNRELEASED = "unreleased"
# Holdout-derived fields a candidate artifact cannot honestly carry (must be null).
CANDIDATE_NULL_FIELDS = ("holdout", "baseline_reference", "temporal_diagnostic", "quality_gates")
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
LIBRARIES = ("scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "joblib")  # §15.3
_SHA256 = r"^[0-9a-f]{64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MlflowLinks(_Strict):
    """Run IDs of the refit, the final holdout evaluation, and the release run."""

    refit: str = Field(min_length=1)
    final_holdout_evaluation: str | None  # null only for candidate artifacts
    release: str | None = None  # null in staging; set by ``freeze``


class InputColumn(_Strict):
    """One model-input column of the API contract (DOC-03 §15.3 ``input_schema``)."""

    name: str
    dtype: str
    nullable: bool
    allowed_values: list[int] | list[str] | None = None
    min: float | None = None
    max: float | None = None
    strict_min: bool = False


class ScopeRule(_Strict):
    column: str
    max_in_domain: float


class ArtifactMetadata(_Strict):
    """Every field of DOC-03 §15.3."""

    # identity
    model_version: str = Field(min_length=1)
    is_release: bool
    model_sha256: str = Field(pattern=_SHA256)
    created_at: str = Field(min_length=1)
    # provenance
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: bool
    data_sha256: str = Field(pattern=_SHA256)
    split_manifest_sha256: str = Field(pattern=_SHA256)
    config_hash: str = Field(pattern=_SHA256)
    pipeline_run_id: str = Field(min_length=1)
    selection_record_id: str = Field(min_length=1)
    mlflow: MlflowLinks
    # environment
    python_version: str = Field(min_length=1)
    library_versions: dict[str, str]
    # model
    selected_candidate: str = Field(min_length=1)
    hyperparameters: dict[str, Any]
    feature_sets: dict[str, list[str]]
    transformed_feature_names: dict[str, list[str]]
    target_transform: dict[str, str]
    training_rows: int = Field(gt=0)
    # metrics (the holdout-derived ones are null only for candidate artifacts)
    cv: dict[str, float]
    holdout: dict[str, float] | None
    baseline_reference: dict[str, float] | None
    temporal_diagnostic: dict[str, Any] | None
    quality_gates: list[dict[str, Any]] | None
    # contract
    input_schema: list[InputColumn]
    schema_hash: str = Field(pattern=_SHA256)
    scope_rule: ScopeRule
    # role (additional project requirement: candidate reference artifacts)
    artifact_role: Literal["production", "candidate"] = "production"
    # reproducibility
    seed: int
    reproducibility_tolerance: float = Field(gt=0)

    def empty_fields(self) -> list[str]:
        """Fields that are missing or empty (AC-044), allowing only the documented staging
        values: ``mlflow.release`` null and ``model_version`` ``"unreleased"`` while
        ``is_release`` is false."""
        candidate = self.artifact_role == "candidate"
        problems = [
            name
            for name, value in self.model_dump().items()
            if not (candidate and name in CANDIDATE_NULL_FIELDS)
            and (value is None or (isinstance(value, (str, list, dict)) and not value))
        ]
        if candidate:
            carried = [f for f in CANDIDATE_NULL_FIELDS if getattr(self, f) is not None]
            if self.mlflow.final_holdout_evaluation is not None:
                carried.append("mlflow.final_holdout_evaluation")
            problems += [f"{f} (must be null for a candidate)" for f in carried]
            if self.is_release:
                problems.append("is_release (a candidate is never released)")
        elif not self.mlflow.final_holdout_evaluation:
            problems.append("mlflow.final_holdout_evaluation")
        if set(self.library_versions) != set(LIBRARIES) or not all(self.library_versions.values()):
            problems.append("library_versions")
        if any(not names for names in self.transformed_feature_names.values()):
            problems.append("transformed_feature_names")
        if self.is_release:
            if self.mlflow.release is None:
                problems.append("mlflow.release")
            if not SEMVER.match(self.model_version):
                problems.append("model_version")
        elif self.model_version != UNRELEASED and not SEMVER.match(self.model_version):
            problems.append("model_version")
        return sorted(set(problems))


def input_schema(schema: SchemaConfig) -> list[InputColumn]:
    """The ordered model-input columns of ``schema.yaml`` (the API contract)."""
    return [
        InputColumn(
            name=s.name,
            dtype=s.dtype,
            nullable=s.nullable,
            allowed_values=s.allowed_values,
            min=s.min,
            max=s.max,
            strict_min=s.strict_min,
        )
        for s in schema.with_role("model_input")
    ]


def schema_hash(columns: list[InputColumn]) -> str:
    """SHA-256 of the normalized input schema (DOC-03 §15.4)."""
    payload = [c.model_dump(mode="json") for c in columns]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def transformed_feature_names(model: Any, selected: str) -> dict[str, list[str]]:
    """``get_feature_names_out()`` of the fitted preprocessing, per component: the blend's
    ``VotingRegressor`` has one ColumnTransformer per inner pipeline."""
    regressor = model.regressor_
    components = getattr(regressor, "named_estimators_", None)
    if components is not None:
        return {
            name: [str(n) for n in pipe.named_steps["preprocess"].get_feature_names_out()]
            for name, pipe in components.items()
        }
    names = regressor.named_steps["preprocess"].get_feature_names_out()
    return {selected: [str(n) for n in names]}


def feature_sets(components: dict[str, str], features: Any) -> dict[str, list[str]]:
    """Retained engineered features used by each component (``{name: branch}``), after the
    committed ablation outcome (DOC-03 §6.6)."""
    from house_price.pipelines.branches import branch_groups

    sets = {}
    for name, branch in components.items():
        groups = branch_groups(branch, features)  # type: ignore[arg-type]
        active = set(groups.numeric + groups.ordinal + groups.nominal)
        sets[name] = [f for f in features.engineered if f in active]
    return sets

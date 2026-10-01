"""``metadata.json`` contract (DOC-03 §15.3 to §15.5; DOC-05 M10-1; AC-044, AC-045)."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from house_price.config import load_project_config
from house_price.data.schema import model_input_columns
from house_price.persistence import metadata
from house_price.persistence.metadata import ArtifactMetadata
from tests.conftest import CONFIG_DIR, REPO_ROOT, artifact_metadata

# Every field of DOC-03 §15.3, grouped as the table groups them.
DOC03_FIELDS = {
    "model_version", "is_release", "model_sha256", "created_at",
    "git_commit", "git_dirty", "data_sha256", "split_manifest_sha256", "config_hash",
    "pipeline_run_id", "selection_record_id", "mlflow",
    "python_version", "library_versions",
    "selected_candidate", "hyperparameters", "feature_sets", "transformed_feature_names",
    "target_transform", "training_rows",
    "cv", "holdout", "baseline_reference", "temporal_diagnostic", "quality_gates",
    "input_schema", "schema_hash", "scope_rule",
    "seed", "reproducibility_tolerance",
    "artifact_role",  # additional project requirement (candidate reference artifacts)
}  # fmt: skip


def test_model_has_exactly_the_doc03_fields() -> None:
    assert set(ArtifactMetadata.model_fields) == DOC03_FIELDS
    assert set(metadata.MlflowLinks.model_fields) == {
        "refit",
        "final_holdout_evaluation",
        "release",
    }
    assert metadata.LIBRARIES == ("scikit-learn", "lightgbm", "numpy", "pandas", "scipy", "joblib")


def test_complete_staging_metadata_has_no_empty_field() -> None:
    meta = artifact_metadata()
    assert meta.empty_fields() == []  # type: ignore[attr-defined]
    assert meta.model_version == "unreleased" and meta.mlflow.release is None  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("override", "field"),
    [({"hyperparameters": {}}, "hyperparameters"), ({"feature_sets": {}}, "feature_sets"),
     ({"quality_gates": []}, "quality_gates"), ({"holdout": {}}, "holdout"),
     ({"transformed_feature_names": {"ridge": []}}, "transformed_feature_names"),
     ({"library_versions": {"numpy": "2.0"}}, "library_versions"),
     ({"model_version": "latest"}, "model_version")],
)  # fmt: skip
def test_empty_or_missing_values_are_reported(override: dict[str, object], field: str) -> None:
    assert field in artifact_metadata(**override).empty_fields()  # type: ignore[attr-defined]


def test_release_metadata_requires_version_and_release_run() -> None:
    released = artifact_metadata(is_release=True, model_version="1.0.0")
    assert "mlflow.release" in released.empty_fields()  # type: ignore[attr-defined]
    unversioned = artifact_metadata(
        is_release=True,
        mlflow=metadata.MlflowLinks(refit="r", final_holdout_evaluation="f", release="rel"),
    )
    assert "model_version" in unversioned.empty_fields()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_sha256", "short"), ("git_commit", "x" * 40), ("data_sha256", ""),
     ("training_rows", 0), ("schema_hash", "Z" * 64)],
)  # fmt: skip
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        artifact_metadata(**{field: value})


def test_unknown_fields_are_rejected() -> None:
    payload = artifact_metadata().model_dump(mode="json")  # type: ignore[attr-defined]
    payload["smoke"] = True  # not a DOC-03 §15.3 field
    with pytest.raises(ValidationError):
        ArtifactMetadata.model_validate(payload)


def test_json_round_trip() -> None:
    meta = artifact_metadata()
    assert ArtifactMetadata.model_validate_json(meta.model_dump_json()) == meta  # type: ignore[attr-defined]


def test_input_schema_is_the_ordered_77_model_inputs() -> None:
    schema = load_project_config(CONFIG_DIR, REPO_ROOT).schema
    columns = metadata.input_schema(schema)
    assert [c.name for c in columns] == model_input_columns(schema) and len(columns) == 77
    first = schema.by_name(columns[0].name)
    assert (columns[0].dtype, columns[0].nullable) == (first.dtype, first.nullable)
    assert all(c.name not in ("Id", "PID", "SalePrice", "SaleType") for c in columns)


def test_schema_hash_is_deterministic_and_sensitive() -> None:
    schema = load_project_config(CONFIG_DIR, REPO_ROOT).schema
    columns = metadata.input_schema(schema)
    first, second = (
        metadata.schema_hash(columns),
        metadata.schema_hash(metadata.input_schema(schema)),
    )
    assert first == second and re.fullmatch(r"[0-9a-f]{64}", first)
    changed = [columns[0].model_copy(update={"nullable": not columns[0].nullable}), *columns[1:]]
    assert metadata.schema_hash(changed) != first


# ------------------------------------------- candidate artifacts (additional requirement)

NULLS = {"holdout": None, "baseline_reference": None, "temporal_diagnostic": None,
         "quality_gates": None}  # fmt: skip


def _links(final: str | None) -> metadata.MlflowLinks:
    return metadata.MlflowLinks(refit="r1", final_holdout_evaluation=final, release=None)


def test_role_defaults_to_production_with_unchanged_rules() -> None:
    meta = artifact_metadata()
    assert meta.artifact_role == "production" and meta.empty_fields() == []  # type: ignore[attr-defined]
    missing = artifact_metadata(**NULLS, mlflow=_links(None))
    assert set(missing.empty_fields()) >= {  # type: ignore[attr-defined]
        "holdout",
        "baseline_reference",
        "temporal_diagnostic",
        "quality_gates",
        "mlflow.final_holdout_evaluation",
    }


def test_a_candidate_has_null_holdout_derived_fields() -> None:
    candidate = artifact_metadata(artifact_role="candidate", mlflow=_links(None), **NULLS)
    assert candidate.empty_fields() == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("field", ["holdout", "baseline_reference", "temporal_diagnostic",
                                   "quality_gates"])  # fmt: skip
def test_a_candidate_carrying_holdout_values_is_flagged(field: str) -> None:
    values = {**NULLS, field: artifact_metadata().model_dump()[field]}  # type: ignore[attr-defined]
    candidate = artifact_metadata(artifact_role="candidate", mlflow=_links(None), **values)
    assert f"{field} (must be null for a candidate)" in candidate.empty_fields()  # type: ignore[attr-defined]


def test_a_candidate_with_a_final_evaluation_link_or_release_is_flagged() -> None:
    linked = artifact_metadata(artifact_role="candidate", mlflow=_links("f1"), **NULLS)
    problems = linked.empty_fields()  # type: ignore[attr-defined]
    assert "mlflow.final_holdout_evaluation (must be null for a candidate)" in problems
    released = artifact_metadata(artifact_role="candidate", mlflow=_links(None), **NULLS,
                                 is_release=True, model_version="1.0.0")  # fmt: skip
    assert "is_release (a candidate is never released)" in released.empty_fields()  # type: ignore[attr-defined]


def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValidationError):
        artifact_metadata(artifact_role="reference")

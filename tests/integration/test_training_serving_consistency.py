"""Training-serving consistency (DOC-04 §9.2; AC-057; DOC-05 M11-5).

``consistency_rows.csv`` holds development-set rows (never holdout rows) chosen to cover
every absent-feature pattern (no garage, no basement, no fireplace, no pool / pool present,
alley, fence, misc feature, ``MasVnrType`` missing and the literal ``None``), missing
``LotFrontage`` and ``GarageYrBlt``, and categories the smoke model never saw in training
(the synthetic training data has no alley, fence, pool or misc feature). Each row goes
through ``POST /predict`` (``NaN`` as ``null``) and through ``model.predict`` directly on
the same row as a one-row frame parsed like the raw file; the prices must be identical.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from house_price.api.app import create_app
from house_price.persistence import artifact
from tests.conftest import CONSISTENCY_CSV, Serving, as_json_records, consistency_frame


@pytest.fixture(scope="module")
def client(serving: Serving) -> Iterator[TestClient]:
    with TestClient(create_app(serving.settings())) as test_client:  # type: ignore[arg-type]
        yield test_client


def test_fixture_covers_the_required_patterns() -> None:
    frame = consistency_frame()
    assert len(frame) <= 20 and frame.shape[1] == 77
    assert frame["GarageType"].isna().any() and frame["BsmtQual"].isna().any()
    assert frame["LotFrontage"].isna().any() and frame["GarageYrBlt"].isna().any()
    assert frame["PoolQC"].notna().any() and frame["PoolQC"].isna().any()
    assert frame["MasVnrType"].eq("None").any() and frame["MasVnrType"].isna().any()
    assert frame["Alley"].notna().any() and frame["MiscFeature"].notna().any()


def test_fixture_rows_are_development_rows() -> None:
    """Ids are checked against the committed split manifest's ``dev_ids`` (the holdout
    file is never opened)."""
    manifest = CONSISTENCY_CSV.parents[2] / "data" / "processed" / "split_manifest.json"
    if not manifest.is_file():
        pytest.skip("split manifest not present")
    ids = json.loads(manifest.read_text(encoding="utf-8"))
    rows = [int(line.split(",", 1)[0]) for line in CONSISTENCY_CSV.read_text().splitlines()[1:]]
    assert set(rows) <= set(ids["dev_ids"]) and not set(rows) & set(ids["holdout_ids"])


def test_training_serving_consistency(client: TestClient, serving: Serving) -> None:
    model, _ = artifact.load_verified(serving.model_dir)  # the same artifact
    frame = consistency_frame()
    for i, record in enumerate(as_json_records(frame)):
        response = client.post("/predict", json=record)
        assert response.status_code == 200, (i, response.text)
        direct = float(model.predict(frame.iloc[[i]])[0])
        assert response.json()["predicted_price"] == direct, i  # exact (AC-057)


def test_batch_endpoint_equals_direct_prediction(client: TestClient, serving: Serving) -> None:
    model, _ = artifact.load_verified(serving.model_dir)
    frame = consistency_frame()
    response = client.post("/predict/batch", json={"properties": as_json_records(frame)})
    assert response.status_code == 200
    prices = [p["predicted_price"] for p in response.json()["predictions"]]
    assert prices == [float(p) for p in model.predict(frame)]


def test_the_api_uses_the_loaded_artifact_object(client: TestClient, serving: Serving) -> None:
    context = client.app.state.model_context  # type: ignore[attr-defined]
    assert context.metadata.model_sha256 == artifact.read_metadata(serving.model_dir).model_sha256
    assert type(context.model).__name__ == "TransformedTargetRegressor"

"""The four endpoints and their contracts (DOC-04 §6, §7, §13, §16.1; DOC-05 M11-1 to M11-4;
AC-048 to AC-056, AC-058) on the smoke artifact, loaded with ``HPP_ALLOW_NON_RELEASE``.

The batch limit is the project's M11 requirement change: 1 to 20 properties per request."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from house_price.api.app import create_app
from house_price.api.schemas import MAX_BATCH_SIZE
from house_price.persistence import artifact
from tests.conftest import Serving, api_example


@pytest.fixture(scope="module")
def client(serving: Serving) -> Iterator[TestClient]:
    with TestClient(create_app(serving.settings())) as test_client:  # type: ignore[arg-type]
        yield test_client


def lines(captured: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in captured.splitlines() if line.startswith("{")]


def events(captured: str, name: str) -> list[dict[str, Any]]:
    return [line for line in lines(captured) if line["event"] == name]


def body(**changes: object) -> dict[str, object]:
    return {**api_example(), **changes}


def metadata_of(serving: Serving) -> dict[str, Any]:
    return json.loads((serving.model_dir / "metadata.json").read_text(encoding="utf-8"))  # type: ignore[no-any-return]


# ------------------------------------------------------------------------ endpoints


def test_health(client: TestClient, serving: Serving, capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": metadata_of(serving)["model_version"],
    }
    assert response.headers["X-Request-ID"]
    assert lines(capfd.readouterr().out) == []  # health checks are not logged (DOC-04 §6.2)


def test_model_info_equals_metadata(
    client: TestClient, serving: Serving, capfd: pytest.CaptureFixture[str]
) -> None:
    capfd.readouterr()
    response = client.get("/model-info")
    assert response.status_code == 200 and response.json() == metadata_of(serving)  # AC-049
    assert lines(capfd.readouterr().out) == []


def test_predict_example(client: TestClient, serving: Serving) -> None:
    response = client.post("/predict", json=api_example())
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"predicted_price", "model_version", "out_of_domain"}
    assert result["predicted_price"] > 0 and np.isfinite(result["predicted_price"])
    assert result["model_version"] == metadata_of(serving)["model_version"]
    assert result["out_of_domain"] is False  # AC-050


def _varied(n: int) -> list[dict[str, object]]:
    return [body(GrLivArea=1000 + 50 * i, LotArea=5000 + 100 * i) for i in range(n)]


def test_batch_20_ok_and_order_preserved(client: TestClient) -> None:
    properties = _varied(MAX_BATCH_SIZE)
    response = client.post("/predict/batch", json={"properties": properties})
    assert response.status_code == 200
    result = response.json()
    assert result["count"] == MAX_BATCH_SIZE == len(result["predictions"])
    assert [p["index"] for p in result["predictions"]] == list(range(MAX_BATCH_SIZE))
    singles = [client.post("/predict", json=p).json() for p in properties]
    for item, single in zip(result["predictions"], singles, strict=True):  # input order
        assert item["predicted_price"] == single["predicted_price"]
        assert item["out_of_domain"] == single["out_of_domain"]
        assert item["model_version"] == single["model_version"]


def test_batch_of_one_ok(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"properties": [api_example()]})
    assert response.status_code == 200 and response.json()["count"] == 1


def test_batch_21_rejected(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"properties": _varied(MAX_BATCH_SIZE + 1)})
    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"] == ["body", "properties"] and error["type"] == "too_long"
    assert "at most 20 items" in error["msg"]


def test_batch_empty_rejected(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"properties": []})
    assert response.status_code == 422 and response.json()["detail"][0]["type"] == "too_short"


def test_batch_validation_is_atomic(client: TestClient, capfd: pytest.CaptureFixture[str]) -> None:
    properties = _varied(5)
    properties[3] = body(PoolQC="Nonsense")
    capfd.readouterr()
    response = client.post("/predict/batch", json={"properties": properties})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "properties", 3, "PoolQC"]
    assert events(capfd.readouterr().out, "prediction.completed") == []  # nothing predicted


def test_batch_extra_top_level_field_rejected(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"properties": [api_example()], "x": 1})
    assert response.status_code == 422 and response.json()["detail"][0]["type"] == "extra_forbidden"


# ----------------------------------------------------------------------- validation

REJECTED = {
    "missing_field": ({"drop": "GrLivArea"}, "missing"),
    "disallowed_category": ({"Neighborhood": "Atlantis"}, "literal_error"),
    "out_of_range": ({"YrSold": 2011}, "less_than_equal"),
    "non_numeric_in_numeric": ({"LotFrontage": "sixty"}, "float_type"),
    "null_in_non_nullable": ({"OverallQual": None}, "int_type"),
    "string_in_int": ({"LotArea": "5000"}, "int_type"),
    "float_in_int": ({"LotArea": 5000.5}, "int_type"),
    "whole_float_in_int": ({"YearBuilt": 1990.0}, "int_type"),
    "bool_in_float": ({"LotFrontage": True}, "float_type"),
    "bool_in_int": ({"Fireplaces": True}, "int_type"),
    "float_in_int_code": ({"MSSubClass": 60.0}, "int_type"),
    "case_mismatch": ({"MSZoning": "rl"}, "literal_error"),
    "raw_header_name": ({"drop": "1stFlrSF", "1st Flr SF": 856.0}, "missing"),
    "extra_field": ({"SalePrice": 200000}, "extra_forbidden"),
    "identifier_field": ({"Id": 1}, "extra_forbidden"),
    "excluded_field": ({"SaleType": "WD"}, "extra_forbidden"),
    "zero_living_area": ({"GrLivArea": 0}, "greater_than"),
    "float_in_living_area": ({"GrLivArea": 1500.0}, "int_type"),
    "non_finite_float": ({"MasVnrArea": "NaN"}, "float_type"),
}


@pytest.mark.parametrize("case", list(REJECTED))
def test_predict_rejects(client: TestClient, case: str, capfd: pytest.CaptureFixture[str]) -> None:
    changes, error_type = REJECTED[case]
    payload = body(**{k: v for k, v in changes.items() if k != "drop"})
    if "drop" in changes:
        payload.pop(str(changes["drop"]))
    capfd.readouterr()
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # AC-052
    assert error_type in {e["type"] for e in response.json()["detail"]}
    assert events(capfd.readouterr().out, "prediction.completed") == []  # nothing predicted


def test_malformed_json_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict", content=b'{"GrLivArea": ', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422 and response.json()["detail"][0]["type"] == "json_invalid"


def test_missing_pool_qc_points_at_the_field(client: TestClient) -> None:
    payload = body()
    payload.pop("PoolQC")
    detail = client.post("/predict", json=payload).json()["detail"]
    assert [(e["loc"], e["type"]) for e in detail] == [(["body", "PoolQC"], "missing")]


def test_predict_accepts_nullable_null(client: TestClient) -> None:
    payload = body(PoolQC=None, Alley=None, LotFrontage=None, GarageYrBlt=None)
    assert client.post("/predict", json=payload).status_code == 200  # AC-053


def test_json_integer_accepted_in_float_field(client: TestClient) -> None:
    as_int = client.post("/predict", json=body(LotFrontage=60)).json()
    as_float = client.post("/predict", json=body(LotFrontage=60.0)).json()
    assert as_int == as_float


def test_out_of_domain_boundary(client: TestClient, serving: Serving) -> None:
    limit = metadata_of(serving)["scope_rule"]["max_in_domain"]
    assert limit == 4000
    at = client.post("/predict", json=body(GrLivArea=4000))
    above = client.post("/predict", json=body(GrLivArea=4001))
    assert at.status_code == above.status_code == 200
    assert at.json()["out_of_domain"] is False and above.json()["out_of_domain"] is True  # AC-054
    assert above.json()["predicted_price"] > 0  # accepted and priced, never clipped
    batch = client.post("/predict/batch",
                        json={"properties": [body(GrLivArea=4000), body(GrLivArea=6000)]})  # fmt: skip
    assert [p["out_of_domain"] for p in batch.json()["predictions"]] == [False, True]


def test_out_of_domain_value_is_not_clipped(client: TestClient) -> None:
    prices = [client.post("/predict", json=body(GrLivArea=v)).json()["predicted_price"]
              for v in (4001, 6000, 9000)]  # fmt: skip
    assert len(set(prices)) == 3  # 6000 and 9000 are not predicted as 4000 (or 4001)


# ---------------------------------------------------------------- logging, request ID


def test_prediction_log_line(client: TestClient, capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()
    response = client.post("/predict", json=api_example())
    (line,) = lines(capfd.readouterr().out)  # exactly one parseable JSON line (AC-056)
    assert line["event"] == "prediction.completed" and line["level"] == "INFO"
    for field in ("timestamp", "service", "request_id", "endpoint", "n_items", "inputs_sha256",
                  "prediction", "out_of_domain_count", "latency_ms", "model_version",
                  "status_code"):  # fmt: skip
        assert field in line, field
    assert line["request_id"] == response.headers["X-Request-ID"]
    assert (line["endpoint"], line["n_items"], line["status_code"]) == ("/predict", 1, 200)
    assert line["prediction"] == response.json()["predicted_price"]
    assert line["latency_ms"] >= 0 and len(line["inputs_sha256"]) == 64


def test_batch_log_line(client: TestClient, capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()
    response = client.post("/predict/batch", json={"properties": _varied(3)})
    (line,) = lines(capfd.readouterr().out)
    assert (line["event"], line["endpoint"], line["n_items"]) == ("prediction.completed",
                                                                  "/predict/batch", 3)  # fmt: skip
    assert line["prediction"] == [p["predicted_price"] for p in response.json()["predictions"]]


def test_identical_inputs_have_identical_hashes(
    client: TestClient, capfd: pytest.CaptureFixture[str]
) -> None:
    capfd.readouterr()
    client.post("/predict", json=api_example())
    client.post("/predict", json=dict(reversed(list(api_example().items()))))  # key order
    client.post("/predict", json=body(LotArea=12345))
    hashes = [
        line["inputs_sha256"] for line in events(capfd.readouterr().out, "prediction.completed")
    ]
    assert hashes[0] == hashes[1] != hashes[2]


def test_no_raw_payload_is_logged(client: TestClient, capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()
    client.post("/predict", json=body(LotArea=98765))
    client.post("/predict", json=body(Neighborhood="SECRET_TOWN"))  # 422
    client.post("/predict/batch", json={"properties": [body(LotArea=98764)]})
    out = capfd.readouterr().out
    for value in ("98765", "98764", "SECRET_TOWN", 'Neighborhood": "', '"LotArea"'):
        assert value not in out, value
    (failure,) = events(out, "request.validation_failed")
    assert failure["errors"] == [{"loc": ["body", "Neighborhood"], "type": "literal_error"}]
    assert set(failure) == {"timestamp", "level", "event", "service", "model_version",
                            "request_id", "endpoint", "errors"}  # fmt: skip


def test_request_id_header(client: TestClient, capfd: pytest.CaptureFixture[str]) -> None:
    capfd.readouterr()
    first = client.post("/predict", json=api_example())
    second = client.post("/predict", json=api_example())
    logged = [line["request_id"] for line in events(capfd.readouterr().out, "prediction.completed")]
    assert logged == [first.headers["X-Request-ID"], second.headers["X-Request-ID"]]
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    invalid = client.post("/predict", json={})
    assert invalid.status_code == 422 and invalid.headers["X-Request-ID"]
    assert client.get("/nope").headers["X-Request-ID"]  # 404 too
    assert client.get("/predict").status_code == 405


# ------------------------------------------------------------ errors are contained


class _Stub:
    def __init__(self, result: object) -> None:
        self.result = result

    def predict(self, frame: object) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return np.full(len(frame), self.result)  # type: ignore[arg-type]


@pytest.fixture
def stubbed(client: TestClient) -> Iterator[Any]:
    original = client.app.state.model_context  # type: ignore[attr-defined]

    def use(result: object) -> None:
        client.app.state.model_context = dataclasses.replace(original, model=_Stub(result))  # type: ignore[attr-defined]

    yield use
    client.app.state.model_context = original  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -5.0])
def test_guard_violation_returns_500(
    client: TestClient, stubbed: Any, capfd: pytest.CaptureFixture[str], value: float
) -> None:
    stubbed(value)
    capfd.readouterr()
    response = client.post("/predict", json=api_example())
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error",
                               "request_id": response.headers["X-Request-ID"]}  # fmt: skip
    out = capfd.readouterr().out
    (violation,) = events(out, "prediction.guard_violation")
    assert violation["level"] == "ERROR" and violation["rows"] == [0]
    assert violation["request_id"] == response.headers["X-Request-ID"]
    assert events(out, "prediction.completed") == []  # the bad number is never returned


def test_guard_violation_in_a_batch_returns_500(client: TestClient, stubbed: Any) -> None:
    stubbed(float("nan"))
    response = client.post("/predict/batch", json={"properties": _varied(3)})
    assert response.status_code == 500 and "predictions" not in response.json()


def test_unexpected_exception_contained(
    client: TestClient, stubbed: Any, capfd: pytest.CaptureFixture[str]
) -> None:
    stubbed(RuntimeError("boom: internal detail"))
    capfd.readouterr()
    response = client.post("/predict", json=api_example())
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error",
                               "request_id": response.headers["X-Request-ID"]}  # fmt: skip
    assert "boom" not in response.text and "Traceback" not in response.text
    (failure,) = events(capfd.readouterr().out, "request.failed")
    assert failure["error_type"] == "RuntimeError" and "Traceback" in failure["traceback"]
    stubbed(250000.0)
    assert client.post("/predict", json=api_example()).status_code == 200  # still serving


# ---------------------------------------------------------------- load once, OpenAPI


def test_model_loaded_once(serving: Serving, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    real_load = joblib.load

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(artifact.joblib, "load", counting)
    with TestClient(create_app(serving.settings())) as fresh:  # type: ignore[arg-type]
        for _ in range(10):
            assert fresh.post("/predict", json=api_example()).status_code == 200
    assert len(calls) == 1  # AC-055


def test_openapi_example_valid(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/health", "/model-info", "/predict", "/predict/batch"}
    assert set(spec["paths"]["/predict"]) == set(spec["paths"]["/predict/batch"]) == {"post"}
    assert set(spec["paths"]["/health"]) == set(spec["paths"]["/model-info"]) == {"get"}
    schemas = spec["components"]["schemas"]
    (example,) = schemas["PropertyInput"]["examples"]
    assert example == api_example()  # AC-058: the documented example ...
    assert client.post("/predict", json=example).status_code == 200  # ... validates
    assert len(schemas["PropertyInput"]["required"]) == 77
    batch = schemas["BatchInput"]["properties"]["properties"]
    assert (batch["minItems"], batch["maxItems"]) == (1, MAX_BATCH_SIZE) == (1, 20)
    assert "1 to 20" in spec["paths"]["/predict/batch"]["post"]["summary"]
    assert client.get("/docs").status_code == 200

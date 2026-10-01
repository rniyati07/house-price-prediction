"""The batch CLI ``house-price predict`` (DOC-04 §10, §16.3; AC-062, AC-063; DOC-05 M11-8).

Exit codes: 0 success, 2 invalid input (no output file, full report), 3 runtime failure.
Batch limit: 1 to 20 rows per file (the project's M11 requirement change)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from house_price.api.app import create_app
from house_price.api.schemas import MAX_BATCH_SIZE
from house_price.cli import EXIT_INVALID_INPUT, EXIT_RUNTIME, main
from house_price.config import load_project_config
from tests.conftest import (
    CONFIG_DIR,
    CONSISTENCY_CSV,
    FIXTURE_CSV,
    REPO_ROOT,
    Serving,
    as_json_records,
    consistency_frame,
)


@pytest.fixture(autouse=True)
def _override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HPP_ALLOW_NON_RELEASE", "true")  # smoke artifact (DOC-04 §16)


def run(serving: Serving, source: Path, output: Path, *extra: str) -> int:
    return main(["predict", "--input", str(source), "--output", str(output),
                 "--model-dir", str(serving.model_dir), "--config-dir", str(CONFIG_DIR),
                 *extra])  # fmt: skip


def text_rows() -> pd.DataFrame:
    return pd.read_csv(CONSISTENCY_CSV, dtype=str, keep_default_na=False)


def write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def dev_raw_rows(n: int) -> pd.DataFrame:
    """``n`` raw-layout rows of ``raw_sample.csv`` that are development-set rows (by the
    committed split manifest; holdout rows are never used)."""
    manifest = REPO_ROOT / "data" / "processed" / "split_manifest.json"
    raw = pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=False)
    if manifest.is_file():
        dev = set(json.loads(manifest.read_text(encoding="utf-8"))["dev_ids"])
        raw = raw[raw["Order"].astype(int).isin(dev)]
    else:
        raw = raw[raw["Order"].astype(int).isin(set(text_rows()["Id"].astype(int)))]
    return raw.head(n).reset_index(drop=True)


@pytest.fixture(scope="module")
def client(serving: Serving) -> Iterator[TestClient]:
    with TestClient(create_app(serving.settings())) as test_client:  # type: ignore[arg-type]
        yield test_client


# --------------------------------------------------------------------- valid files


def test_valid_file(serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "out.csv"
    assert run(serving, CONSISTENCY_CSV, output) == 0
    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert list(result.columns) == ["row_index", "Id", "predicted_price", "out_of_domain",
                                    "model_version"]  # fmt: skip
    assert result["row_index"].tolist() == [str(i) for i in range(len(result))]  # input order
    assert result["Id"].tolist() == text_rows()["Id"].tolist()  # Id passes through
    assert set(result["out_of_domain"]) <= {"true", "false"}
    assert set(result["model_version"]) == {"unreleased"}
    assert all(float(p) > 0 for p in result["predicted_price"])
    (completed,) = [json.loads(line) for line in capsys.readouterr().err.splitlines()
                    if '"batch.completed"' in line]  # fmt: skip
    assert completed["n_rows"] == len(result) and len(completed["input_sha256"]) == 64


def test_cli_equals_api(serving: Serving, client: TestClient, tmp_path: Path) -> None:  # AC-063
    output = tmp_path / "out.csv"
    assert run(serving, CONSISTENCY_CSV, output) == 0
    cli = pd.read_csv(output, dtype=str, keep_default_na=False)
    records = as_json_records(consistency_frame())
    singles = [client.post("/predict", json=r).json() for r in records]
    batch = client.post("/predict/batch", json={"properties": records}).json()["predictions"]
    for row, single, item in zip(cli.itertuples(), singles, batch, strict=True):
        assert float(row.predicted_price) == single["predicted_price"] == item["predicted_price"]
        assert (row.out_of_domain == "true") == single["out_of_domain"]
        assert row.model_version == single["model_version"]


def test_out_of_domain_rows_are_scored_and_flagged(serving: Serving, tmp_path: Path) -> None:
    rows = text_rows().head(3)
    rows.loc[1, "GrLivArea"] = "4001"
    rows.loc[2, "GrLivArea"] = "4000"
    output = tmp_path / "out.csv"
    assert run(serving, write(rows, tmp_path / "in.csv"), output) == 0
    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert result["out_of_domain"].tolist() == ["false", "true", "false"]


def test_cli_column_rules_raw_layout(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A raw-dataset-layout file (``Order``, ``PID``, ``Gr Liv Area``, ``Sale Type``,
    ``Sale Condition``, ``SalePrice``) is scored without renaming: headers normalized, ``Id``
    and ``PID`` pass through unchanged, the excluded columns and the target are ignored."""
    raw = dev_raw_rows(12)
    raw = raw[list(reversed(raw.columns))]  # any column order
    output = tmp_path / "out.csv"
    assert run(serving, write(raw, tmp_path / "raw.csv"), output) == 0
    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert list(result.columns) == ["row_index", "Id", "PID", "predicted_price",
                                    "out_of_domain", "model_version"]  # fmt: skip
    assert result["Id"].tolist() == raw["Order"].tolist()
    assert result["PID"].tolist() == raw["PID"].tolist()  # text unchanged (leading zeros)
    err = capsys.readouterr().err
    (ignored,) = [json.loads(line) for line in err.splitlines() if "batch.columns_ignored" in line]
    assert set(ignored["columns"]) == {"SaleType", "SaleCondition", "SalePrice"}


def test_raw_layout_and_canonical_layout_agree(serving: Serving, tmp_path: Path) -> None:
    raw = dev_raw_rows(10)
    schema = load_project_config(CONFIG_DIR, REPO_ROOT).schema
    canonical = raw.rename(columns=schema.source_to_name)
    assert run(serving, write(raw, tmp_path / "raw.csv"), tmp_path / "a.csv") == 0
    assert run(serving, write(canonical, tmp_path / "canon.csv"), tmp_path / "b.csv") == 0
    a, b = (pd.read_csv(tmp_path / n, dtype=str) for n in ("a.csv", "b.csv"))
    pd.testing.assert_frame_equal(a, b)


def test_batch_limit_20_accepted(serving: Serving, tmp_path: Path) -> None:
    rows = pd.concat([text_rows()] * 2, ignore_index=True).head(MAX_BATCH_SIZE)
    output = tmp_path / "out.csv"
    assert run(serving, write(rows, tmp_path / "in.csv"), output) == 0
    assert len(pd.read_csv(output)) == MAX_BATCH_SIZE == 20


# ------------------------------------------------------------------- invalid files


def invalid(serving: Serving, frame: pd.DataFrame | Path, tmp_path: Path,
            capsys: pytest.CaptureFixture[str]) -> str:  # fmt: skip
    """Exit 2, no output file, and the report (stderr) returned."""
    source = frame if isinstance(frame, Path) else write(frame, tmp_path / "in.csv")
    output = tmp_path / "out.csv"
    capsys.readouterr()
    assert run(serving, source, output) == EXIT_INVALID_INPUT
    assert not output.exists() and not (tmp_path / "out.csv.partial").exists()
    return capsys.readouterr().err


def test_cli_rejects_invalid_file_with_a_full_report(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # AC-062
    rows = text_rows()
    rows.loc[1, "GrLivArea"] = "-5"  # out of range
    rows.loc[4, "Neighborhood"] = "Atlantis"  # disallowed category
    rows.loc[7, "LotArea"] = "big"  # not a number
    rows.loc[9, "OverallQual"] = "NA"  # null in a non-nullable column
    rows.loc[11, "YrSold"] = "2011"  # out of range
    report = invalid(serving, rows, tmp_path, capsys)
    for row, column in ((1, "GrLivArea"), (4, "Neighborhood"), (7, "LotArea"),
                        (9, "OverallQual"), (11, "YrSold")):  # fmt: skip
        assert f"row_index={row} (line {row + 2}) column='{column}'" in report, (row, column)
    assert "'Atlantis'" in report and "'big'" in report
    failed = [json.loads(line) for line in report.splitlines() if '"batch.failed"' in line]
    assert failed[0]["n_invalid_rows"] == 5 and failed[0]["exit_code"] == EXIT_INVALID_INPUT


def test_cli_rejects_unexpected_and_duplicated_columns(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = text_rows()
    rows["Colour"] = "blue"  # unknown column
    rows["Gr Liv Area"] = rows["GrLivArea"]  # duplicates GrLivArea after normalization
    report = invalid(serving, rows, tmp_path, capsys)
    assert "column='Colour' check='unexpected_column'" in report
    assert "column='GrLivArea' check='duplicate_column'" in report


def test_cli_rejects_missing_columns(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = invalid(serving, text_rows().drop(columns=["PoolQC", "GrLivArea"]), tmp_path, capsys)
    assert "column='PoolQC' check='column_in_dataframe'" in report
    assert "column='GrLivArea' check='column_in_dataframe'" in report


def test_cli_rejects_more_than_20_rows(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = pd.concat([text_rows()] * 2, ignore_index=True).head(MAX_BATCH_SIZE + 1)
    report = invalid(serving, rows, tmp_path, capsys)
    assert "MAX_BATCH_SIZE = 20" in report and "value=21" in report


def test_cli_rejects_an_empty_file(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header_only = text_rows().head(0)
    assert "value=0" in invalid(serving, header_only, tmp_path, capsys)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert "unreadable CSV" in invalid(serving, empty, tmp_path, capsys)


def test_cli_rejects_a_missing_input_file(
    serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "not found" in invalid(serving, tmp_path / "absent.csv", tmp_path, capsys)


def test_dn18_tokens(serving: Serving, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Only ``NA`` and the empty cell are missing: ``None`` stays the ``MasVnrType`` category,
    and ``null`` is not a missing token (so it is rejected as a category)."""
    rows = text_rows().head(2)
    rows.loc[0, "MasVnrType"] = "None"
    rows.loc[1, "MasVnrType"] = ""
    assert run(serving, write(rows, tmp_path / "ok.csv"), tmp_path / "ok_out.csv") == 0
    rows.loc[1, "MasVnrType"] = "null"
    assert "column='MasVnrType'" in invalid(serving, rows, tmp_path, capsys)


# ------------------------------------------------------------------ runtime failures


def test_cli_refuses_a_non_release_artifact_without_the_override(
    serving: Serving, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    monkeypatch.delenv("HPP_ALLOW_NON_RELEASE")
    output = tmp_path / "out.csv"
    assert run(serving, CONSISTENCY_CSV, output) == EXIT_RUNTIME
    assert not output.exists() and "is_release=false" in capsys.readouterr().err


def test_cli_refuses_a_tampered_artifact(serving: Serving, tmp_path: Path) -> None:
    import shutil

    model_dir = tmp_path / "model"
    shutil.copytree(serving.model_dir, model_dir)
    with (model_dir / "model.joblib").open("ab") as handle:
        handle.write(b"x")
    output = tmp_path / "out.csv"
    code = main(["predict", "--input", str(CONSISTENCY_CSV), "--output", str(output),
                 "--model-dir", str(model_dir), "--config-dir", str(CONFIG_DIR)])  # fmt: skip
    assert code == EXIT_RUNTIME and not output.exists()


def test_cli_guard_violation_exits_3(
    serving: Serving, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from house_price.persistence import artifact

    class NaNModel:
        def predict(self, frame: pd.DataFrame) -> object:
            return np.full(len(frame), np.nan)

    monkeypatch.setattr(artifact.joblib, "load", lambda *a, **k: NaNModel())
    output = tmp_path / "out.csv"
    assert run(serving, CONSISTENCY_CSV, output) == EXIT_RUNTIME
    assert not output.exists()


def test_cli_reads_the_model_dir_from_the_environment(
    serving: Serving, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPP_MODEL_DIR", str(serving.model_dir))
    monkeypatch.setenv("HPP_CONFIG_DIR", str(CONFIG_DIR))
    output = tmp_path / "out.csv"
    assert main(["predict", "--input", str(CONSISTENCY_CSV), "--output", str(output)]) == 0
    assert output.exists()

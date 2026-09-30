"""OOF diagnostics (DOC-03 §11.6, DN-09; DOC-05 M9-2; AC-037): the three gate views and
their tables, computed from OOF predictions on development rows. No pass/fail (IN-10)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price.evaluation import diagnostics


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 200
    dev = pd.DataFrame({"Id": np.arange(1, n + 1),
                        "SalePrice": np.round(np.exp(rng.normal(12, 0.4, n))),
                        "Neighborhood": rng.choice(["A", "B", "C", "D"], n)})  # fmt: skip
    oof = pd.concat([
        pd.DataFrame({"Id": dev["Id"], "repeat": r, "fold": 1,
                      "log_prediction": np.log1p(dev["SalePrice"]) + rng.normal(0, 0.1, n)})
        for r in (1, 2, 3)
    ], ignore_index=True)  # fmt: skip
    return oof, dev


def test_oof_is_the_mean_over_repeats(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    oof, _ = data
    agg = diagnostics.aggregate_oof(oof)
    expected = oof.groupby("Id")["log_prediction"].mean().to_numpy()
    assert np.allclose(agg["log_prediction"], expected) and set(agg["n_repeats"]) == {3}


def test_report_has_all_plots_and_tables(
    data: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    oof, dev = data
    report = diagnostics.write_report("ridge", oof, dev, tmp_path)
    for name in (diagnostics.RESIDUAL_PLOT, diagnostics.DECILE_PLOT, diagnostics.NEIGHBORHOOD_PLOT,
                 diagnostics.DECILE_TABLE, diagnostics.NEIGHBORHOOD_TABLE,
                 diagnostics.TREND_TABLE, diagnostics.OOF_TABLE):  # fmt: skip
        assert (tmp_path / name).is_file() and (tmp_path / name).stat().st_size > 100, name
    assert len(report.deciles) == 10 and report.deciles["n"].sum() == len(dev)
    assert (report.deciles["ci_low"] <= report.deciles["mean_signed_error"]).all()
    nb = report.neighborhoods
    assert list(nb["log_rmse"]) == sorted(nb["log_rmse"], reverse=True)
    assert nb["n"].sum() == len(dev)
    errors = report.frame["signed_error"]
    assert report.overall_log_rmse == pytest.approx(float(np.sqrt(np.mean(errors**2))))
    summary = report.summary()
    assert summary["n_rows"] == len(dev) and summary["repeats_per_row"] == [3]
    assert "pass" not in str(summary).lower()  # evidence only (IN-10)


def test_signed_error_is_predicted_minus_actual(data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    oof, dev = data
    frame = diagnostics.diagnostic_frame(oof.assign(log_prediction=oof["log_prediction"] * 0 + 13),
                                         dev, id_column="Id", target="SalePrice",
                                         group="Neighborhood")  # fmt: skip
    assert np.allclose(frame["signed_error"], 13 - np.log1p(frame["SalePrice"]))


def test_oof_must_cover_exactly_the_development_rows(
    data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    oof, dev = data
    with pytest.raises(diagnostics.DiagnosticsError):
        diagnostics.diagnostic_frame(oof[oof["Id"] != 1], dev, id_column="Id",
                                     target="SalePrice", group="Neighborhood")  # fmt: skip

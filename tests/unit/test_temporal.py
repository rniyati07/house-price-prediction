"""Temporal diagnostic (DOC-03 §13, DN-10; DOC-05 M9-4; AC-043 logic): development rows
only, 2006-2009 train, 2010 test, labelled "not used for selection"."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from house_price.config import load_feature_config
from house_price.data import split
from house_price.data.load import load_raw
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.evaluation import temporal
from house_price.pipelines import build_pipeline
from tests.conftest import CONFIG_DIR, make_env


def test_split_uses_the_documented_years() -> None:
    dev = pd.DataFrame({"Id": range(10), "YrSold": [2006, 2007, 2008, 2009, 2010] * 2})
    train, test = temporal.temporal_split(dev)
    assert set(train["YrSold"]) == {2006, 2007, 2008, 2009} and set(test["YrSold"]) == {2010}
    assert len(train) + len(test) == len(dev)


@pytest.mark.parametrize("years", [[2006, 2011], [2005, 2010]])
def test_years_outside_the_window_are_refused(years: list[int]) -> None:
    with pytest.raises(temporal.TemporalError, match="outside"):
        temporal.temporal_split(pd.DataFrame({"Id": [1, 2], "YrSold": years}))


def test_both_parts_are_required() -> None:
    with pytest.raises(temporal.TemporalError, match="both"):
        temporal.temporal_split(pd.DataFrame({"Id": [1, 2], "YrSold": [2006, 2007]}))


def test_temporal_run_uses_development_rows_only(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    assert split.main(env.cli_args()) == 0
    config = env.load()
    in_scope, scope = apply_scope_rule(load_raw(config), config.data.scope, "Id")
    dev = create_or_load_split(in_scope, scope, config, verify_holdout_file=False).dev
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    features = load_feature_config(CONFIG_DIR)
    result = temporal.run_temporal(
        lambda: build_pipeline(Ridge(alpha=10.0), "linear", features, config.schema),
        dev, config.schema,
    )  # fmt: skip
    ids = set(result.train_ids) | set(result.test_ids)
    assert ids <= set(manifest["dev_ids"]) and not ids & set(manifest["holdout_ids"])
    assert result.n_train + result.n_test == len(dev)
    by_id = dev.set_index("Id")["YrSold"]
    assert set(by_id.loc[result.test_ids]) == {2010}
    assert set(by_id.loc[result.train_ids]) <= {2006, 2007, 2008, 2009}
    assert set(result.metrics) == {"log_rmse", "mae", "mape", "r2"}
    assert np.isfinite(list(result.metrics.values())).all()
    assert result.as_record()["label"] == "not used for selection"

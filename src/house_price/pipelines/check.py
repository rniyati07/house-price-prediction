"""M5 preprocessing check: ``python -m house_price.pipelines`` (DOC-05 M5 verification).

For each branch, the actual pipeline from :func:`build_pipeline` is built with the real
configuration; only its preprocessing steps (semantic fill -> feature engineering ->
``ColumnTransformer``) are fitted, on the development set. No estimator is fitted, and the
holdout is never read. What the fitted transformer reports (input and output columns per
group, missing indicators, the target transform) is written to an M5 run record.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from house_price.config import (
    DEFAULT_CONFIG_DIR,
    ConfigError,
    FeatureConfig,
    SchemaConfig,
    config_hash,
    load_feature_config,
    load_project_config,
)
from house_price.data.errors import DataError
from house_price.data.load import load_raw, sha256_file
from house_price.data.schema import select_model_input
from house_price.data.scope import apply_scope_rule
from house_price.data.split import create_or_load_split
from house_price.pipelines.branches import BRANCHES, Branch, branch_groups
from house_price.pipelines.build import build_pipeline
from house_price.results import ResultRun, start_run

GROUPS = ("numeric", "ordinal", "nominal")
INDICATOR = "missingindicator_"


class CheckError(DataError):
    """The preprocessing check cannot run on the current project state."""


@dataclass(frozen=True)
class BranchReport:
    branch: Branch
    n_rows: int
    n_inputs: int
    n_engineered_output: int
    configured: dict[str, int]  # columns per group in features.yaml (incl. dropped)
    output: dict[str, int]  # features per group after the fitted ColumnTransformer
    n_output: int
    indicators: list[str]
    output_has_missing: bool
    target_func: str
    target_inverse_func: str
    target_round_trip_max_abs_error: float

    def as_record(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def check_branch(
    branch: Branch, X: pd.DataFrame, y: pd.Series, features: FeatureConfig, schema: SchemaConfig
) -> BranchReport:
    """Fit the preprocessing of the real ``branch`` pipeline on ``X`` and describe it."""
    model = build_pipeline(DummyRegressor(), branch, features, schema)  # coverage checked here
    preprocessing = model.regressor[:-1]  # every step except the (unfitted) estimator
    output = preprocessing.fit_transform(X, np.log1p(y))
    engineered = preprocessing[:-1].transform(X)
    names = [str(name) for name in output.columns]
    groups = branch_groups(branch, features)
    target = y.to_numpy(dtype="float64")
    round_trip = model.inverse_func(model.func(target))
    return BranchReport(
        branch=branch, n_rows=len(output), n_inputs=X.shape[1],
        n_engineered_output=engineered.shape[1],
        configured={g: len(getattr(groups, g)) for g in (*GROUPS, "dropped")},
        output={g: sum(n.startswith(f"{g}__") for n in names) for g in GROUPS},
        n_output=len(names), indicators=[n for n in names if INDICATOR in n],
        output_has_missing=bool(output.isna().to_numpy().any()),
        target_func=model.func.__name__, target_inverse_func=model.inverse_func.__name__,
        target_round_trip_max_abs_error=float(np.max(np.abs(round_trip - target))),
    )  # fmt: skip


def run_check(
    run: ResultRun, config_dir: Path | None = None, root: Path | None = None
) -> list[BranchReport]:
    """Load the development set through the M2 layer and check both branches."""
    root = (root or Path.cwd()).resolve()
    config_dir = config_dir or root / DEFAULT_CONFIG_DIR
    config = load_project_config(config_dir, root)
    features = load_feature_config(config_dir)
    if not config.manifest_path.is_file():
        raise CheckError(f"split manifest not found at {config.manifest_path}; "
                         "create the split first (make split)")  # fmt: skip
    raw = load_raw(config)
    in_scope, record = apply_scope_rule(raw, config.data.scope, config.schema.id_column)
    dev = create_or_load_split(in_scope, record, config).dev
    run.log_lineage(
        data_sha256=config.data.raw_sha256,
        split_manifest_sha256=sha256_file(config.manifest_path),
        config_hash=config_hash(config.data, config.validation, config.schema, features),
        config_hash_covers=["data", "validation", "schema", "features"],
        seed=config.validation.seed,
    )
    X, y = select_model_input(dev, config.schema), dev[config.schema.target]
    reports = [check_branch(branch, X, y, features, config.schema) for branch in BRANCHES]
    run.log_params({"fitted_on": "development set", "estimator_fitted": False,
                    "branches": list(BRANCHES)})  # fmt: skip
    for rep in reports:
        prefix = f"{rep.branch}."
        run.log_metrics({
            f"{prefix}n_rows": rep.n_rows, f"{prefix}n_inputs": rep.n_inputs,
            f"{prefix}n_engineered_output": rep.n_engineered_output,
            f"{prefix}n_output": rep.n_output, f"{prefix}n_indicators": len(rep.indicators),
            **{f"{prefix}output.{g}": n for g, n in rep.output.items()},
        })  # fmt: skip
        run.log_value(rep.branch, rep.as_record())
    run.log_value("group_coverage", "passed for every branch (check_group_coverage)")
    return reports


def report(reports: list[BranchReport]) -> str:
    lines = [(f"{'branch':<8}{'rows':>6}{'inputs':>8}{'engineered':>12}{'numeric':>9}"
              f"{'ordinal':>9}{'nominal':>9}{'total':>7}  indicators")]  # fmt: skip
    for r in reports:
        lines.append(
            f"{r.branch:<8}{r.n_rows:>6}{r.n_inputs:>8}{r.n_engineered_output:>12}"
            f"{r.output['numeric']:>9}{r.output['ordinal']:>9}{r.output['nominal']:>9}"
            f"{r.n_output:>7}  {', '.join(r.indicators) or 'none'}"
        )
    first = reports[0]
    lines.append(f"target transform: {first.target_func} / {first.target_inverse_func}; "
                 "no estimator fitted; holdout not read")  # fmt: skip
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m house_price.pipelines",
        description="Fit the M5 preprocessing on the development set and record what it produces.",
    )
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    parser.add_argument("--results-dir", type=Path, default=None, help="default: <root>/results")
    args = parser.parse_args(argv)
    root = (args.root or Path.cwd()).resolve()
    try:
        with start_run("M5", root=root, results_dir=args.results_dir,
                       entry_point="python -m house_price.pipelines", argv=argv) as run:  # fmt: skip
            reports = run_check(run, args.config_dir, root)
            print(report(reports))
            print(f"Run record: {run.path}")
    except (ConfigError, DataError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0

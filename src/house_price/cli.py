"""The ``house-price`` command line (FR-023, DOC-03 §3.2).

Subcommands are added milestone by milestone. M6 adds ``train`` (extended by M7 with the
ablation, the RC-02 check, and the development checks, by M8 with tuning and the comparison,
and by M9 with selection and diagnostics); M9 adds ``evaluate`` (up to the temporal
diagnostic) and ``--smoke`` (DN-17); M10 adds the production refit to ``evaluate`` and
``freeze``; M11 adds ``predict``, the batch CLI (DOC-04 §10; exit codes 0 / 2 / 3).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from house_price.config import ConfigError, SchemaConfig
from house_price.data.errors import DataError


def _train(args: argparse.Namespace) -> int:
    from house_price.models.train import EXIT_RC02_STOP, report, run_train
    from house_price.tracking import TrackingError

    def progress(message: str) -> None:
        print(f"[train] {message}", file=sys.stderr, flush=True)

    try:
        result = run_train(
            args.config_dir,
            args.root,
            args.tracking_uri,
            args.results_dir,
            args.argv,
            progress,
            smoke=args.smoke,
        )
    except (ConfigError, DataError, TrackingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(report(result))
    return EXIT_RC02_STOP if result.stopped else 0


def _evaluate(args: argparse.Namespace) -> int:
    from house_price.evaluation.holdout import run_evaluate
    from house_price.models.selection import SelectionError
    from house_price.tracking import TrackingError

    def progress(message: str) -> None:
        print(f"[evaluate] {message}", file=sys.stderr, flush=True)

    try:
        summary = run_evaluate(args.config_dir, args.root, args.tracking_uri, smoke=args.smoke,
                               argv=args.argv, log=progress)  # fmt: skip
    except (ConfigError, DataError, TrackingError, SelectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    import json

    print(json.dumps(summary, indent=2))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    from house_price.persistence.artifact import FreezeError, freeze
    from house_price.tracking import TrackingError, default_tracking_uri

    root = (args.root or Path.cwd()).resolve()
    try:
        result = freeze(args.version, root, args.config_dir or root / "configs",
                        args.tracking_uri or default_tracking_uri(root))  # fmt: skip
    except (ConfigError, DataError, TrackingError, FreezeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"frozen {result.version} -> {result.directory} (model_sha256 {result.model_sha256}; "
          f"hpp-release run {result.release_run_id})")  # fmt: skip
    print(f"Tag the release commit explicitly: git tag v{result.version}")
    return 0


# ------------------------------------------------------------- predict (M11, DOC-04 §10)

EXIT_INVALID_INPUT = 2  # AC-062: no output file, full validation report
EXIT_RUNTIME = 3  # artifact, configuration, guard (SD-16) or other runtime failure


@dataclass(frozen=True)
class InputCheck:
    """The validated 77-column frame (schema order), identifier and ignored columns, and
    every problem found (``FAILURE_COLUMNS`` rows; empty when the file is valid)."""

    frame: pd.DataFrame
    passthrough: pd.DataFrame
    ignored: list[str]
    problems: pd.DataFrame


def check_input(text: pd.DataFrame, schema: SchemaConfig) -> InputCheck:
    """DOC-04 §10.1 steps 3 and 4 on a text frame read with the DN-18 tokens: header
    normalization with the ingestion mapping, the SD-12 column rules, the batch size limit,
    casting, and lazy validation with the Pandera inference schema. Collects everything."""
    from house_price.api.schemas import MAX_BATCH_SIZE, MIN_BATCH_SIZE
    from house_price.data.errors import FAILURE_COLUMNS, DataValidationError
    from house_price.data.load import cast_to_schema, rename_source_columns
    from house_price.data.schema import (
        build_inference_schema,
        model_input_columns,
        validate_frame,
    )

    problems: list[dict[str, object]] = []
    renamed = rename_source_columns(text, schema)
    duplicated = renamed.columns[renamed.columns.duplicated()].unique().tolist()
    problems += [{"column": c, "check": "duplicate_column", "index": None, "failure_case": c}
                 for c in duplicated]  # fmt: skip
    frame = renamed.loc[:, ~renamed.columns.duplicated()]
    inputs = model_input_columns(schema)
    identifiers = [s.name for s in schema.with_role("identifier") if s.name in frame.columns]
    ignored = [s.name for s in schema.with_role("excluded", "target") if s.name in frame.columns]
    known = set(inputs) | set(identifiers) | set(ignored)
    problems += [{"column": c, "check": "unexpected_column", "index": None, "failure_case": c}
                 for c in frame.columns if c not in known]  # fmt: skip
    if not MIN_BATCH_SIZE <= len(frame) <= MAX_BATCH_SIZE:
        problems.append({"column": None, "index": None, "failure_case": len(frame),
                         "check": f"batch_size between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE} "
                                  f"rows (MAX_BATCH_SIZE = {MAX_BATCH_SIZE})"})  # fmt: skip
    cast = cast_to_schema(frame.loc[:, [c for c in inputs if c in frame.columns]], schema)
    validated = cast.frame
    try:
        validated = validate_frame(cast.frame, build_inference_schema(schema), cast.failures)
    except DataValidationError as exc:
        problems += exc.failure_cases.to_dict("records")
    return InputCheck(
        frame=validated,
        passthrough=frame.loc[:, identifiers],
        ignored=ignored,
        problems=pd.DataFrame(problems, columns=FAILURE_COLUMNS),
    )


def validation_report(problems: pd.DataFrame) -> str:
    """Every failure: row (0-based ``row_index`` and file line), column, check, value."""
    lines = [f"invalid input: {len(problems)} problem(s); no output file was written"]
    for row in problems.itertuples(index=False):
        where = "file" if pd.isna(row.index) else (
            f"row_index={int(row.index)} (line {int(row.index) + 2})")  # fmt: skip
        lines.append(f"  {where} column={row.column!r} check={row.check!r} "
                     f"value={row.failure_case!r}")  # fmt: skip
    return "\n".join(lines)


def run_predict(
    input_path: Path,
    output_path: Path,
    model_dir: Path,
    config_dir: Path,
    *,
    allow_non_release: bool = False,
    log_level: str = "INFO",
) -> int:
    """DOC-04 §10.1: verified load (startup steps 3 to 10), DN-18 parsing, SD-12 columns,
    lazy inference-schema validation, one ``predict`` call, the SD-16 guard, output CSV."""
    import logging
    import os
    import time

    from house_price.api.logging import BATCH_LOGGER, configure_logging, log_event
    from house_price.api.predict import (
        GuardViolation,
        StartupError,
        load_model_context,
        predict_frame,
    )
    from house_price.config import DataConfig, load_model
    from house_price.data.load import read_text_csv, sha256_file

    started = time.perf_counter()
    logger = configure_logging(log_level, BATCH_LOGGER, stream="stderr")
    try:
        context = load_model_context(model_dir, config_dir, allow_non_release=allow_non_release,
                                     logger=logger, warm_up=False)  # fmt: skip
        tokens = load_model(DataConfig, config_dir / "data.yaml").missing_tokens
    except (StartupError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    def failed(code: int, digest: str | None, invalid_rows: int, message: str) -> int:
        log_event(logger, logging.ERROR, "batch.failed", input_sha256=digest,
                  n_invalid_rows=invalid_rows, exit_code=code)  # fmt: skip
        print(message, file=sys.stderr)
        return code

    if not input_path.is_file():
        return failed(EXIT_INVALID_INPUT, None, 0, f"invalid input: {input_path} not found")
    digest = sha256_file(input_path)
    try:
        text = read_text_csv(input_path, tokens)
    except (ValueError, UnicodeDecodeError) as exc:  # pandas parser and empty-file errors
        return failed(EXIT_INVALID_INPUT, digest, 0, f"invalid input: unreadable CSV: {exc}")
    check = check_input(text, context.schema)
    if check.ignored:
        log_event(logger, logging.INFO, "batch.columns_ignored", columns=check.ignored)
    if not check.problems.empty:
        rows = check.problems["index"].dropna().nunique()
        return failed(EXIT_INVALID_INPUT, digest, int(rows), validation_report(check.problems))
    try:
        predictions = predict_frame(context, check.frame)
    except GuardViolation as exc:
        return failed(EXIT_RUNTIME, digest, 0, f"ERROR: {exc}; no output file was written")
    except Exception as exc:  # noqa: BLE001 - any prediction failure is a runtime failure
        return failed(EXIT_RUNTIME, digest, 0, f"ERROR: prediction failed: {exc!r}")
    output = pd.DataFrame({"row_index": range(len(check.frame))})
    for column in check.passthrough.columns:  # Id / PID copied unchanged (text as read)
        output[column] = check.passthrough[column].to_numpy()
    output["predicted_price"] = [repr(p) for p in predictions.prices]  # round-trip safe
    output["out_of_domain"] = ["true" if f else "false" for f in predictions.out_of_domain]
    output["model_version"] = context.metadata.model_version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    output.to_csv(partial, index=False)
    os.replace(partial, output_path)  # the output appears complete or not at all
    log_event(logger, logging.INFO, "batch.completed", input_sha256=digest,
              n_rows=len(output), model_version=context.metadata.model_version,
              duration_ms=round((time.perf_counter() - started) * 1000, 3))  # fmt: skip
    print(f"wrote {len(output)} predictions to {output_path}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from house_price.api.settings import Settings, SettingsError

    try:
        settings = Settings.from_env()
    except SettingsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    return run_predict(args.input, args.output, args.model_dir or settings.model_dir,
                       args.config_dir or settings.config_dir,
                       allow_non_release=settings.allow_non_release,
                       log_level=settings.log_level)  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="house-price", description="House Price Prediction")
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser(
        "train", help="baselines, ablation (RC-02), tuning, comparison, selection (M6-M9)"
    )
    train.add_argument("--config-dir", type=Path, default=None, help="default: <root>/configs")
    train.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    train.add_argument("--tracking-uri", default=None, help="default: file store <root>/mlruns")
    train.add_argument("--results-dir", type=Path, default=None, help="default: <root>/results")
    train.add_argument("--smoke", action="store_true", help="DN-17 smoke run (hpp-smoke)")
    train.set_defaults(handler=_train)
    evaluate = commands.add_parser(
        "evaluate",
        help="final holdout evaluation, baseline reference, gates, temporal diagnostic (M9)",
    )
    evaluate.add_argument("--config-dir", type=Path, default=None, help="default: <root>/configs")
    evaluate.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    evaluate.add_argument("--tracking-uri", default=None, help="default: file store <root>/mlruns")
    evaluate.add_argument("--smoke", action="store_true",
                          help="DN-17: the holdout substitute; never the real holdout")  # fmt: skip
    evaluate.set_defaults(handler=_evaluate)
    freeze = commands.add_parser("freeze", help="release the staging artifact as x.y.z (M10)")
    freeze.add_argument("--version", required=True, help="semantic version x.y.z")
    freeze.add_argument("--config-dir", type=Path, default=None, help="default: <root>/configs")
    freeze.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    freeze.add_argument("--tracking-uri", default=None, help="default: file store <root>/mlruns")
    freeze.set_defaults(handler=_freeze)
    predict = commands.add_parser("predict", help="score a CSV with the verified artifact (M11)")
    predict.add_argument("--input", type=Path, required=True, help="input CSV (SD-12 columns)")
    predict.add_argument("--output", type=Path, required=True, help="output CSV")
    predict.add_argument("--model-dir", type=Path, default=None, help="default: HPP_MODEL_DIR")
    predict.add_argument("--config-dir", type=Path, default=None, help="default: HPP_CONFIG_DIR")
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.argv = list(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 0
    handler = args.handler
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())

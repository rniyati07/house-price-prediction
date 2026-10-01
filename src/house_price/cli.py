"""The ``house-price`` command line (FR-023, DOC-03 §3.2).

Subcommands are added milestone by milestone. M6 adds ``train`` (extended by M7 with the
ablation, the RC-02 check, and the development checks, by M8 with tuning and the comparison,
and by M9 with selection and diagnostics); M9 adds ``evaluate`` (up to the temporal
diagnostic) and ``--smoke`` (DN-17); M10 adds the production refit to ``evaluate`` and
``freeze``; ``predict`` (M11) follows.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from house_price.config import ConfigError
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

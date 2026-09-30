"""The ``house-price`` command line (FR-023, DOC-03 §3.2).

Subcommands are added milestone by milestone. M6 adds ``train`` (extended by M7 with the
ablation, the RC-02 check, and the development checks); ``evaluate`` and
``freeze`` (M9, M10) and ``predict`` (M11) follow.
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
            args.config_dir, args.root, args.tracking_uri, args.results_dir, args.argv, progress
        )
    except (ConfigError, DataError, TrackingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(report(result))
    return EXIT_RC02_STOP if result.stopped else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="house-price", description="House Price Prediction")
    commands = parser.add_subparsers(dest="command")
    train = commands.add_parser(
        "train", help="baselines, feature ablation (RC-02 check), development checks (M6-M7)"
    )
    train.add_argument("--config-dir", type=Path, default=None, help="default: <root>/configs")
    train.add_argument("--root", type=Path, default=None, help="project root (default: cwd)")
    train.add_argument("--tracking-uri", default=None, help="default: file store <root>/mlruns")
    train.add_argument("--results-dir", type=Path, default=None, help="default: <root>/results")
    train.set_defaults(handler=_train)
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

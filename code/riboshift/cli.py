from __future__ import annotations

import argparse
from pathlib import Path

from riboshift.config import load_config
from riboshift.core import build_core
from riboshift.ingest import run_ingest
from riboshift.splits import make_splits
from riboshift.validate import validate_artifact_dir, validate_config


def build_parser() -> argparse.ArgumentParser:
    # Keep the CLI small and explicit so the release can be reproduced with a
    # few top-level commands.
    parser = argparse.ArgumentParser(description="RiboShift benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest source tables into a standardized artifact table")
    ingest_parser.add_argument("--config", required=True)
    ingest_parser.add_argument("--output-dir", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate config and/or generated artifacts")
    validate_parser.add_argument("--config", required=False)
    validate_parser.add_argument("--artifact-dir", required=False)

    core_parser = subparsers.add_parser("build-core", help="Build core benchmark tables from source configs")
    core_parser.add_argument("--config", required=True)
    core_parser.add_argument("--output-dir", required=True)

    split_parser = subparsers.add_parser("make-splits", help="Generate benchmark splits and matched decoys")
    split_parser.add_argument("--config", required=True)
    split_parser.add_argument("--artifact-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        config = load_config(args.config)
        run_ingest(config, Path(args.output_dir))
        return 0
    if args.command == "validate":
        errors = []
        if args.config:
            errors.extend(validate_config(load_config(args.config)))
        if args.artifact_dir:
            errors.extend(validate_artifact_dir(args.artifact_dir))
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Validation passed")
        return 0
    if args.command == "build-core":
        config = load_config(args.config)
        build_core(config, Path(args.output_dir))
        return 0
    if args.command == "make-splits":
        config = load_config(args.config)
        make_splits(config, Path(args.artifact_dir))
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

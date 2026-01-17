"""ctok command-line entrypoint."""

from __future__ import annotations

import argparse

from ctok.cli import bench, build, encode, eval as eval_cmd, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctok",
        description="Controlled tokenization system for structured inputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build.add_parser(subparsers)
    encode.add_parser(subparsers)
    eval_cmd.add_parser(subparsers)
    bench.add_parser(subparsers)
    train.add_parser(subparsers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

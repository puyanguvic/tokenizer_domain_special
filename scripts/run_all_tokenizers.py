#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ALGORITHMS = ("unigram", "bpe", "wordpiece")
DATASETS = ["hdfs", "phish_html", "phishing_email", "waf"]


def _parse_algorithms(raw: str) -> list[str]:
    choices = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{key}'. Options: {', '.join(ALGORITHMS)}")
        choices.append(key)
    if not choices:
        raise ValueError("No algorithms specified.")
    return choices


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run all dataset tokenizer training jobs.")
    ap.add_argument(
        "--algorithms",
        default="unigram,bpe,wordpiece",
        help="Comma-separated list of algorithms to run.",
    )
    ap.add_argument("--verify", action="store_true", help="Verify each tokenizer after training.")
    ap.add_argument("--no_auto_retry", action="store_true", help="Disable automatic retry on failure.")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override (repeatable), e.g. --override train.vocab_size=8192",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "train_tokenizer_hydra.py"
    algorithms = _parse_algorithms(args.algorithms)

    for name in DATASETS:
        for algo in algorithms:
            cmd = [
                sys.executable,
                str(script_path),
                f"dataset={name}",
                f"train.algorithm={algo}",
            ]
            if args.verify:
                cmd.append("train.verify=true")
            if args.no_auto_retry:
                cmd.append("train.auto_retry=false")
            cmd.extend(args.override)
            print(f"[run] {name}/{algo} -> {' '.join(cmd)}")
            subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

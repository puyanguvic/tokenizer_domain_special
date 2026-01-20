#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = ["hdfs", "phish_html", "phishing_email", "waf"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run all dataset Unigram training jobs.")
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
    script_path = Path(__file__).resolve().parent / "train_unigram_hydra.py"

    for name in DATASETS:
        cmd = [sys.executable, str(script_path), f"dataset={name}"]
        if args.verify:
            cmd.append("train.verify=true")
        if args.no_auto_retry:
            cmd.append("train.auto_retry=false")
        cmd.extend(args.override)
        print(f"[run] {name} -> {' '.join(cmd)}")
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

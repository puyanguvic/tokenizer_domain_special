#!/usr/bin/env python3
"""Run paper-style experiments and write outputs under results/.

This script is designed to be a practical, reproducible driver for:
- Table 1 dataset summary
- Fig 1 rate-distortion frontier
- Table 2 main results (probe + interface metrics)
- Fig 2 robustness (tokenization stability proxy)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ctok.datasets.config import DatasetConfig
from ctok.experiments.plotting import plot_rate_distortion, plot_robustness
from ctok.experiments.report import build_report
from ctok.experiments.sweeps import default_tokenizer_specs, run_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ctok paper experiments.")
    parser.add_argument("--config-root", default=None, help="Repo root (auto-detected).")
    parser.add_argument("--artifacts-dir", default="artifacts/tokenizers", help="Tokenizer artifacts root.")
    parser.add_argument("--results-dir", default="results", help="Results output root.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["waf_http", "phishing_html", "hdfs_v1"],
        help="Dataset config names under configs/datasets/.",
    )
    parser.add_argument(
        "--tokenizers",
        nargs="+",
        default=["ctok", "bpe", "unigram", "byte", "boundary_heal"],
        help="Tokenizer config names under configs/tokenizers/.",
    )
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=[16000],
        help="Vocab sizes to sweep (e.g. 8000 16000 32000).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--train-split", default="train", help="HF split for training.")
    parser.add_argument("--eval-split", default="validation", help="HF split for evaluation.")
    parser.add_argument("--max-train-samples", type=int, default=20000, help="Cap train samples.")
    parser.add_argument("--max-eval-samples", type=int, default=5000, help="Cap eval samples.")
    parser.add_argument("--max-text-chars", type=int, default=8192, help="Truncate texts for speed.")
    parser.add_argument("--force-tokenizers", action="store_true", help="Rebuild tokenizer artifacts.")
    parser.add_argument("--report-only", action="store_true", help="Only rebuild tables/figures from existing metrics.")
    parser.add_argument("--prefix-samples", type=int, default=2, help="Prefix samples per example (distortion).")
    parser.add_argument("--robust-samples", type=int, default=1000, help="Samples for robustness proxy.")
    args = parser.parse_args()

    config_root = _resolve_config_root(args.config_root)
    tokenizer_specs = [s for s in default_tokenizer_specs(config_root) if s.name in set(args.tokenizers)]

    artifacts_dir = Path(args.artifacts_dir)
    results_dir = Path(args.results_dir)
    metrics_dir = results_dir / "metrics"
    figs_dir = results_dir / "figs"

    sweeps: dict[str, dict[str, Any]] = {}
    if metrics_dir.is_dir():
        for path in metrics_dir.glob("*_sweep.json"):
            try:
                sweeps[path.stem.removesuffix("_sweep")] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

    if not args.report_only:
        for dataset_name in args.datasets:
            try:
                dataset_config = _load_dataset_config(config_root / "configs" / "datasets" / f"{dataset_name}.yaml")
                sweeps[dataset_name] = run_sweep(
                    dataset_config=dataset_config,
                    tokenizer_specs=tokenizer_specs,
                    vocab_sizes=args.vocab_sizes,
                    artifacts_dir=artifacts_dir,
                    results_dir=metrics_dir,
                    seed=args.seed,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    max_train_samples=args.max_train_samples,
                    max_eval_samples=args.max_eval_samples,
                    max_text_chars=args.max_text_chars,
                    force_tokenizers=args.force_tokenizers,
                    prefix_samples_per_example=args.prefix_samples,
                    robust_samples=args.robust_samples,
                )
            except Exception as exc:
                print(f"[skip] {dataset_name}: {exc}")

    dataset_rows: list[dict[str, Any]] = []
    all_rd: list[dict[str, Any]] = []
    all_main: list[dict[str, Any]] = []
    all_robust: list[dict[str, Any]] = []

    for dataset_name, payload in sorted(sweeps.items()):
        if not isinstance(payload, dict):
            continue
        try:
            dataset_config = _load_dataset_config(config_root / "configs" / "datasets" / f"{dataset_name}.yaml")
        except Exception:
            continue
        stats = payload.get("dataset_stats", {})
        dataset_rows.append(
            {
                "domain": dataset_name,
                "task": dataset_config.task,
                "train": stats.get("train_total") or stats.get("train_count"),
                "eval": stats.get("eval_total") or stats.get("eval_count"),
                "avg_bytes_train": stats.get("train_avg_bytes"),
                "avg_bytes_eval": stats.get("eval_avg_bytes"),
            }
        )
        all_rd.extend(payload.get("rate_distortion", []))
        all_main.extend(payload.get("main", []))
        all_robust.extend(payload.get("robustness", []))

    build_report(dataset_rows=dataset_rows, main_rows=all_main, output_dir=results_dir)
    if all_rd:
        plot_rate_distortion(
            records=all_rd,
            output_path=figs_dir / "rd_frontier.pdf",
            title="E1: Rate–distortion",
        )
    if all_robust:
        plot_robustness(
            records=all_robust,
            output_path=figs_dir / "robustness.pdf",
            title="E3: Robustness proxy",
        )

    return 0


def _resolve_config_root(value: str | None) -> Path:
    if value:
        return Path(value)
    start = Path.cwd()
    for parent in [start, *start.parents]:
        if (parent / "configs").is_dir():
            return parent
    return start


def _load_dataset_config(path: Path) -> DatasetConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset config must be a mapping: {path}")
    return DatasetConfig(**payload)


if __name__ == "__main__":
    raise SystemExit(main())

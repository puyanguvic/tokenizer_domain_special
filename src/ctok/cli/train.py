"""Model training command."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml

from ctok.datasets.config import DatasetConfig
from ctok.models.train import train_roberta


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("train", help="Train a model on a task with a tokenizer.")
    parser.add_argument(
        "--task",
        "--dataset",
        dest="task",
        default=os.getenv("CTOK_DEFAULT_TASK"),
        help="Dataset name or config path.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CTOK_DEFAULT_MODEL"),
        help="Model name or config path.",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.getenv("CTOK_DEFAULT_TOKENIZER"),
        help="Tokenizer artifact path or name.",
    )
    parser.add_argument(
        "--config-root",
        default=os.getenv("CTOK_CONFIG_ROOT"),
        help="Repo root containing configs/ (defaults to cwd search).",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.getenv("CTOK_ARTIFACTS_DIR"),
        help="Tokenizer artifacts root (default: artifacts/tokenizers).",
    )
    parser.add_argument(
        "--results-dir",
        default=os.getenv("CTOK_RESULTS_DIR"),
        help="Results root (default: results/runs).",
    )
    parser.add_argument("--output", help="Output directory (default: results root + names).")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--train-split", default="train", help="Train split name.")
    parser.add_argument("--eval-split", default="validation", help="Eval split name.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit eval samples.")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    task = _require_value(args.task, "task", "CTOK_DEFAULT_TASK")
    model = _require_value(args.model, "model", "CTOK_DEFAULT_MODEL")
    tokenizer = _require_value(args.tokenizer, "tokenizer", "CTOK_DEFAULT_TOKENIZER")

    config_root = _resolve_config_root(args.config_root)
    datasets_dir = config_root / "configs" / "datasets"
    models_dir = config_root / "configs" / "models"
    artifacts_root = _resolve_root(
        args.artifacts_dir,
        config_root / "artifacts" / "tokenizers",
    )
    results_root = _resolve_root(args.results_dir, config_root / "results" / "runs")

    dataset_path = _resolve_named_config(task, datasets_dir, "dataset")
    model_path = _resolve_named_config(model, models_dir, "model")
    tokenizer_path = _resolve_tokenizer_path(tokenizer, artifacts_root)

    dataset_config = _load_dataset_config(dataset_path)
    model_name, model_label = _load_model_info(model_path)
    tokenizer_label = Path(tokenizer_path).name

    output_dir = Path(args.output) if args.output else results_root / _output_name(
        dataset_config.name,
        model_label,
        tokenizer_label,
    )

    metrics = train_roberta(
        dataset_config=dataset_config,
        model_name=model_name,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    print(metrics)
    return 0


def _output_name(dataset: str, model: str, tokenizer: str) -> str:
    return f"{model}_{dataset}_{tokenizer}"


def _require_value(value: str | None, name: str, env_var: str) -> str:
    if value:
        return value
    raise ValueError(f"{name} is required. Pass --{name} or set {env_var}.")


def _resolve_config_root(value: str | None) -> Path:
    if value:
        return Path(value)
    return _find_repo_root(Path.cwd())


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "configs").is_dir():
            return parent
    return start


def _resolve_root(value: str | None, fallback: Path) -> Path:
    return Path(value) if value else fallback


def _resolve_named_config(name_or_path: str, config_dir: Path, label: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".yaml":
        candidate = candidate.with_suffix(".yaml")
    config_path = config_dir / candidate.name
    if config_path.exists():
        return config_path
    available = _list_configs(config_dir)
    message = f"{label} config not found: {name_or_path}."
    if available:
        message += f" Available {label}s: {', '.join(available)}"
    raise FileNotFoundError(message)


def _resolve_tokenizer_path(name_or_path: str, artifacts_root: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    candidate = artifacts_root / name_or_path
    if candidate.exists():
        return candidate
    available = _list_dirs(artifacts_root)
    message = f"Tokenizer artifact not found: {name_or_path}."
    if available:
        message += f" Available tokenizers: {', '.join(available)}"
    raise FileNotFoundError(message)


def _list_configs(config_dir: Path) -> list[str]:
    if not config_dir.is_dir():
        return []
    return sorted(path.stem for path in config_dir.glob("*.yaml"))


def _list_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _load_dataset_config(path: Path) -> DatasetConfig:
    payload = _load_yaml(path, "Dataset")
    return DatasetConfig(**payload)


def _load_model_info(path: Path) -> tuple[str, str]:
    payload = _load_yaml(path, "Model")
    hf_id = payload.get("hf_id")
    if not hf_id:
        raise ValueError("Model config must include 'hf_id'.")
    label = str(payload.get("name") or path.stem)
    return str(hf_id), label


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} config must be a mapping.")
    return payload

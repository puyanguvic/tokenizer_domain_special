#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# Ensure repo root is on sys.path when running as a script.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts import train_tokenizer as trainer


def _cfg_to_args(cfg: DictConfig) -> argparse.Namespace:
    dataset_choice = HydraConfig.get().runtime.choices.get("dataset")
    if not dataset_choice:
        raise ValueError("Missing dataset choice from Hydra overrides.")
    return argparse.Namespace(
        # Hydra stores config group choices here; avoids needing dataset.name in yaml.
        dataset=dataset_choice,
        algorithm=cfg.train.algorithm,
        split=cfg.train.split,
        max_samples=cfg.train.max_samples,
        vocab_size=cfg.train.vocab_size,
        model_max_length=cfg.train.model_max_length,
        max_piece_length=cfg.train.max_piece_length,
        wordpiece_max_input_chars=cfg.train.wordpiece_max_input_chars,
        no_progress=cfg.train.no_progress,
        max_chars=cfg.train.max_chars,
        ascii_only=cfg.train.ascii_only,
        streaming=cfg.train.streaming,
        allow_parallelism=cfg.train.allow_parallelism,
        auto_retry=cfg.train.auto_retry,
        verify=cfg.train.verify,
        _single_run=False,
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    args = _cfg_to_args(cfg)
    args.algorithm = trainer.resolve_algorithm(args.algorithm)
    if args.auto_retry:
        trainer.run_with_auto_retry(args)
    else:
        trainer.train_once(args)


if __name__ == "__main__":
    main()

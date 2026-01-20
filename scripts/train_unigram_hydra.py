#!/usr/bin/env python3
from __future__ import annotations

import argparse

import hydra
from omegaconf import DictConfig

from scripts import train_waf_unigram as trainer


def _cfg_to_args(cfg: DictConfig) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=cfg.dataset.name,
        split=cfg.train.split,
        max_samples=cfg.train.max_samples,
        vocab_size=cfg.train.vocab_size,
        model_max_length=cfg.train.model_max_length,
        max_piece_length=cfg.train.max_piece_length,
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
    if args.auto_retry:
        trainer.run_with_auto_retry(args)
    else:
        trainer.train_once(args)


if __name__ == "__main__":
    main()

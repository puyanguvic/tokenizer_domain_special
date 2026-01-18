#!/usr/bin/env python3
from __future__ import annotations

from argparse import Namespace

import hydra
from omegaconf import DictConfig, OmegaConf

from run_ctok_experiment import run


@hydra.main(version_base=None, config_path="configs", config_name="ctok_experiment")
def main(cfg: DictConfig) -> None:
    print("Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    args = Namespace(**cfg)
    run(args)


if __name__ == "__main__":
    main()

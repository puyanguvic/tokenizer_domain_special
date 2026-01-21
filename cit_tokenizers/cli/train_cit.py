from __future__ import annotations
import argparse
from .common import add_common_corpus_args, add_common_contract_args, load_contract_config
from ..cit.trainer import train_cit, CITTrainConfig

def main():
    ap = argparse.ArgumentParser()
    add_common_corpus_args(ap)
    add_common_contract_args(ap)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--model-max-length", type=int, default=512)
    args = ap.parse_args()

    contract_cfg = load_contract_config(args)
    cit_cfg = CITTrainConfig(
        vocab_size=int(args.vocab_size),
        min_frequency=int(args.min_frequency),
        model_max_length=int(args.model_max_length),
    )
    train_cit(
        corpus=args.corpus,
        outdir=args.outdir,
        cit_cfg=cit_cfg,
        contract_cfg=contract_cfg,
        fmt=args.format,
        text_key=args.text_key,
        max_samples=args.max_samples,
    )

if __name__ == "__main__":
    main()

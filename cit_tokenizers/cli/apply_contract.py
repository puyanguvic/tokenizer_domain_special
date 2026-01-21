from __future__ import annotations
import argparse, sys, json
from .common import add_common_corpus_args, add_common_contract_args, load_contract_config
from ..data import iter_text
from ..contract import Contract

def main():
    ap = argparse.ArgumentParser()
    add_common_corpus_args(ap)
    add_common_contract_args(ap)
    ap.add_argument("--out", required=False, default=None, help="Write contracted text to a .txt file. Default: stdout.")
    args = ap.parse_args()

    cfg = load_contract_config(args)
    contract = Contract(cfg)

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for s in iter_text(args.corpus, fmt=args.format, text_key=args.text_key, max_samples=args.max_samples):
            y = contract.apply(s)
            if out_f:
                out_f.write(y + "\n")
            else:
                sys.stdout.write(y + "\n")
    finally:
        if out_f:
            out_f.close()

if __name__ == "__main__":
    main()

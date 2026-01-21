from __future__ import annotations
import argparse, json, os
from ..contract import ContractConfig

def add_common_corpus_args(p: argparse.ArgumentParser):
    p.add_argument("--corpus", required=True, help="Path to corpus file.")
    p.add_argument("--format", default="txt", choices=["txt","jsonl","parquet"])
    p.add_argument("--text-key", default="text", help="Field key for jsonl/parquet.")
    p.add_argument("--max-samples", type=int, default=None)

def add_common_contract_args(p: argparse.ArgumentParser):
    p.add_argument("--contract-json", default=None, help="Optional ContractConfig JSON file. Overrides flags below.")
    p.add_argument("--no-json-serialization", action="store_true")
    p.add_argument("--no-typed-hygiene", action="store_true")
    p.add_argument("--no-numeric-buckets", action="store_true")
    p.add_argument("--long-num-min-digits", type=int, default=6)

def load_contract_config(args) -> ContractConfig:
    if args.contract_json:
        with open(args.contract_json, "r", encoding="utf-8") as f:
            return ContractConfig.from_json(f.read())
    return ContractConfig(
        enable_json_serialization=not args.no_json_serialization,
        enable_typed_hygiene=not args.no_typed_hygiene,
        enable_numeric_buckets=not args.no_numeric_buckets,
        long_num_min_digits=int(args.long_num_min_digits),
    )

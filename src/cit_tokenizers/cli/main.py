from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ..config import CITBuildConfig, CITTrainerConfig
from ..interface.contract import Contract, ContractConfig
from ..io.data import iter_text
from ..utils.logging import configure_logging
from ..cit.trainer import CITTrainer
from ..cit.validate import validate_artifact
from ..cit.runtime import load_artifact
from ..baselines.bpe_hygiene.trainer import train_bpe_hygiene
from ..baselines.wordpiece_hygiene.trainer import train_wordpiece_hygiene
from ..baselines.unigram_hygiene.trainer import train_unigram_hygiene
from ..artifacts.export import export_cit_as_hf_dir
from ..io.hf_datasets import DatasetPullSpec, pull_to_parquet


def _add_global_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--log-level",
        default=None,
        help="Logging level (e.g., INFO, DEBUG). Also respects CIT_LOG_LEVEL env var.",
    )


def _add_contract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--contract-json", default=None, help="Path to ContractConfig JSON")
    p.add_argument("--no-json-serialization", action="store_true")
    p.add_argument("--no-typed-hygiene", action="store_true")
    p.add_argument("--no-numeric-buckets", action="store_true")
    p.add_argument("--long-num-min-digits", type=int, default=6)


def _load_contract(args: argparse.Namespace) -> ContractConfig:
    if args.contract_json:
        return ContractConfig.from_json(Path(args.contract_json).read_text(encoding="utf-8"))
    return ContractConfig(
        enable_json_serialization=not args.no_json_serialization,
        enable_typed_hygiene=not args.no_typed_hygiene,
        enable_numeric_buckets=not args.no_numeric_buckets,
        long_num_min_digits=int(args.long_num_min_digits),
    )


def _add_corpus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", required=True, help="Path to corpus file")
    p.add_argument("--format", default="txt", choices=["txt", "jsonl", "parquet"])
    p.add_argument("--text-key", default="text")
    p.add_argument("--max-samples", type=int, default=None)


def _add_cit_trainer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="Optional CITBuildConfig JSON file")
    p.add_argument("--outdir", required=True)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--min-freq", type=int, default=10)
    p.add_argument("--preset", default="default", choices=["default", "http", "waf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lambda-rd", type=float, default=0.0)
    p.add_argument("--distortion-mode", default="none", choices=["none", "boundary_penalty"])
    p.add_argument("--boundary-penalty", type=float, default=1.0)


def cmd_train_cit(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    build_cfg: Optional[CITBuildConfig] = None
    if args.config:
        build_cfg = CITBuildConfig.from_json(Path(args.config).read_text(encoding="utf-8"))

    contract_cfg = _load_contract(args) if build_cfg is None else build_cfg.contract

    trainer_cfg = (
        CITTrainerConfig(
            vocab_size=int(args.vocab_size),
            min_freq=int(args.min_freq),
            preset=str(args.preset),
            seed=int(args.seed),
            lambda_rd=float(args.lambda_rd),
            distortion_mode=str(args.distortion_mode),
            boundary_penalty=float(args.boundary_penalty),
            sample_texts=int(args.max_samples) if args.max_samples is not None else None,
        )
        if build_cfg is None
        else build_cfg.trainer
    )

    # Persist unified build config into artifact meta for reproducibility.
    build_cfg = (
        CITBuildConfig(
            trainer=trainer_cfg,
            contract=contract_cfg,
            corpus_format=str(args.format),
            text_key=str(args.text_key),
            max_samples=int(args.max_samples) if args.max_samples is not None else None,
        )
        if build_cfg is None
        else build_cfg
    )

    texts = iter_text(args.corpus, fmt=str(args.format), text_key=str(args.text_key), max_samples=args.max_samples)
    trainer = CITTrainer(build_config=build_cfg)
    trainer.train_from_iterator(texts, args.outdir)


def cmd_validate(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)

    art = load_artifact(Path(args.artifact_dir))
    contract = Contract(art.contract)
    typed_symbols = contract.typed_symbols()

    issues = validate_artifact(
        art,
        typed_symbols,
        max_long_hex_fraction=args.max_long_hex_fraction,
        max_long_hex_count=args.max_long_hex_count,
        max_b64_fraction=args.max_b64_fraction,
        max_b64_count=args.max_b64_count,
    )
    if issues:
        for iss in issues:
            print(f"[FAIL] {iss.code}: {iss.message}")
        raise SystemExit(2)
    print("[OK] artifact is valid")


def cmd_export_hf(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    export_cit_as_hf_dir(
        artifact_dir=str(args.artifact_dir),
        outdir=str(args.outdir),
        model_max_length=int(args.model_max_length),
        overwrite=bool(args.overwrite),
    )
    print(f"[OK] exported to: {args.outdir}")


def cmd_dataset_pull(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    spec = DatasetPullSpec(
        dataset=str(args.dataset),
        subset=str(args.subset) if args.subset is not None else None,
        split=str(args.split),
        text_key=str(args.text_key),
        label_key=str(args.label_key) if args.label_key is not None else None,
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
    )
    out = pull_to_parquet(spec, out_path=str(args.out), shuffle=bool(args.shuffle), seed=int(args.seed))
    print(f"[OK] wrote: {out}")


def _add_baseline_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--outdir", required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--min-freq", type=int, default=10)
    p.add_argument("--model-max-length", type=int, default=512)


def cmd_train_bpeh(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    contract_cfg = _load_contract(args)
    train_bpe_hygiene(
        corpus=args.corpus,
        outdir=args.outdir,
        vocab_size=int(args.vocab_size),
        contract_cfg=contract_cfg,
        fmt=str(args.format),
        text_key=str(args.text_key),
        max_samples=args.max_samples,
        min_frequency=int(args.min_freq),
        model_max_length=int(args.model_max_length),
    )


def cmd_train_wordpieceh(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    contract_cfg = _load_contract(args)
    train_wordpiece_hygiene(
        corpus=args.corpus,
        outdir=args.outdir,
        vocab_size=int(args.vocab_size),
        contract_cfg=contract_cfg,
        fmt=str(args.format),
        text_key=str(args.text_key),
        max_samples=args.max_samples,
        min_frequency=int(args.min_freq),
        model_max_length=int(args.model_max_length),
    )


def cmd_train_unigramh(args: argparse.Namespace) -> None:
    configure_logging(args.log_level)
    contract_cfg = _load_contract(args)
    train_unigram_hygiene(
        corpus=args.corpus,
        outdir=args.outdir,
        vocab_size=int(args.vocab_size),
        contract_cfg=contract_cfg,
        fmt=str(args.format),
        text_key=str(args.text_key),
        max_samples=args.max_samples,
        min_frequency=int(args.min_freq),
        model_max_length=int(args.model_max_length),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cit", description="CIT tokenizer toolchain")
    _add_global_args(p)

    sub = p.add_subparsers(dest="cmd", required=True)

    # Train
    train = sub.add_parser("train", help="Train a tokenizer")
    train_sub = train.add_subparsers(dest="algo", required=True)

    cit_p = train_sub.add_parser("cit", help="Train CIT tokenizer")
    _add_corpus_args(cit_p)
    _add_contract_args(cit_p)
    _add_cit_trainer_args(cit_p)
    cit_p.set_defaults(func=cmd_train_cit)

    bpe_p = train_sub.add_parser("bpeh", help="Train BPE+Hygiene baseline")
    _add_corpus_args(bpe_p)
    _add_contract_args(bpe_p)
    _add_baseline_args(bpe_p)
    bpe_p.set_defaults(func=cmd_train_bpeh)

    wp_p = train_sub.add_parser("wordpieceh", help="Train WordPiece+Hygiene baseline")
    _add_corpus_args(wp_p)
    _add_contract_args(wp_p)
    _add_baseline_args(wp_p)
    wp_p.set_defaults(func=cmd_train_wordpieceh)

    uni_p = train_sub.add_parser("unigramh", help="Train Unigram+Hygiene baseline")
    _add_corpus_args(uni_p)
    _add_contract_args(uni_p)
    _add_baseline_args(uni_p)
    uni_p.set_defaults(func=cmd_train_unigramh)

    # Validate
    val = sub.add_parser("validate", help="Validate an artifact directory")
    val.add_argument("--artifact-dir", required=True)
    val.add_argument("--max-long-hex-fraction", type=float, default=0.005, help="Fail if long-hex tokens exceed this fraction of vocab.")
    val.add_argument("--max-long-hex-count", type=int, default=32, help="Fail if long-hex tokens exceed this absolute count.")
    val.add_argument("--max-b64-fraction", type=float, default=0.002, help="Fail if base64-like tokens exceed this fraction of vocab.")
    val.add_argument("--max-b64-count", type=int, default=16, help="Fail if base64-like tokens exceed this absolute count.")
    val.set_defaults(func=cmd_validate)

    # Export
    exp = sub.add_parser("export-hf", help="Export a CIT artifact as an HF-style folder")
    exp.add_argument("--artifact-dir", required=True)
    exp.add_argument("--outdir", required=True)
    exp.add_argument("--model-max-length", type=int, default=512)
    exp.add_argument("--overwrite", action="store_true")
    exp.set_defaults(func=cmd_export_hf)

    # Dataset helpers
    ds = sub.add_parser("dataset", help="Dataset helpers")
    ds_sub = ds.add_subparsers(dest="ds_cmd", required=True)
    pull = ds_sub.add_parser("pull", help="Download a HF dataset and save as parquet")
    pull.add_argument("--dataset", required=True, help="HF dataset name (e.g., 'imdb')")
    pull.add_argument("--subset", default=None, help="Optional subset/config name")
    pull.add_argument("--split", default="train")
    pull.add_argument("--text-key", default="text")
    pull.add_argument("--label-key", default=None)
    pull.add_argument("--max-samples", type=int, default=None)
    pull.add_argument("--shuffle", action="store_true")
    pull.add_argument("--seed", type=int, default=0)
    pull.add_argument("--out", required=True, help="Output parquet path")
    pull.set_defaults(func=cmd_dataset_pull)

    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

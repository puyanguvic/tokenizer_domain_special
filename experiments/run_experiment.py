#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import random
from typing import List, Tuple
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure src/ layout is importable when running from the repo checkout.
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cit_tokenizers.interface.contract import ContractConfig
from cit_tokenizers.corpus import export_dataset_corpus, resolve_dataset_key, resolve_dataset_path
from cit_tokenizers.io.data import iter_text
from cit_tokenizers.cit.trainer import CITTrainer, CITTrainerConfig
from cit_tokenizers.config import CITBuildConfig
from cit_tokenizers.baselines.bpe_hygiene.trainer import train_bpe_hygiene
from cit_tokenizers.baselines.wordpiece_hygiene.trainer import train_wordpiece_hygiene
from cit_tokenizers.baselines.unigram_hygiene.trainer import train_unigram_hygiene


ALGORITHMS = ("cit", "bpeh", "wordpieceh", "unigramh")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export a dataset to JSONL and train a tokenizer.")
    ap.add_argument("--dataset", required=True, help="Dataset key (e.g., hdfs, phish_html, phishing_email, waf).")
    ap.add_argument("--split", default="train", help="Dataset split to use.")
    ap.add_argument("--algorithm", required=True, choices=ALGORITHMS)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--min-frequency", type=int, default=10)
    ap.add_argument("--model-max-length", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--text-key", default=None, help="Override the text field used during export.")
    ap.add_argument("--no-formatter", action="store_true", help="Disable dataset-specific formatting.")
    ap.add_argument("--streaming", action="store_true", help="Stream from HF instead of full download.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing corpus JSONL if present.")
    ap.add_argument("--name", default=None, help="Tokenizer output dir name under tokenizers/.")
    ap.add_argument("--outdir", default=None, help="Override full output directory path.")
    ap.add_argument("--no-json-serialization", action="store_true")
    ap.add_argument("--no-typed-hygiene", action="store_true")
    ap.add_argument("--no-numeric-buckets", action="store_true")
    ap.add_argument("--long-num-min-digits", type=int, default=6)
    ap.add_argument(
        "--lowercase",
        default="none",
        choices=("none", "ascii"),
        help="Optional case normalization applied in the contract (default: none).",
    )
    ap.add_argument("--cit-preset", choices=("default", "http", "waf"), default="default")
    ap.add_argument("--cit-len-min", type=int, default=None)
    ap.add_argument("--cit-len-max", type=int, default=24)
    char_group = ap.add_mutually_exclusive_group()
    char_group.add_argument("--cit-include-chars", dest="cit_include_chars", action="store_true")
    char_group.add_argument("--no-cit-include-chars", dest="cit_include_chars", action="store_false")
    ap.set_defaults(cit_include_chars=None)
    ap.add_argument("--cit-lambda-rd", type=float, default=0.0)
    ap.add_argument(
        "--cit-distortion-mode",
        default="none",
        choices=("none", "boundary_penalty"),
    )
    ap.add_argument("--cit-boundary-penalty", type=float, default=1.0)
    ap.add_argument("--cit-sample-texts", type=int, default=200000, help="Set 0 to disable sampling.")
    ap.add_argument("--cit-symbol-ngram-min", type=int, default=2)
    ap.add_argument("--cit-symbol-ngram-max", type=int, default=None)
    ap.add_argument("--wordpiece-prefix", default="##")
    ap.add_argument("--verify", action="store_true", help="Verify tokenizer by tokenizing sampled examples.")
    ap.add_argument("--verify-n", type=int, default=5, help="Number of dataset examples to print for verification.")
    ap.add_argument("--verify-seed", type=int, default=0, help="Random seed for verification sampling.")
    ap.add_argument(
        "--verify-source",
        choices=("corpus", "example"),
        default="corpus",
        help="Verify source: sample from exported corpus JSONL, or use --example string.",
    )
    ap.add_argument("--example", default=None, help="Example text to tokenize after training (when --verify-source=example).")

    return ap.parse_args()


def _resolve_outdir(repo_root: Path, dataset_key: str, algorithm: str, name: Optional[str], outdir: Optional[str]) -> Path:
    if outdir:
        return Path(outdir)
    if name:
        return repo_root / "tokenizers" / name
    return repo_root / "tokenizers" / f"{dataset_key}_{algorithm}_tokenizer"


def _tokenizer_exists(outdir: Path, algorithm: str) -> bool:
    if not outdir.exists():
        return False
    if algorithm == "cit":
        return (outdir / "cit_artifact.json").exists()
    return (outdir / "tokenizer.json").exists()


def _ensure_corpus(
    dataset_key: str,
    split: str,
    *,
    cache_dir: Path,
    corpus_dir: Path,
    text_key: Optional[str],
    max_samples: Optional[int],
    streaming: bool,
    use_formatter: bool,
    overwrite: bool,
) -> Path:
    dataset_path = resolve_dataset_path(dataset_key)
    corpus_name = f"{dataset_path.replace('/', '__')}_{split}.jsonl"
    corpus_path = corpus_dir / corpus_name
    if corpus_path.exists() and not overwrite:
        if max_samples is not None:
            print(f"[info] Using existing corpus at {corpus_path} (max_samples ignored).")
        return corpus_path
    return export_dataset_corpus(
        dataset_key,
        split=split,
        out_dir=corpus_dir,
        filename=corpus_name,
        text_key=text_key,
        cache_dir=cache_dir,
        max_samples=max_samples,
        streaming=streaming,
        use_formatter=use_formatter,
    )


def _sample_texts_from_corpus(corpus_path: Path, *, n: int, seed: int) -> Tuple[List[str], int]:
    """
    Reservoir-sample up to n texts from a potentially huge JSONL corpus without loading into memory.
    Returns (samples, total_seen).
    """
    rng = random.Random(seed)
    samples: List[str] = []
    seen = 0

    for text in iter_text(str(corpus_path), fmt="jsonl", text_key="text", max_samples=None):
        seen += 1
        if len(samples) < n:
            samples.append(text)
        else:
            # replace elements with decreasing probability
            j = rng.randrange(seen)
            if j < n:
                samples[j] = text

    return samples, seen


def _contract_from_args(args: argparse.Namespace) -> ContractConfig:
    return ContractConfig(
        enable_json_serialization=not args.no_json_serialization,
        enable_typed_hygiene=not args.no_typed_hygiene,
        enable_numeric_buckets=not args.no_numeric_buckets,
        long_num_min_digits=int(args.long_num_min_digits),
        lowercase_mode=str(getattr(args, "lowercase", "none")),
    )


def _update_cit_max_length(outdir: Path, model_max_length: int) -> None:
    cfg_path = outdir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data["model_max_length"] = int(model_max_length)
    cfg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cache_dir = REPO_ROOT / "datasets" / "hf_cache"
    corpus_dir = REPO_ROOT / "datasets" / "corpus"
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = resolve_dataset_key(args.dataset)
    outdir = _resolve_outdir(REPO_ROOT, dataset_key, args.algorithm, args.name, args.outdir)
    already_trained = _tokenizer_exists(outdir, args.algorithm)

    if args.verify and already_trained:
        print(f"[skip] Tokenizer exists at {outdir}; skipping training.")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"[1/3] Exporting corpus for {dataset_key}:{args.split}")
        corpus_path = _ensure_corpus(
            dataset_key,
            args.split,
            cache_dir=cache_dir,
            corpus_dir=corpus_dir,
            text_key=args.text_key,
            max_samples=args.max_samples,
            streaming=args.streaming,
            use_formatter=not args.no_formatter,
            overwrite=args.overwrite,
        )
        print(f"[1/3] Corpus ready at {corpus_path}")

        contract_cfg = _contract_from_args(args)
        print(f"[2/3] Training {args.algorithm} tokenizer -> {outdir}")

        if args.algorithm == "cit":
            sample_texts = int(args.cit_sample_texts)
            if sample_texts <= 0:
                sample_texts = None
            cfg = CITTrainerConfig(
                vocab_size=int(args.vocab_size),
                min_freq=int(args.min_frequency),
                len_min=args.cit_len_min,
                len_max=int(args.cit_len_max),
                lambda_rd=float(args.cit_lambda_rd),
                distortion_mode=args.cit_distortion_mode,
                boundary_penalty=float(args.cit_boundary_penalty),
                sample_texts=sample_texts,
                preset=args.cit_preset,
                include_char_vocab=args.cit_include_chars,
                symbol_ngram_min_len=int(args.cit_symbol_ngram_min),
                symbol_ngram_max_len=args.cit_symbol_ngram_max,
            )
            build_cfg = CITBuildConfig(
                trainer=cfg,
                contract=contract_cfg,
                corpus_format="jsonl",
                text_key="text",
                max_samples=int(args.max_samples) if args.max_samples is not None else None,
            )
            trainer = CITTrainer(build_config=build_cfg)
            texts = iter_text(
                str(corpus_path),
                fmt="jsonl",
                text_key="text",
                max_samples=args.max_samples,
            )
            trainer.train_from_iterator(texts, outdir)
            _update_cit_max_length(outdir, args.model_max_length)
        elif args.algorithm == "bpeh":
            train_bpe_hygiene(
                corpus=str(corpus_path),
                outdir=str(outdir),
                vocab_size=int(args.vocab_size),
                contract_cfg=contract_cfg,
                fmt="jsonl",
                text_key="text",
                max_samples=args.max_samples,
                min_frequency=int(args.min_frequency),
                model_max_length=int(args.model_max_length),
            )
        elif args.algorithm == "wordpieceh":
            train_wordpiece_hygiene(
                corpus=str(corpus_path),
                outdir=str(outdir),
                vocab_size=int(args.vocab_size),
                contract_cfg=contract_cfg,
                fmt="jsonl",
                text_key="text",
                max_samples=args.max_samples,
                min_frequency=int(args.min_frequency),
                continuing_subword_prefix=args.wordpiece_prefix,
                model_max_length=int(args.model_max_length),
            )
        elif args.algorithm == "unigramh":
            train_unigram_hygiene(
                corpus=str(corpus_path),
                outdir=str(outdir),
                vocab_size=int(args.vocab_size),
                contract_cfg=contract_cfg,
                fmt="jsonl",
                text_key="text",
                max_samples=args.max_samples,
                min_frequency=int(args.min_frequency),
                model_max_length=int(args.model_max_length),
            )
        else:
            raise ValueError(f"Unknown algorithm '{args.algorithm}'")

        print(f"[3/3] Saved tokenizer to {outdir}")


        if args.verify:
            if args.algorithm == "cit":
                from cit_tokenizers.tokenization_cit import CITTokenizer
                tok = CITTokenizer.from_pretrained(str(outdir))
                unk_token = getattr(tok, "unk_token", "[UNK]")
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(outdir), use_fast=True)
            unk_token = getattr(tok, "unk_token", "[UNK]")

        if args.verify_source == "example":
            example = args.example or "GET /index.html?x=1 HTTP/1.1"
            texts = [example]
            print(f"[verify] source=example")
        else:
            # Use the exported corpus for the chosen split
            # If tokenizer existed and we skipped training, we still need to locate the corpus path.
            if "corpus_path" not in locals():
                corpus_path = _ensure_corpus(
                    dataset_key,
                    args.split,
                    cache_dir=cache_dir,
                    corpus_dir=corpus_dir,
                    text_key=args.text_key,
                    max_samples=args.max_samples,
                    streaming=args.streaming,
                    use_formatter=not args.no_formatter,
                    overwrite=False,
                )
            texts, seen = _sample_texts_from_corpus(
                corpus_path,
                n=int(args.verify_n),
                seed=int(args.verify_seed),
            )
            print(f"[verify] source=corpus split={args.split} n={len(texts)} seen={seen} corpus={corpus_path}")

        for i, x in enumerate(texts):
            tokens = tok.tokenize(x)
            unk_cnt = sum(1 for t in tokens if t == unk_token)
            denom = max(1, len(tokens))
            print("\n" + "=" * 80)
            print(f"[verify:{i}] raw[:200] = {x[:200]!r}")
            print(f"[verify:{i}] tokens_len = {len(tokens)}  UNK = {unk_cnt}/{denom} ({unk_cnt/denom:.3f})")
            print(f"[verify:{i}] tokens[:120] = {tokens[:120]}")

if __name__ == "__main__":
    main()

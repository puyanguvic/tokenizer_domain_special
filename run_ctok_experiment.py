#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

PRESET_KEYS = {
    "logfit-project/HDFS_v1": {"text_key": "content", "label_key": "anomaly"},
}


def resolve_keys(dataset_name: str, text_key: str, label_key: str) -> Tuple[str, str | None]:
    preset = PRESET_KEYS.get(dataset_name, {})
    resolved_text = text_key or preset.get("text_key", "")
    resolved_label = label_key or preset.get("label_key", "")
    if not resolved_text:
        raise SystemExit(
            f"Missing text_key for dataset '{dataset_name}'. Provide --text_key or add to PRESET_KEYS."
        )
    return resolved_text, (resolved_label if resolved_label else None)


def iter_dataset_rows(dataset: Iterable[dict]) -> Iterable[dict]:
    for row in dataset:
        if isinstance(row, dict):
            yield row
        else:
            yield dict(row)


def write_jsonl_corpus(
    dataset: Iterable[dict],
    out_path: Path,
    text_key: str,
    label_key: str | None,
    max_samples: int | None,
    preview_count: int,
) -> Tuple[List[str], int, int, List[str]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    previews: List[str] = []
    first_keys: List[str] = []
    missing_text = 0
    n = 0
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    total = None
    try:
        total = len(dataset)  # type: ignore[arg-type]
    except Exception:
        total = None
    if total is not None and max_samples is not None:
        total = min(total, max_samples)

    iterator = iter_dataset_rows(dataset)
    if tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="Writing corpus", unit="rows")
    with out_path.open("w", encoding="utf-8") as f:
        for row in iterator:
            if not first_keys:
                first_keys = sorted(row.keys())
            if text_key not in row:
                missing_text += 1
                continue
            text = row[text_key]
            if text is None:
                missing_text += 1
                continue
            obj = {text_key: text}
            if label_key and label_key in row:
                obj[label_key] = row[label_key]
            f.write(json.dumps(obj, ensure_ascii=True) + "\n")
            if len(previews) < preview_count:
                previews.append(str(text))
            n += 1
            if max_samples is not None and n >= max_samples:
                break
    print(f"Wrote {n} samples to {out_path}")
    if missing_text:
        print(f"Skipped {missing_text} rows missing text_key='{text_key}'")
    if first_keys:
        print(f"Detected row keys: {first_keys}")
    return previews, n, missing_text, first_keys


def build_ctok_artifact(
    corpus_path: Path,
    outdir: Path,
    text_key: str,
    label_key: str | None,
    args: argparse.Namespace,
) -> None:
    repo_root = Path(__file__).resolve().parent
    build_script = repo_root / "ctok_core" / "build_ctok_from_corpus.py"
    if not build_script.exists():
        raise FileNotFoundError(f"Missing builder: {build_script}")

    cmd = [
        sys.executable,
        str(build_script),
        "--corpus",
        str(corpus_path),
        "--format",
        "jsonl",
        "--text_key",
        text_key,
        "--label_key",
        "" if label_key is None else label_key,
        "--outdir",
        str(outdir),
        "--vocab_size",
        str(args.vocab_size),
        "--max_len",
        str(args.max_len),
        "--min_freq",
        str(args.min_freq),
        "--max_samples",
        str(args.max_samples),
        "--semantic_mode",
        args.semantic_mode,
        "--lambda_sem",
        str(args.lambda_sem),
        "--semantic_top_k",
        str(args.semantic_top_k),
        "--model_max_length",
        str(args.model_max_length),
    ]
    if args.use_ascii_base:
        cmd.append("--use_ascii_base")
    if args.emit_code:
        cmd.append("--emit_code")

    print("Building CTok artifact...")
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def run(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit("Missing dependency: pip install datasets") from exc

    load_kwargs = {"split": args.split, "streaming": args.streaming, "cache_dir": args.cache_dir}
    if args.config:
        print(f"Loading dataset={args.dataset} config={args.config} split={args.split} streaming={args.streaming}")
        dataset = load_dataset(args.dataset, args.config, **load_kwargs)
    else:
        print(f"Loading dataset={args.dataset} split={args.split} streaming={args.streaming}")
        dataset = load_dataset(args.dataset, **load_kwargs)

    if hasattr(dataset, "features") and dataset.features:
        print(f"Dataset columns: {sorted(dataset.features.keys())}")

    text_key, label_key = resolve_keys(args.dataset, args.text_key, args.label_key)
    tokenizer_name = args.tokenizer_name or f"{args.dataset.replace('/', '__')}_{args.split}"
    tokenizer_root = Path(args.tokenizer_root)
    outdir = Path(args.outdir) if args.outdir else tokenizer_root / tokenizer_name
    corpus_out = Path(args.corpus_out) if args.corpus_out else Path("datasets") / "corpus" / f"{tokenizer_name}.jsonl"
    label_key = label_key
    outdir.mkdir(parents=True, exist_ok=True)
    corpus_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {args.cache_dir}")
    print(f"Corpus path: {corpus_out}")
    print(f"Tokenizer outdir: {outdir}")
    print(f"Using text_key='{text_key}' label_key='{label_key or ''}'")

    if not args.streaming and args.sample_ratio < 1.0:
        total = len(dataset)
        keep = max(1, int(total * args.sample_ratio))
        print(f"Sampling {keep}/{total} rows (ratio={args.sample_ratio}) with seed={args.seed}")
        dataset = dataset.shuffle(seed=args.seed).select(range(keep))
    elif args.streaming and args.sample_ratio < 1.0:
        print("Streaming mode ignores --sample_ratio; use --max_samples to limit size.")

    if corpus_out.exists() and not args.force_corpus:
        print(f"Reusing existing corpus: {corpus_out}")
        previews = []
        kept = 1
        first_keys = []
    else:
        previews, kept, _, first_keys = write_jsonl_corpus(
            dataset,
            corpus_out,
            text_key=text_key,
            label_key=label_key,
            max_samples=args.max_samples,
            preview_count=args.preview,
        )
    if kept == 0:
        msg = f"No samples written; text_key='{text_key}' not found or empty."
        if first_keys:
            msg += f" Available keys example: {first_keys}"
        raise SystemExit(msg)

    build_ctok_artifact(
        corpus_path=corpus_out,
        outdir=outdir,
        text_key=text_key,
        label_key=label_key,
        args=args,
    )

    if previews:
        from transformers import AutoTokenizer

        print("Loading tokenizer from artifact...")
        tok = AutoTokenizer.from_pretrained(str(outdir), trust_remote_code=True)
        print("Tokenizer:", type(tok))
        for i, text in enumerate(previews, start=1):
            print(f"[{i}] {text}")
            print(tok.tokenize(text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="HF dataset name, e.g., logfit-project/HDFS_v1")
    ap.add_argument("--config", default="", help="Optional HF dataset config name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_key", default="", help="Optional text field; if empty, use preset when available")
    ap.add_argument("--label_key", default="", help="Optional label field; if empty, use preset when available")
    ap.add_argument("--outdir", default="", help="Tokenizer output directory (default: tokenizers/<tokenizer_name>)")
    ap.add_argument("--tokenizer_name", default="", help="Tokenizer folder name under tokenizers/")
    ap.add_argument("--tokenizer_root", default="tokenizers")
    ap.add_argument("--corpus_out", default="", help="Where to write jsonl corpus (default: datasets/corpus/<tokenizer_name>.jsonl)")
    ap.add_argument("--max_samples", type=int, default=200000)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--preview", type=int, default=3, help="Show tokenization previews after build")
    ap.add_argument("--sample_ratio", type=float, default=0.1, help="Use only a fraction of the split (0-1]")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="datasets/hf_cache")
    ap.add_argument("--force_corpus", action="store_true")

    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_freq", type=int, default=50)
    ap.add_argument("--semantic_mode", choices=["none", "mi"], default="none")
    ap.add_argument("--lambda_sem", type=float, default=0.0)
    ap.add_argument("--semantic_top_k", type=int, default=50000)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--use_ascii_base", action="store_true")
    ap.add_argument("--emit_code", action="store_true")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

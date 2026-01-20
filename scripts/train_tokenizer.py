#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
ALGORITHMS = ("unigram", "bpe", "wordpiece")
ALGORITHM_LABELS = {
    "unigram": "Unigram",
    "bpe": "BPE",
    "wordpiece": "WordPiece",
}
AUTO_VOCAB_GROWTH_FACTOR = 2.0
AUTO_VOCAB_MIN_INCREMENT = 2048
AUTO_VOCAB_MAX_GROWTHS = 3


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train a tokenizer (Unigram/BPE/WordPiece) from a supported HF dataset."
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Dataset name: hdfs, phish_html, phishing_email, waf (aliases supported).",
    )
    ap.add_argument(
        "--algorithm",
        default="unigram",
        type=str.lower,
        choices=sorted(ALGORITHMS),
        help="Tokenizer algorithm to train.",
    )
    ap.add_argument("--split", default="train", help="Dataset split to use.")
    ap.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of rows.")
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--max_piece_length", type=int, default=16, help="Unigram max piece length.")
    ap.add_argument(
        "--wordpiece_max_input_chars",
        type=int,
        default=100,
        help="WordPiece max input characters per word.",
    )
    ap.add_argument("--no_progress", action="store_true", help="Disable trainer progress output.")
    ap.add_argument(
        "--max_chars",
        type=int,
        default=None,
        help="Truncate each sample to this many characters before training.",
    )
    ap.add_argument(
        "--ascii_only",
        action="store_true",
        help="Drop non-ASCII characters before training to keep the character set small.",
    )
    ap.add_argument("--streaming", action="store_true", help="Stream dataset instead of fully loading it.")
    ap.add_argument(
        "--allow_parallelism",
        action="store_true",
        help="Allow tokenizers to use thread parallelism (can be unstable on some systems).",
    )
    retry_group = ap.add_mutually_exclusive_group()
    retry_group.add_argument(
        "--auto_retry",
        dest="auto_retry",
        action="store_true",
        help="Automatically retry with safer settings if training fails.",
    )
    retry_group.add_argument(
        "--no_auto_retry",
        dest="auto_retry",
        action="store_false",
        help="Disable automatic retry on failure.",
    )
    ap.set_defaults(auto_retry=True)
    ap.add_argument("--_single_run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--verify", action="store_true", help="Load the tokenizer via AutoTokenizer after training.")
    return ap.parse_args()


def _to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_to_str(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _clean_text(text: str, *, ascii_only: bool, max_chars: Optional[int]) -> str:
    if not text:
        return ""
    if max_chars is not None and max_chars > 0:
        text = text[:max_chars]
    if ascii_only:
        text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


def _pick_first(row: Mapping[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            val = _to_str(row[key]).strip()
            if val:
                return val
    return ""


def _fallback_text(row: Mapping[str, object]) -> str:
    for key, val in row.items():
        if key.lower() in {"label", "anomaly", "target", "class"}:
            continue
        text = _to_str(val).strip()
        if text:
            return text
    return ""


def format_waf(row: Mapping[str, object]) -> str:
    method = _pick_first(row, ["method"])
    url = _pick_first(row, ["url"])
    proto = _pick_first(row, ["protocol"])
    headers = _pick_first(row, ["headers"])
    body = _pick_first(row, ["body"])
    return (
        f"<METHOD> {method}\n"
        f"<URL> {url}\n"
        f"<PROT> {proto}\n"
        f"<HDR>\n{headers}\n"
        f"<BODY>\n{body}\n"
    ).strip()


def format_hdfs(row: Mapping[str, object]) -> str:
    content = _pick_first(row, ["content"])
    date = _pick_first(row, ["date"])
    time = _pick_first(row, ["time"])
    level = _pick_first(row, ["level"])
    component = _pick_first(row, ["component"])
    block_id = _pick_first(row, ["block_id"])
    parts = []
    if date or time:
        parts.append(f"<TS> {date} {time}".strip())
    if level:
        parts.append(f"<LEVEL> {level}")
    if component:
        parts.append(f"<COMP> {component}")
    if block_id:
        parts.append(f"<BLK> {block_id}")
    if content:
        parts.append(f"<MSG> {content}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


def format_phish_html(row: Mapping[str, object]) -> str:
    html = _pick_first(row, ["text"])
    path = _pick_first(row, ["path"])
    year = _pick_first(row, ["year"])
    parts = []
    if path:
        parts.append(f"<PATH> {path}")
    if year:
        parts.append(f"<YEAR> {year}")
    if html:
        parts.append(f"<HTML>\n{html}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


def format_phishing_email(row: Mapping[str, object]) -> str:
    subject = _pick_first(row, ["subject"])
    sender = _pick_first(row, ["sender"])
    receiver = _pick_first(row, ["receiver"])
    body = _pick_first(row, ["text"])
    date = _pick_first(row, ["date"])
    dataset_name = _pick_first(row, ["dataset_name"])
    urls = _pick_first(row, ["urls"])
    parts = []
    if subject:
        parts.append(f"<SUBJ> {subject}")
    if sender:
        parts.append(f"<FROM> {sender}")
    if receiver:
        parts.append(f"<TO> {receiver}")
    if date:
        parts.append(f"<DATE> {date}")
    if dataset_name:
        parts.append(f"<DATASET> {dataset_name}")
    if urls:
        parts.append(f"<URLS> {urls}")
    if body:
        parts.append(f"<BODY>\n{body}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    formatter: Callable[[Mapping[str, object]], str]


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "hdfs": DatasetSpec(path="logfit-project/HDFS_v1", formatter=format_hdfs),
    "phish_html": DatasetSpec(path="puyang2025/phish_html", formatter=format_phish_html),
    "phishing_email": DatasetSpec(
        path="puyang2025/seven-phishing-email-datasets", formatter=format_phishing_email
    ),
    "waf": DatasetSpec(path="puyang2025/waf_data_v2", formatter=format_waf),
}

DATASET_ALIASES: Dict[str, str] = {
    "hdfs_v1": "hdfs",
    "hdfs-v1": "hdfs",
    "phish-html": "phish_html",
    "phishing-email": "phishing_email",
    "waf_v2": "waf",
    "waf-v2": "waf",
}


def resolve_dataset(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in DATASET_REGISTRY:
        return key
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(f"Unknown dataset '{name}'. Options: {', '.join(sorted(DATASET_REGISTRY))}")


def resolve_algorithm(name: str) -> str:
    key = name.strip().lower()
    if key in ALGORITHMS:
        return key
    raise ValueError(f"Unknown algorithm '{name}'. Options: {', '.join(ALGORITHMS)}")


def _clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    data = dict(vars(args))
    data.update(updates)
    return argparse.Namespace(**data)


def _add_attempt(
    attempts: list[argparse.Namespace],
    base: argparse.Namespace,
    **updates,
) -> argparse.Namespace:
    if all(getattr(base, key) == val for key, val in updates.items()):
        return base
    nxt = _clone_args(base, **updates)
    attempts.append(nxt)
    return nxt


def _build_attempts(args: argparse.Namespace) -> list[argparse.Namespace]:
    attempts: list[argparse.Namespace] = []
    current = _clone_args(args)
    attempts.append(current)

    if current.max_chars is None or current.max_chars > 4000:
        current = _add_attempt(attempts, current, max_chars=4000)

    if current.vocab_size > 20000:
        current = _add_attempt(attempts, current, vocab_size=20000)
    if current.vocab_size > 10000:
        current = _add_attempt(attempts, current, vocab_size=10000)

    if current.max_samples is None or current.max_samples > 50000:
        current = _add_attempt(attempts, current, max_samples=50000)

    if not current.ascii_only:
        current = _add_attempt(attempts, current, ascii_only=True)

    if not current.streaming:
        current = _add_attempt(attempts, current, streaming=True)

    return attempts


def _attempt_signature(
    args: argparse.Namespace,
) -> tuple[str, int, Optional[int], Optional[int], bool, bool, int]:
    return (
        args.algorithm,
        args.vocab_size,
        args.max_chars,
        args.max_samples,
        args.ascii_only,
        args.streaming,
        args.wordpiece_max_input_chars,
    )


def _next_vocab_size(current: int) -> int:
    grown = int(current * AUTO_VOCAB_GROWTH_FACTOR)
    return max(current + AUTO_VOCAB_MIN_INCREMENT, grown)


def _args_to_cli(args: argparse.Namespace) -> list[str]:
    cmd = ["--dataset", args.dataset, "--algorithm", args.algorithm, "--split", args.split]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    cmd += ["--vocab_size", str(args.vocab_size)]
    cmd += ["--model_max_length", str(args.model_max_length)]
    cmd += ["--max_piece_length", str(args.max_piece_length)]
    if args.no_progress:
        cmd.append("--no_progress")
    if args.max_chars is not None:
        cmd += ["--max_chars", str(args.max_chars)]
    if args.ascii_only:
        cmd.append("--ascii_only")
    if args.streaming:
        cmd.append("--streaming")
    if args.allow_parallelism:
        cmd.append("--allow_parallelism")
    if args.verify:
        cmd.append("--verify")
    if args.wordpiece_max_input_chars is not None:
        cmd += ["--wordpiece_max_input_chars", str(args.wordpiece_max_input_chars)]
    return cmd


def _describe_attempt(base: argparse.Namespace, attempt: argparse.Namespace) -> str:
    fields = ["vocab_size", "max_chars", "max_samples", "ascii_only", "streaming"]
    changes = []
    for field in fields:
        before = getattr(base, field)
        after = getattr(attempt, field)
        if before != after:
            changes.append(f"{field}={after}")
    return ", ".join(changes) if changes else "no changes"


VOCAB_TOO_SMALL_MSG = (
    "vocab_size is smaller than the number of required symbols. "
    "Increase --vocab_size or re-run with --ascii_only."
)


def _is_vocab_size_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "not large enough to contain all chars" in msg
        or "vocab size too small" in msg
        or "smaller than the number of alphabet" in msg
        or "smaller than the number of unique characters" in msg
        or VOCAB_TOO_SMALL_MSG.lower() in msg
    )


def _build_tokenizer_and_trainer(args: argparse.Namespace):
    from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
    from tokenizers.models import BPE, Unigram, WordPiece
    from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

    algo = resolve_algorithm(args.algorithm)
    if algo == "unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(
            vocab_size=args.vocab_size,
            show_progress=not args.no_progress,
            special_tokens=SPECIAL_TOKENS,
            unk_token="[UNK]",
            max_piece_length=args.max_piece_length,
        )
    elif algo == "bpe":
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            show_progress=not args.no_progress,
            special_tokens=SPECIAL_TOKENS,
        )
    elif algo == "wordpiece":
        tokenizer = Tokenizer(
            WordPiece(
                unk_token="[UNK]",
                max_input_chars_per_word=args.wordpiece_max_input_chars,
            )
        )
        trainer = WordPieceTrainer(
            vocab_size=args.vocab_size,
            show_progress=not args.no_progress,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.decoder = decoders.WordPiece(prefix="##")
    else:
        raise ValueError(f"Unhandled algorithm '{algo}'")

    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer, trainer, algo


def train_once(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from tokenizers import processors
    from transformers import AutoTokenizer

    if not args.allow_parallelism:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = repo_root / "datasets" / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = resolve_dataset(args.dataset)
    spec = DATASET_REGISTRY[dataset_key]

    print(f"[1/3] Loading dataset {spec.path} ({args.split})")
    ds = load_dataset(
        spec.path,
        split=args.split,
        cache_dir=str(cache_dir),
        streaming=args.streaming,
    )
    if args.max_samples is not None and not args.streaming:
        ds = ds.select(range(args.max_samples))

    length = None if args.streaming else len(ds)
    if args.streaming and args.max_samples is not None:
        length = args.max_samples

    def iterator():
        rows = islice(ds, args.max_samples) if args.streaming and args.max_samples else ds
        for row in rows:
            text = spec.formatter(row)
            text = _clean_text(text, ascii_only=args.ascii_only, max_chars=args.max_chars)
            if text:
                yield text

    tokenizer, trainer, algorithm = _build_tokenizer_and_trainer(args)

    outdir = repo_root / "tokenizers" / f"{dataset_key}_{algorithm}_tokenizer"
    outdir.mkdir(parents=True, exist_ok=True)

    algo_label = ALGORITHM_LABELS.get(algorithm, algorithm)
    print(f"[2/3] Training {algo_label} tokenizer -> {outdir}")

    try:
        tokenizer.train_from_iterator(iterator(), trainer=trainer, length=length)
    except Exception as exc:
        if _is_vocab_size_error(exc):
            raise RuntimeError(VOCAB_TOO_SMALL_MSG) from exc
        raise

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    tokenizer.save(str(outdir / "tokenizer.json"))

    with open(outdir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_max_length": args.model_max_length,
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            f,
            indent=2,
        )

    with open(outdir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "unk_token": "[UNK]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            },
            f,
            indent=2,
        )

    print(f"[3/3] Saved tokenizer to {outdir}")

    if args.verify:
        tok = AutoTokenizer.from_pretrained(str(outdir))
        print("Loaded tokenizer type:", type(tok))
        print("backend_tokenizer type:", type(tok.backend_tokenizer))


def run_with_auto_retry(args: argparse.Namespace) -> None:
    attempts = _build_attempts(args)
    queue = list(attempts)
    seen = {_attempt_signature(attempt) for attempt in queue}
    script_path = Path(__file__).resolve()
    last_code = None
    growths = 0
    idx = 0

    while queue:
        attempt = queue.pop(0)
        idx += 1
        total = idx + len(queue)
        desc = _describe_attempt(args, attempt)
        print(f"[auto] Attempt {idx}/{total}: {desc}")
        cmd = [sys.executable, str(script_path), *_args_to_cli(attempt), "--_single_run"]
        proc = subprocess.run(cmd)
        last_code = proc.returncode
        if proc.returncode == 0:
            return
        if proc.returncode == 64:
            if growths < AUTO_VOCAB_MAX_GROWTHS:
                next_vocab = _next_vocab_size(attempt.vocab_size)
                if next_vocab > attempt.vocab_size:
                    grown_attempt = _clone_args(attempt, vocab_size=next_vocab)
                    sig = _attempt_signature(grown_attempt)
                    if sig not in seen:
                        seen.add(sig)
                        queue.insert(0, grown_attempt)
                        growths += 1
                        print(
                            f"[auto] vocab_size too small; will try with --vocab_size {next_vocab}."
                        )
                        continue
            if not attempt.ascii_only:
                print("[auto] vocab_size too small; will try with --ascii_only.")

    if last_code == 64:
        raise SystemExit(VOCAB_TOO_SMALL_MSG)
    raise SystemExit("Training failed after automatic retries.")


def main() -> None:
    args = parse_args()
    args.algorithm = resolve_algorithm(args.algorithm)

    if args.auto_retry and not args._single_run:
        run_with_auto_retry(args)
        return

    try:
        train_once(args)
    except RuntimeError as exc:
        if args._single_run and _is_vocab_size_error(exc):
            print(str(exc), file=sys.stderr)
            raise SystemExit(64) from exc
        raise


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def parse_boundaries(boundaries: str) -> Set[str]:
    # Interpret escapes (\n, \t, \" ...) using JSON string decoding
    decoded = json.loads('"' + boundaries.replace('"', '\\"') + '"')
    return set(decoded)


def iter_txt(path: str, max_samples: Optional[int]) -> Iterable[Tuple[Optional[str], str]]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield None, line
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_tsv(path: str, max_samples: Optional[int], label_key: str = "label") -> Iterable[Tuple[str, str]]:
    # Format: label\ttext
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            y, x = parts[0], parts[1]
            yield y, x
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_jsonl(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    import json

    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            x = obj[text_key]
            y = obj[label_key] if label_key and label_key in obj else None
            yield (str(y) if y is not None else None), str(x)
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_parquet(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    # Prefer pyarrow.dataset for large datasets; fallback to pandas.
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        cols = [text_key] + ([label_key] if label_key else [])
        scanner = dataset.scanner(columns=cols)
        n = 0
        for batch in scanner.to_batches():
            table = batch.to_pydict()
            texts = table[text_key]
            labels = table[label_key] if label_key else [None] * len(texts)
            for y, x in zip(labels, texts):
                if x is None:
                    continue
                yield (str(y) if y is not None else None), str(x)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return
    except Exception:
        import pandas as pd

        df = pd.read_parquet(path, columns=[text_key] + ([label_key] if label_key else []))
        n = 0
        for row in df.itertuples(index=False):
            if label_key:
                y, x = getattr(row, label_key), getattr(row, text_key)
            else:
                x = getattr(row, text_key)
                y = None
            if x is None:
                continue
            yield (str(y) if y is not None else None), str(x)
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def corpus_iter(fmt: str, path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterable[Tuple[Optional[str], str]]:
    if fmt == "txt":
        return iter_txt(path, max_samples)
    if fmt == "tsv":
        return iter_tsv(path, max_samples)
    if fmt == "jsonl":
        return iter_jsonl(path, max_samples, text_key=text_key, label_key=label_key)
    if fmt == "parquet":
        return iter_parquet(path, max_samples, text_key=text_key, label_key=label_key)
    raise ValueError(f"Unknown --format: {fmt}")


def collect_base_chars(samples: Iterable[str], boundaries: Set[str], max_base_chars: int, use_ascii_base: bool) -> Set[str]:
    chars: Set[str] = set()
    if use_ascii_base:
        for i in range(128):
            chars.add(chr(i))
    chars |= boundaries
    for s in samples:
        for ch in s:
            chars.add(ch)
            if len(chars) >= max_base_chars:
                return chars
    return chars


def collect_candidates(
    texts: Iterable[str],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for text in texts:
        s = text[:max_chars_per_sample]
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2:
                    cnt[cur] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        cnt[cur + s[j]] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        cnt[s[i - 1] + cur] += 1
            i += 1
    # filter
    for k in list(cnt.keys()):
        if cnt[k] < min_freq:
            del cnt[k]
    return cnt


def estimate_gain(token: str, freq: int) -> float:
    # With single-char base vocab, a token of length L saves (L-1) per occurrence.
    L = len(token)
    return float(freq) * max(L - 1, 0)


def compute_token_mi(
    token: str,
    label_counts: Dict[str, int],
    token_label_counts: Dict[str, Dict[str, int]],
) -> float:
    # MI approx: sum_y p(y|t) log p(y|t)/p(y)
    t_counts = token_label_counts.get(token)
    if not t_counts:
        return 0.0
    total_t = sum(t_counts.values())
    total = sum(label_counts.values())
    if total_t == 0 or total == 0:
        return 0.0
    mi = 0.0
    for y, c in t_counts.items():
        py_t = c / total_t
        py = label_counts.get(y, 0) / total
        if py_t > 0 and py > 0:
            mi += py_t * math.log(py_t / py)
    return mi


def build_vocab(
    base_chars: Set[str],
    candidates: Counter[str],
    vocab_size: int,
    special_tokens: Sequence[str],
    semantic_mode: str,
    lambda_sem: float,
    label_counts: Dict[str, int],
    token_label_counts: Dict[str, Dict[str, int]],
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    cur_id = 0

    # specials first
    for st in special_tokens:
        token_to_id[st] = cur_id
        cur_id += 1

    # base chars
    for ch in sorted(base_chars):
        if ch in token_to_id:
            continue
        token_to_id[ch] = cur_id
        cur_id += 1

    if len(token_to_id) >= vocab_size:
        # If base already exceeds budget, truncate deterministically.
        items = list(token_to_id.items())[:vocab_size]
        return {k: i for i, (k, _) in enumerate(items)}

    scored: List[Tuple[float, str]] = []
    for tok, f in candidates.items():
        if tok in token_to_id:
            continue
        gain = estimate_gain(tok, f)
        if gain <= 0:
            continue
        score = gain
        if semantic_mode == "mi" and lambda_sem > 0 and label_counts:
            score += lambda_sem * compute_token_mi(tok, label_counts, token_label_counts)
        scored.append((score, tok))

    scored.sort(key=lambda x: (-x[0], x[1]))  # deterministic

    for score, tok in scored:
        if len(token_to_id) >= vocab_size:
            break
        token_to_id[tok] = cur_id
        cur_id += 1

    return token_to_id


def build_token_label_counts(
    labeled_samples: List[Tuple[str, str]],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    semantic_top_k: int,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    # Build candidate list first, then count token occurrences by label for top_k.
    texts = [x for _, x in labeled_samples]
    cands = collect_candidates(texts, boundaries, max_len, min_freq, allow_boundary_at_ends, max_chars_per_sample)
    top = set([tok for tok, _ in cands.most_common(semantic_top_k)])

    label_counts: Dict[str, int] = Counter([y for y, _ in labeled_samples])
    token_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for y, x in labeled_samples:
        # simple presence counting in each sample: count all occurrences (not just binary) for cheap signal
        s = x[:max_chars_per_sample]
        n = len(s)
        i = 0
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2 and cur in top:
                    token_label_counts[cur][y] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if t in top:
                            token_label_counts[t][y] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if t in top:
                            token_label_counts[t][y] += 1
            i += 1

    return dict(label_counts), {k: dict(v) for k, v in token_label_counts.items()}


def write_fast_tokenizer_json(outdir: str, token_to_id: Dict[str, int], special_tokens: Sequence[str]) -> None:
    # Build a tokenizers backend tokenizer.json using WordPiece greedy longest-match.
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.processors import TemplateProcessing

    unk = "[UNK]"
    if unk not in token_to_id:
        raise ValueError("[UNK] must be in vocab")

    model = WordPiece(vocab=token_to_id, unk_token=unk, continuing_subword_prefix="")
    tok = Tokenizer(model)

    # Add post-processor for CLS/SEP
    cls = "[CLS]"
    sep = "[SEP]"
    if cls in token_to_id and sep in token_to_id:
        tok.post_processor = TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[(cls, token_to_id[cls]), (sep, token_to_id[sep])],
        )

    tok.save(os.path.join(outdir, "tokenizer.json"))


def write_artifact(
    outdir: str,
    token_to_id: Dict[str, int],
    boundaries: Set[str],
    vocab_size: int,
    max_len: int,
    min_freq: int,
    fmt: str,
    text_key: str,
    label_key: Optional[str],
    semantic_mode: str,
    lambda_sem: float,
    semantic_top_k: int,
    model_max_length: int,
    emit_code: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # vocab.json for debugging / slow tokenizer
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=True, indent=2)

    meta = {
        "match_special_tokens": False,
        "artifact_version": "ctok-fast-v1",
        "build": {
            "format": fmt,
            "text_key": text_key,
            "label_key": label_key,
            "vocab_size_requested": vocab_size,
            "vocab_size_actual": len(token_to_id),
            "max_len": max_len,
            "min_freq": min_freq,
            "boundaries": sorted(list(boundaries)),
            "semantic_mode": semantic_mode,
            "lambda_sem": lambda_sem,
            "semantic_top_k": semantic_top_k,
        },
    }
    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

    # tokenizer_config.json: IMPORTANT auto_map tuple = [slow, fast]
    tok_cfg = {
        "tokenizer_class": "CTokTokenizerFast",
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_ctok.CTokTokenizer",
                "tokenization_ctok_fast.CTokTokenizerFast",
            ]
        },
        "model_max_length": model_max_length,
        "padding_side": "right",
        "truncation_side": "right",
    }
    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=True, indent=2)

    sp_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(outdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(sp_map, f, ensure_ascii=True, indent=2)

    # Fast tokenizer backend
    write_fast_tokenizer_json(outdir, token_to_id, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    if emit_code:
        here = Path(__file__).resolve().parent
        for fn in ["tokenization_ctok.py", "tokenization_ctok_fast.py"]:
            shutil.copy(str(here / fn), os.path.join(outdir, fn))

    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# CTok Fast Tokenizer Artifact\n\n"
            "This directory is loadable via `AutoTokenizer.from_pretrained(path, trust_remote_code=True)`.\n\n"
            "Files:\n"
            "- tokenizer.json: Rust backend (WordPiece greedy longest-match with empty continuation prefix)\n"
            "- vocab.json: token->id (debug / slow tokenizer)\n"
            "- ctok_meta.json: build metadata\n"
            "- tokenizer_config.json, special_tokens_map.json: Transformers integration\n"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus (txt/tsv/jsonl/parquet) or parquet directory")
    ap.add_argument("--format", default="parquet", choices=["txt", "tsv", "jsonl", "parquet"])
    ap.add_argument("--text_key", default="text", help="For jsonl/parquet: text field")
    ap.add_argument("--label_key", default="label", help="For jsonl/parquet: label field; set to empty to disable")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_freq", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=200000)
    ap.add_argument("--max_chars_per_sample", type=int, default=4096)
    ap.add_argument("--boundaries", type=str, default="=&?:/\\n\\t <>\\\"'", help="Boundary characters (supports escapes)")
    ap.add_argument("--no_boundary_ends", action="store_true")

    ap.add_argument("--use_ascii_base", action="store_true", help="Include ASCII chars (0..127) in base vocab")
    ap.add_argument("--max_base_chars", type=int, default=4096)

    ap.add_argument("--semantic_mode", choices=["none", "mi"], default="none")
    ap.add_argument("--lambda_sem", type=float, default=0.0)
    ap.add_argument("--semantic_top_k", type=int, default=50000)

    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--emit_code", action="store_true")

    args = ap.parse_args()

    boundaries = parse_boundaries(args.boundaries)
    label_key = args.label_key if args.label_key else None

    it = corpus_iter(args.format, args.corpus, args.max_samples, args.text_key, label_key)

    # Materialize a subset for base chars + optional MI stats (keeps code simple).
    samples: List[Tuple[Optional[str], str]] = []
    for y, x in it:
        samples.append((y, x))
    texts = [x for _, x in samples]

    base_chars = collect_base_chars(texts, boundaries, max_base_chars=args.max_base_chars, use_ascii_base=args.use_ascii_base)

    allow_boundary_at_ends = not args.no_boundary_ends

    # candidates (frequency)
    cands = collect_candidates(
        texts=texts,
        boundaries=boundaries,
        max_len=args.max_len,
        min_freq=args.min_freq,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=args.max_chars_per_sample,
    )

    label_counts: Dict[str, int] = {}
    token_label_counts: Dict[str, Dict[str, int]] = {}

    if args.semantic_mode == "mi" and label_key is not None:
        labeled = [(y, x) for y, x in samples if y is not None]
        if labeled:
            label_counts, token_label_counts = build_token_label_counts(
                labeled_samples=[(str(y), x) for y, x in labeled],
                boundaries=boundaries,
                max_len=args.max_len,
                min_freq=args.min_freq,
                allow_boundary_at_ends=allow_boundary_at_ends,
                max_chars_per_sample=args.max_chars_per_sample,
                semantic_top_k=args.semantic_top_k,
            )

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    vocab = build_vocab(
        base_chars=base_chars,
        candidates=cands,
        vocab_size=args.vocab_size,
        special_tokens=special,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        label_counts=label_counts,
        token_label_counts=token_label_counts,
    )

    write_artifact(
        outdir=args.outdir,
        token_to_id=vocab,
        boundaries=boundaries,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        fmt=args.format,
        text_key=args.text_key,
        label_key=label_key,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        semantic_top_k=args.semantic_top_k,
        model_max_length=args.model_max_length,
        emit_code=args.emit_code,
    )

    print(f"Wrote CTok FAST artifact to: {args.outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(cands)}")


if __name__ == "__main__":
    main()

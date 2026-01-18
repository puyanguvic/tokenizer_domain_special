from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import hygiene
from . import pretokenize

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None

_HYGIENE_CFG: Optional[hygiene.HygieneConfig] = None
_PRETOK_CFG: Optional[pretokenize.PreTokenizerConfig] = None


def _init_hygiene_worker(cfg_dict: Dict[str, object]) -> None:
    global _HYGIENE_CFG
    _HYGIENE_CFG = hygiene.HygieneConfig.from_dict(cfg_dict)


def _apply_hygiene_sample(sample: Tuple[Optional[str], str]) -> Tuple[Optional[str], str]:
    y, x = sample
    if _HYGIENE_CFG is None:
        return y, x
    return y, hygiene.apply_hygiene(x, _HYGIENE_CFG)


def _init_pretok_worker(cfg_dict: Dict[str, object]) -> None:
    global _PRETOK_CFG
    _PRETOK_CFG = pretokenize.PreTokenizerConfig.from_dict(cfg_dict)


def _apply_pretok_sample(sample: Tuple[Optional[str], str]) -> Tuple[Optional[str], str]:
    y, x = sample
    if _PRETOK_CFG is None:
        return y, x
    return y, pretokenize.apply_pretokenize(x, _PRETOK_CFG)


def _collect_base_chars_chunk(args: Tuple[List[object], Set[str], int]) -> Set[str]:
    items, boundaries, max_base_chars = args
    chars: Set[str] = set(boundaries)
    for item in items:
        s = _get_text(item)
        for ch in s:
            chars.add(ch)
            if len(chars) >= max_base_chars:
                return chars
    return chars


def _progress(iterable, total=None, desc: str = "", unit: str = "it"):
    if _tqdm is None:
        return iterable
    return _tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        file=sys.stdout,
        dynamic_ncols=True,
    )


def parse_boundaries(boundaries: str) -> Set[str]:
    # Interpret escapes (\n, \t, \" ...) using unicode_escape decoding.
    decoded = boundaries.encode("utf-8").decode("unicode_escape")
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


def _get_text(item: object) -> str:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[1])
    return str(item)


def collect_base_chars(
    samples: Iterable[object],
    boundaries: Set[str],
    max_base_chars: int,
    use_ascii_base: bool,
    extra_tokens: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 1,
) -> Set[str]:
    chars: Set[str] = set()
    if use_ascii_base:
        chars |= hygiene.ascii_base_chars()
    chars |= boundaries
    if extra_tokens:
        chars.update(extra_tokens)
    sample_list = samples if isinstance(samples, list) else list(samples)
    total = len(sample_list)
    if max_samples is not None:
        sample_list = sample_list[:max_samples]
        total = len(sample_list)

    if num_workers <= 1 or total == 0:
        iterator = _progress(sample_list, total=total, desc="Collecting base chars", unit="samples")
        for item in iterator:
            s = _get_text(item)
            for ch in s:
                chars.add(ch)
                if len(chars) >= max_base_chars:
                    return chars
        return chars

    print(f"Collecting base chars with {num_workers} workers")
    chunk_size = max(1, total // (num_workers * 4))
    chunks = [sample_list[i : i + chunk_size] for i in range(0, total, chunk_size)]
    args_list = [(c, boundaries, max_base_chars) for c in chunks]
    with mp.Pool(processes=num_workers) as pool:
        results_iter = pool.imap_unordered(_collect_base_chars_chunk, args_list, chunksize=1)
        if _tqdm is not None:
            results_iter = _tqdm(
                results_iter,
                total=len(args_list),
                desc="Collecting base chars (mp)",
                unit="chunks",
                file=sys.stdout,
                dynamic_ncols=True,
            )
        for cset in results_iter:
            chars.update(cset)
            if len(chars) >= max_base_chars:
                break
    return chars


def _collect_candidates_chunk(args: Tuple[List[object], Set[str], int, int, bool, int]) -> Counter[str]:
    items, boundaries, max_len, min_freq, allow_boundary_at_ends, max_chars_per_sample = args
    cnt: Counter[str] = Counter()
    for item in items:
        text = _get_text(item)
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
    return cnt


def _count_token_label_chunk(
    args: Tuple[List[Tuple[Optional[str], str]], Set[str], int, bool, int, Set[str]]
) -> Tuple[Counter[str], Dict[str, Dict[str, int]]]:
    chunk, boundaries, max_len, allow_boundary_at_ends, max_chars_per_sample, top = args
    local_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    local_labels: Counter[str] = Counter()
    for y, x in chunk:
        if y is None:
            continue
        local_labels[y] += 1
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
                    local_counts[cur][y] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if t in top:
                            local_counts[t][y] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if t in top:
                            local_counts[t][y] += 1
            i += 1
    return local_labels, {k: dict(v) for k, v in local_counts.items()}


def _collect_doc_stats_chunk(
    args: Tuple[List[object], Set[str], int, bool, int, Set[str]]
) -> Tuple[Counter[str], Dict[str, int]]:
    items, candidates, max_len, allow_boundary_at_ends, max_chars_per_sample, boundaries = args
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    for item in items:
        text = _get_text(item)
        s = text[:max_chars_per_sample]
        n = len(s)
        i = 0
        local: Counter[str] = Counter()
        while i < n:
            if s[i] in boundaries:
                i += 1
                continue
            j = i
            while j < n and (j - i) < max_len and s[j] not in boundaries:
                j += 1
                cur = s[i:j]
                if len(cur) >= 2 and cur in candidates:
                    local[cur] += 1
                if allow_boundary_at_ends:
                    if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                        t = cur + s[j]
                        if t in candidates:
                            local[t] += 1
                    if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                        t = s[i - 1] + cur
                        if t in candidates:
                            local[t] += 1
            i += 1
        for tok in local:
            doc_freq[tok] += 1
            prev = max_in_doc.get(tok, 0)
            if local[tok] > prev:
                max_in_doc[tok] = local[tok]
    return doc_freq, max_in_doc


def collect_candidates(
    texts: Iterable[object],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    num_workers: int = 1,
) -> Counter[str]:
    cnt: Counter[str] = Counter()
    text_list = texts if isinstance(texts, list) else list(texts)
    total = len(text_list)
    if num_workers <= 1 or total == 0:
        for item in _progress(text_list, total=total, desc="Collecting candidates", unit="samples"):
            text = _get_text(item)
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
    else:
        print(f"Collecting candidates with {num_workers} workers")
        chunk_size = max(1, total // (num_workers * 4))
        chunks = [text_list[i : i + chunk_size] for i in range(0, total, chunk_size)]
        args_list = [(c, boundaries, max_len, min_freq, allow_boundary_at_ends, max_chars_per_sample) for c in chunks]
        with mp.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_collect_candidates_chunk, args_list, chunksize=1)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Collecting candidates (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            for c in results_iter:
                cnt.update(c)
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
    required_tokens: Sequence[str],
    semantic_mode: str,
    lambda_sem: float,
    label_counts: Dict[str, int],
    token_label_counts: Dict[str, Dict[str, int]],
    junk_penalty_beta: float = 0.0,
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    cur_id = 0

    # specials first
    for st in special_tokens:
        token_to_id[st] = cur_id
        cur_id += 1

    # required tokens (e.g., typed tokens)
    for tok in sorted(set(required_tokens)):
        if tok in token_to_id:
            continue
        token_to_id[tok] = cur_id
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
        if junk_penalty_beta > 0:
            score -= junk_penalty_beta * hygiene.junk_score(tok)
        scored.append((score, tok))

    scored.sort(key=lambda x: (-x[0], x[1]))  # deterministic

    for score, tok in scored:
        if len(token_to_id) >= vocab_size:
            break
        token_to_id[tok] = cur_id
        cur_id += 1

    return token_to_id


def collect_doc_stats(
    texts: Iterable[object],
    candidates: Set[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    num_workers: int = 1,
) -> Tuple[Counter[str], Dict[str, int]]:
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    total = len(texts) if hasattr(texts, "__len__") else None
    if num_workers <= 1 or total == 0:
        for item in _progress(texts, total=total, desc="Doc stats", unit="samples"):
            text = _get_text(item)
            s = text[:max_chars_per_sample]
            n = len(s)
            i = 0
            local: Counter[str] = Counter()
            while i < n:
                if s[i] in boundaries:
                    i += 1
                    continue
                j = i
                while j < n and (j - i) < max_len and s[j] not in boundaries:
                    j += 1
                    cur = s[i:j]
                    if len(cur) >= 2 and cur in candidates:
                        local[cur] += 1
                    if allow_boundary_at_ends:
                        if j < n and (len(cur) + 1) <= max_len and s[j] in boundaries:
                            t = cur + s[j]
                            if t in candidates:
                                local[t] += 1
                        if i > 0 and (len(cur) + 1) <= max_len and s[i - 1] in boundaries:
                            t = s[i - 1] + cur
                            if t in candidates:
                                local[t] += 1
                i += 1
            for tok in local:
                doc_freq[tok] += 1
                prev = max_in_doc.get(tok, 0)
                if local[tok] > prev:
                    max_in_doc[tok] = local[tok]
    else:
        print(f"Doc stats with {num_workers} workers")
        text_list = texts if isinstance(texts, list) else list(texts)
        chunk_size = max(1, len(text_list) // (num_workers * 4))
        chunks = [text_list[i : i + chunk_size] for i in range(0, len(text_list), chunk_size)]
        args_list = [(c, candidates, max_len, allow_boundary_at_ends, max_chars_per_sample, boundaries) for c in chunks]
        with mp.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_collect_doc_stats_chunk, args_list, chunksize=1)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Doc stats (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            for local_doc_freq, local_max_in_doc in results_iter:
                doc_freq.update(local_doc_freq)
                for tok, cnt in local_max_in_doc.items():
                    prev = max_in_doc.get(tok, 0)
                    if cnt > prev:
                        max_in_doc[tok] = cnt
    return doc_freq, max_in_doc


def filter_candidates(
    candidates: Counter[str],
    texts: Iterable[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    filter_value_fragments: bool,
    typed_tokens: Sequence[str],
    min_doc_freq: int,
    max_doc_concentration: float,
    num_workers: int = 1,
) -> Counter[str]:
    filtered = Counter()
    for tok, cnt in candidates.items():
        if hygiene.is_typed_token_fragment(tok, typed_tokens):
            continue
        if filter_value_fragments and hygiene.is_value_fragment(tok):
            continue
        filtered[tok] = cnt

    if min_doc_freq <= 1 and max_doc_concentration >= 1.0:
        return filtered

    doc_freq, max_in_doc = collect_doc_stats(
        texts,
        candidates=set(filtered.keys()),
        boundaries=boundaries,
        max_len=max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=max_chars_per_sample,
        num_workers=num_workers,
    )
    out = Counter()
    for tok, cnt in filtered.items():
        if min_doc_freq > 1 and doc_freq.get(tok, 0) < min_doc_freq:
            continue
        if max_doc_concentration < 1.0:
            ratio = max_in_doc.get(tok, 0) / max(cnt, 1)
            if ratio > max_doc_concentration:
                continue
        out[tok] = cnt
    return out


def build_token_label_counts(
    labeled_samples: List[Tuple[Optional[str], str]],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    semantic_top_k: int,
    candidates: Optional[Counter[str]] = None,
    num_workers: int = 1,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    # Build candidate list first, then count token occurrences by label for top_k.
    texts = [x for y, x in labeled_samples if y is not None]
    if candidates is None:
        candidates = collect_candidates(
            texts=texts,
            boundaries=boundaries,
            max_len=max_len,
            min_freq=min_freq,
            allow_boundary_at_ends=allow_boundary_at_ends,
            max_chars_per_sample=max_chars_per_sample,
            num_workers=num_workers,
        )
    top = set([tok for tok, _ in candidates.most_common(semantic_top_k)])

    label_counts: Dict[str, int] = Counter([y for y, _ in labeled_samples if y is not None])
    token_label_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = len(labeled_samples) if hasattr(labeled_samples, "__len__") else None
    if num_workers <= 1 or total == 0:
        for y, x in _progress(labeled_samples, total=total, desc="Counting token-labels", unit="samples"):
            if y is None:
                continue
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
    else:
        print(f"Counting token-labels with {num_workers} workers")
        chunk_size = max(1, total // (num_workers * 4))
        chunks = [labeled_samples[i : i + chunk_size] for i in range(0, total, chunk_size)]
        args_list = [(c, boundaries, max_len, allow_boundary_at_ends, max_chars_per_sample, top) for c in chunks]
        with mp.Pool(processes=num_workers) as pool:
            results_iter = pool.imap_unordered(_count_token_label_chunk, args_list, chunksize=1)
            if _tqdm is not None:
                results_iter = _tqdm(
                    results_iter,
                    total=len(args_list),
                    desc="Counting token-labels (mp)",
                    unit="chunks",
                    file=sys.stdout,
                    dynamic_ncols=True,
                )
            label_counts = Counter()
            token_label_counts = defaultdict(lambda: defaultdict(int))
            for local_labels, local_counts in results_iter:
                label_counts.update(local_labels)
                for tok, lab_map in local_counts.items():
                    for y, c in lab_map.items():
                        token_label_counts[tok][y] += c

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
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    hygiene_metrics: Dict[str, float],
    hygiene_build: Dict[str, object],
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # vocab.json for debugging / slow tokenizer
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=True, indent=2)

    meta = {
        "match_special_tokens": False,
        "artifact_version": "ctok-fast-v1",
        "pipeline_locked": True,
        "lowercase": bool(hygiene_build.get("lowercase", False)),
        "hygiene": hygiene_cfg.to_dict(),
        "pretokenizer": pretok_cfg.to_dict(),
        "hygiene_metrics": hygiene_metrics,
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
            "pretokenizer": "generic" if pretok_cfg.enabled else "none",
            "lowercase": bool(hygiene_build.get("lowercase", False)),
            "base_chars_max_samples": args.base_chars_max_samples,
            **hygiene_build,
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
        for fn in ["tokenization_ctok.py", "tokenization_ctok_fast.py", "hygiene.py", "pretokenize.py"]:
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


def build_ctok_from_samples(
    samples: List[Tuple[Optional[str], str]],
    text_key: str,
    label_key: Optional[str],
    outdir: str,
    args: argparse.Namespace,
) -> None:
    boundaries = parse_boundaries(args.boundaries)
    num_workers = args.num_workers
    if num_workers <= 0:
        num_workers = max(1, mp.cpu_count() - 1)

    hygiene_cfg = hygiene.default_hygiene_config()
    hygiene_cfg.enabled = not args.no_hygiene
    if not hygiene_cfg.enabled:
        hygiene_cfg.typed_tokens = []
        hygiene_cfg.patterns = []

    pretok_cfg = pretokenize.default_pretokenizer_config()
    pretok_cfg.enabled = args.pretokenizer != "none"
    if not pretok_cfg.enabled:
        pretok_cfg.patterns = []

    if args.lowercase:
        samples = [(y, x.lower()) for y, x in samples]
    if hygiene_cfg.enabled:
        if num_workers > 1 and len(samples) > 0:
            print(f"Applying hygiene with {num_workers} workers")
            cfg_dict = hygiene_cfg.to_dict()
            with mp.Pool(processes=num_workers, initializer=_init_hygiene_worker, initargs=(cfg_dict,)) as pool:
                results_iter = pool.imap(_apply_hygiene_sample, samples, chunksize=256)
                if _tqdm is not None:
                    results_iter = _tqdm(
                        results_iter,
                        total=len(samples),
                        desc="Applying hygiene",
                        unit="samples",
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                samples = list(results_iter)
        else:
            samples = [(y, hygiene.apply_hygiene(x, hygiene_cfg)) for y, x in samples]
    if pretok_cfg.enabled:
        if num_workers > 1 and len(samples) > 0:
            print(f"Applying pretokenizer with {num_workers} workers")
            cfg_dict = pretok_cfg.to_dict()
            with mp.Pool(processes=num_workers, initializer=_init_pretok_worker, initargs=(cfg_dict,)) as pool:
                results_iter = pool.imap(_apply_pretok_sample, samples, chunksize=256)
                if _tqdm is not None:
                    results_iter = _tqdm(
                        results_iter,
                        total=len(samples),
                        desc="Applying pretokenizer",
                        unit="samples",
                        file=sys.stdout,
                        dynamic_ncols=True,
                    )
                samples = list(results_iter)
        else:
            samples = [(y, pretokenize.apply_pretokenize(x, pretok_cfg)) for y, x in samples]
    texts = samples

    base_chars = collect_base_chars(
        texts,
        boundaries,
        max_base_chars=args.max_base_chars,
        use_ascii_base=args.use_ascii_base,
        extra_tokens=hygiene_cfg.typed_tokens,
        max_samples=args.base_chars_max_samples,
        num_workers=num_workers,
    )

    allow_boundary_at_ends = not args.no_boundary_ends

    cands = collect_candidates(
        texts=texts,
        boundaries=boundaries,
        max_len=args.max_len,
        min_freq=args.min_freq,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=args.max_chars_per_sample,
        num_workers=num_workers,
    )
    cands_raw = cands
    cands = filter_candidates(
        candidates=cands,
        texts=texts,
        boundaries=boundaries,
        max_len=args.max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=args.max_chars_per_sample,
        filter_value_fragments=not args.no_filter_value_fragments,
        typed_tokens=hygiene_cfg.typed_tokens,
        min_doc_freq=args.min_doc_freq,
        max_doc_concentration=args.max_doc_concentration,
        num_workers=num_workers,
    )

    label_counts: Dict[str, int] = {}
    token_label_counts: Dict[str, Dict[str, int]] = {}

    if args.semantic_mode == "mi" and label_key is not None:
        label_counts, token_label_counts = build_token_label_counts(
            labeled_samples=samples,
                boundaries=boundaries,
                max_len=args.max_len,
                min_freq=args.min_freq,
                allow_boundary_at_ends=allow_boundary_at_ends,
                max_chars_per_sample=args.max_chars_per_sample,
                semantic_top_k=args.semantic_top_k,
                candidates=cands_raw,
                num_workers=num_workers,
        )
        if cands:
            token_label_counts = {k: v for k, v in token_label_counts.items() if k in cands}

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    vocab = build_vocab(
        base_chars=base_chars,
        candidates=cands,
        vocab_size=args.vocab_size,
        special_tokens=special,
        required_tokens=hygiene_cfg.typed_tokens,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        label_counts=label_counts,
        token_label_counts=token_label_counts,
        junk_penalty_beta=args.junk_penalty_beta,
    )

    write_artifact(
        outdir=outdir,
        token_to_id=vocab,
        boundaries=boundaries,
        vocab_size=args.vocab_size,
        max_len=args.max_len,
        min_freq=args.min_freq,
        fmt=args.format,
        text_key=text_key,
        label_key=label_key,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        semantic_top_k=args.semantic_top_k,
        model_max_length=args.model_max_length,
        emit_code=args.emit_code,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        hygiene_metrics=hygiene.vocab_hygiene_metrics(vocab.keys(), hygiene_cfg.typed_tokens),
        hygiene_build={
            "hygiene_enabled": hygiene_cfg.enabled,
            "filter_value_fragments": not args.no_filter_value_fragments,
            "min_doc_freq": args.min_doc_freq,
            "max_doc_concentration": args.max_doc_concentration,
            "junk_penalty_beta": args.junk_penalty_beta,
            "lowercase": bool(args.lowercase),
        },
    )

    print(f"Wrote CTok FAST artifact to: {outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(cands)}")


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
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_chars_per_sample", type=int, default=4096)
    ap.add_argument("--boundaries", type=str, default="=&?:/\\n\\t <>\\\"'", help="Boundary characters (supports escapes)")
    ap.add_argument("--no_boundary_ends", action="store_true")

    ap.add_argument("--use_ascii_base", action="store_true", help="Include ASCII chars (0..127) in base vocab")
    ap.add_argument("--max_base_chars", type=int, default=4096)
    ap.add_argument("--base_chars_max_samples", type=int, default=200000)
    ap.add_argument("--num_workers", type=int, default=0, help="Parallel workers (0=auto)")

    ap.add_argument("--semantic_mode", choices=["none", "mi"], default="none")
    ap.add_argument("--lambda_sem", type=float, default=0.0)
    ap.add_argument("--semantic_top_k", type=int, default=50000)
    ap.add_argument("--no_hygiene", action="store_true", help="Disable hygiene replacements")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before hygiene/pretokenization")
    ap.add_argument("--pretokenizer", choices=["none", "generic"], default="none")
    ap.add_argument("--no_filter_value_fragments", action="store_true", help="Disable value-fragment candidate filtering")
    ap.add_argument("--min_doc_freq", type=int, default=1)
    ap.add_argument("--max_doc_concentration", type=float, default=1.0)
    ap.add_argument("--junk_penalty_beta", type=float, default=0.5)

    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--emit_code", action="store_true")

    args = ap.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    label_key = args.label_key if args.label_key else None

    it = corpus_iter(args.format, args.corpus, args.max_samples, args.text_key, label_key)

    samples: List[Tuple[Optional[str], str]] = []
    for y, x in _progress(it, desc="Reading corpus", unit="samples"):
        samples.append((y, x))

    build_ctok_from_samples(
        samples=samples,
        text_key=args.text_key,
        label_key=label_key,
        outdir=args.outdir,
        args=args,
    )


if __name__ == "__main__":
    main()

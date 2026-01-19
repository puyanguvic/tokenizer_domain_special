from __future__ import annotations

"""Build a CTok tokenizer artifact from a corpus.

User-facing CLI is intentionally minimal. Performance knobs are internal.

This trainer uses a Unigram-style vocabulary induction with iterative pruning
(hard-EM via Viterbi tokenization on a sampled subset), under CTok domain
constraints:
  - Apply hygiene + optional pretokenization.
  - Treat whitespace and boundary characters as hard boundaries.
  - Keep base characters, boundary chars, and typed tokens as required tokens.

Output: an artifact directory loadable via
  AutoTokenizer.from_pretrained(outdir, trust_remote_code=True)

Notes:
  - The runtime tokenizer (fast) is WordPiece greedy longest-match; here we
    *train* the vocabulary using a Unigram objective, then export as a greedy
    matcher for predictable linear-time execution.
"""

import argparse
import gzip
import json
import math
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Imports (package or script)
# -----------------------------
try:
    from . import hygiene  # type: ignore
    from . import pretokenize  # type: ignore
except Exception:  # pragma: no cover
    import hygiene  # type: ignore
    import pretokenize  # type: ignore


try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


def _progress(it, total=None, desc: str = "", unit: str = "it"):
    if _tqdm is None:
        return it
    return _tqdm(it, total=total, desc=desc, unit=unit, file=sys.stdout, dynamic_ncols=True)


def parse_boundaries(boundaries: str) -> Set[str]:
    decoded = boundaries.encode("utf-8").decode("unicode_escape")
    return set(decoded)


# -----------------------------
# Corpus readers
# -----------------------------
def iter_txt(path: str, max_samples: Optional[int]) -> Iterator[Tuple[Optional[str], str]]:
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


def iter_tsv(path: str, max_samples: Optional[int]) -> Iterator[Tuple[str, str]]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_jsonl(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterator[Tuple[Optional[str], str]]:
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


def iter_parquet(path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterator[Tuple[Optional[str], str]]:
    try:
        import pyarrow.dataset as ds  # type: ignore

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
        import pandas as pd  # type: ignore

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


def corpus_iter(fmt: str, path: str, max_samples: Optional[int], text_key: str, label_key: Optional[str]) -> Iterator[Tuple[Optional[str], str]]:
    if fmt == "txt":
        return iter_txt(path, max_samples)
    if fmt == "tsv":
        return iter_tsv(path, max_samples)
    if fmt == "jsonl":
        return iter_jsonl(path, max_samples, text_key=text_key, label_key=label_key)
    if fmt == "parquet":
        return iter_parquet(path, max_samples, text_key=text_key, label_key=label_key)
    raise ValueError(f"Unknown --format: {fmt}")


# -----------------------------
# Internal locked config
# -----------------------------
@dataclass(frozen=True)
class _Locked:
    # Cache (always on): multiple passes reuse preprocessed output.
    cache_name: str = "_ctok_preprocessed.jsonl.gz"

    # Sampling for Unigram hard-EM (Viterbi)
    sample_words: int = 250_000
    max_word_len: int = 256

    # Candidate pool sizes
    top_words_k: int = 250_000
    top_subwords_k_mult: int = 50  # subword_k ~= vocab_size * mult
    top_subwords_k_cap: int = 1_000_000
    top_words_for_subwords: int = 120_000

    # Training schedule
    prune_frac: float = 0.20
    max_prune_iters: int = 6
    smoothing: float = 1e-4
    junk_penalty_beta: float = 0.4

    # Determinism
    rng_seed: int = 0


# -----------------------------
# SpaceSaving heavy-hitter (approx top-k)
# -----------------------------
class SpaceSaving:
    """SpaceSaving heavy-hitter with optional weighted updates.

    Maintains at most k keys with approximate counts.
    """

    def __init__(self, k: int):
        self.k = max(int(k), 1)
        self.counts: Dict[str, int] = {}

    def update(self, key: str, inc: int = 1) -> None:
        if inc <= 0:
            return
        if key in self.counts:
            self.counts[key] += inc
            return
        if len(self.counts) < self.k:
            self.counts[key] = inc
            return
        # Replace current minimum.
        min_key, min_val = min(self.counts.items(), key=lambda kv: kv[1])
        del self.counts[min_key]
        self.counts[key] = min_val + inc

    def items(self) -> List[Tuple[str, int]]:
        return list(self.counts.items())

    def topk(self, k: Optional[int] = None) -> List[Tuple[str, int]]:
        kk = self.k if k is None else max(int(k), 1)
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:kk]


# -----------------------------
# Trie for token matching
# -----------------------------
class _TrieNode:
    __slots__ = ("kids", "term")

    def __init__(self):
        self.kids: Dict[str, "_TrieNode"] = {}
        self.term: List[int] = []


class TokenTrie:
    def __init__(self):
        self.root = _TrieNode()

    def add(self, tok: str, tid: int) -> None:
        node = self.root
        for ch in tok:
            node = node.kids.setdefault(ch, _TrieNode())
        node.term.append(tid)

    def iter_matches(self, s: str, i: int, max_len: int) -> List[Tuple[int, int]]:
        """Return (tid, length) pairs starting at s[i]."""
        out: List[Tuple[int, int]] = []
        node = self.root
        n = len(s)
        j = i
        while j < n and (j - i) < max_len:
            ch = s[j]
            node = node.kids.get(ch)
            if node is None:
                break
            j += 1
            if node.term:
                l = j - i
                for tid in node.term:
                    out.append((tid, l))
        return out


# -----------------------------
# Preprocess + cache
# -----------------------------
def _apply_pipeline(x: str, lowercase: bool, hygiene_cfg: hygiene.HygieneConfig, pretok_cfg: pretokenize.PreTokenizerConfig) -> str:
    if lowercase:
        x = x.lower()
    if hygiene_cfg.enabled:
        x = hygiene.apply_hygiene(x, hygiene_cfg)
    if pretok_cfg.enabled:
        x = pretokenize.apply_pretokenize(x, pretok_cfg)
    return x


def _build_or_load_cache(
    *,
    outdir: str,
    fmt: str,
    corpus_path: str,
    max_samples: Optional[int],
    text_key: str,
    label_key: Optional[str],
    lowercase: bool,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    locked: _Locked,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, locked.cache_name)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    it = corpus_iter(fmt, corpus_path, max_samples, text_key, label_key)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        for y, x in _progress(it, desc="Preprocessing corpus", unit="samples"):
            px = _apply_pipeline(x, lowercase, hygiene_cfg, pretok_cfg)
            rec = {"y": y, "x": px}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return cache_path


def _build_or_load_cache_from_samples(
    *,
    outdir: str,
    samples: Sequence[Tuple[Optional[str], str]],
    max_samples: Optional[int],
    lowercase: bool,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    locked: _Locked,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, locked.cache_name)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        return cache_path

    limit = len(samples) if max_samples is None else min(len(samples), max_samples)
    with gzip.open(cache_path, "wt", encoding="utf-8") as f:
        iterator = samples[:limit]
        for y, x in _progress(iterator, total=limit, desc="Preprocessing samples", unit="samples"):
            px = _apply_pipeline(x, lowercase, hygiene_cfg, pretok_cfg)
            rec = {"y": y, "x": px}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return cache_path


def _iter_cache(cache_path: str) -> Iterator[Tuple[Optional[str], str]]:
    with gzip.open(cache_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            y = obj.get("y")
            x = obj.get("x")
            if x is None:
                continue
            yield (None if y is None else str(y)), str(x)


# -----------------------------
# Candidate generation (words + derived subwords)
# -----------------------------
def _is_valid_tok(tok: str, boundaries: Set[str], typed: Set[str], max_len: int) -> bool:
    if not tok:
        return False
    if tok in typed:
        return False
    if len(tok) < 2 or len(tok) > max_len:
        return False
    if any(ch in boundaries for ch in tok):
        return False
    # avoid obvious value fragments
    if hygiene.is_value_fragment(tok):
        return False
    return True


def _collect_top_words_and_samples(
    cache_path: str,
    *,
    boundaries: Set[str],
    typed_tokens: Sequence[str],
    max_len: int,
    locked: _Locked,
) -> Tuple[List[Tuple[str, int]], List[str], Set[str]]:
    typed = set(typed_tokens)
    hh = SpaceSaving(locked.top_words_k)
    sample: List[str] = []
    seen_words = 0
    char_set: Set[str] = set()

    rng = random.Random(locked.rng_seed)

    for _, text in _progress(_iter_cache(cache_path), desc="Scanning cache (words)", unit="samples"):
        # update base-char set (bounded by ascii in the end, but keep observed for metadata/debug)
        for ch in text:
            char_set.add(ch)

        for w in text.split():
            if not w or w in typed:
                continue
            if any(ch in boundaries for ch in w):
                continue
            if len(w) > locked.max_word_len:
                w = w[: locked.max_word_len]
            if len(w) < 2:
                continue
            if hygiene.is_value_fragment(w):
                continue

            hh.update(w, 1)

            # Reservoir sample of word occurrences.
            seen_words += 1
            if len(sample) < locked.sample_words:
                sample.append(w)
            else:
                j = rng.randint(0, seen_words - 1)
                if j < locked.sample_words:
                    sample[j] = w

    top_words = hh.topk(locked.top_words_k)
    # Filter invalid / too-long words for direct inclusion.
    top_words = [(w, c) for (w, c) in top_words if _is_valid_tok(w, boundaries, typed, max_len)]
    return top_words, sample, char_set


def _derive_subwords_from_words(
    top_words: List[Tuple[str, int]],
    *,
    boundaries: Set[str],
    typed_tokens: Sequence[str],
    max_len: int,
    vocab_size: int,
    locked: _Locked,
) -> List[Tuple[str, int]]:
    typed = set(typed_tokens)
    k = min(int(vocab_size * locked.top_subwords_k_mult), locked.top_subwords_k_cap)
    k = max(k, 200_000)
    hh = SpaceSaving(k)

    # Only use a prefix of top words for subword derivation.
    for w, c in top_words[: locked.top_words_for_subwords]:
        if not w or any(ch in boundaries for ch in w):
            continue
        n = len(w)
        # Bound cost for very long words.
        if n > 80:
            w = w[:80]
            n = len(w)
        for i in range(n):
            # generate substrings up to max_len
            for j in range(i + 2, min(n, i + max_len) + 1):
                sub = w[i:j]
                if sub in typed:
                    continue
                if hygiene.is_value_fragment(sub):
                    continue
                hh.update(sub, c)

    subwords = hh.topk(k)
    subwords = [(sw, c) for (sw, c) in subwords if _is_valid_tok(sw, boundaries, typed, max_len)]
    return subwords


# -----------------------------
# Unigram (hard-EM via Viterbi) + pruning
# -----------------------------
def _build_trie(tokens: Sequence[str]) -> Tuple[TokenTrie, Dict[str, int]]:
    tok2id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
    trie = TokenTrie()
    for t, i in tok2id.items():
        trie.add(t, i)
    return trie, tok2id


def _viterbi_counts(
    words: Sequence[str],
    tokens: Sequence[str],
    probs: List[float],
    max_len: int,
) -> List[float]:
    """Hard-EM: Viterbi tokenization counts over sampled words."""
    trie, _ = _build_trie(tokens)
    neglog = [(-math.log(max(p, 1e-12))) for p in probs]
    counts = [0.0 for _ in tokens]

    for w in _progress(words, desc="Unigram Viterbi", unit="words"):
        if not w:
            continue
        n = len(w)
        if n == 0:
            continue
        dp = [math.inf] * (n + 1)
        bp_tid = [-1] * (n + 1)
        bp_len = [0] * (n + 1)
        dp[0] = 0.0

        for i in range(n):
            if dp[i] == math.inf:
                continue
            matches = trie.iter_matches(w, i, max_len)
            if not matches:
                # fallback: single char must exist in tokens
                ch = w[i]
                # try direct
                # (we assume base chars are required and always present)
                # linear search would be too slow, but missing matches should be rare
                # because trie contains chars.
                matches = []
                # we let it fail and rely on required chars; if char is missing,
                # we skip this word.
            for tid, l in matches:
                j = i + l
                cost = dp[i] + neglog[tid]
                if cost < dp[j]:
                    dp[j] = cost
                    bp_tid[j] = tid
                    bp_len[j] = l

        if dp[n] == math.inf:
            continue
        # backtrack
        j = n
        while j > 0:
            tid = bp_tid[j]
            l = bp_len[j]
            if tid < 0 or l <= 0:
                break
            counts[tid] += 1.0
            j -= l

    return counts


def _prune_unigram_vocab(
    *,
    required: Set[str],
    initial_tokens: List[str],
    initial_counts: Dict[str, int],
    sample_words: List[str],
    vocab_size: int,
    max_len: int,
    locked: _Locked,
) -> List[str]:
    if vocab_size <= len(required):
        # Degenerate: cannot fit any learned tokens.
        return sorted(required)

    tokens = list(dict.fromkeys(initial_tokens))
    # initialize probs from counts
    raw = [float(initial_counts.get(t, 1)) for t in tokens]
    tot = sum(raw)
    probs = [c / tot for c in raw]

    target = vocab_size

    for it in range(locked.max_prune_iters):
        if len(tokens) <= target:
            break
        # Hard-EM counts
        vcounts = _viterbi_counts(sample_words, tokens, probs, max_len)
        # update probs with smoothing
        sm = locked.smoothing
        tot = sum(vcounts) + sm * len(tokens)
        probs = [(c + sm) / tot for c in vcounts]

        # rank removable tokens by (count * saving) - junk
        removable: List[Tuple[float, str]] = []
        for t, c in zip(tokens, vcounts):
            if t in required:
                continue
            saving = max(len(t) - 1, 1)
            score = (c * saving) - locked.junk_penalty_beta * hygiene.junk_score(t)
            removable.append((score, t))
        # if nothing removable, stop
        if not removable:
            break
        removable.sort(key=lambda kv: (kv[0], kv[1]))  # low score first

        need_remove = len(tokens) - target
        batch = max(int(len(tokens) * locked.prune_frac), 1)
        batch = min(batch, need_remove)

        to_remove = {t for _, t in removable[:batch]}
        tokens = [t for t in tokens if t not in to_remove]
        # renormalize probs list
        keep_mask = [t not in to_remove for t in tokens]  # wrong length after update, rebuild
        # recompute counts for new token list
        raw = [float(initial_counts.get(t, 1)) for t in tokens]
        tot = sum(raw)
        probs = [c / tot for c in raw]

    # Final sort: required first handled by caller.
    # Order learned tokens by descending probability proxy (count) then lexicographically.
    learned = [t for t in tokens if t not in required]
    learned.sort(key=lambda t: (-initial_counts.get(t, 1), t))
    final = list(required) + learned
    return final[:target]


# -----------------------------
# Artifact writing
# -----------------------------
def write_fast_tokenizer_json(outdir: str, token_to_id: Dict[str, int]) -> None:
    """Write a Rust `tokenizers` backend tokenizer.json.

    If the optional `tokenizers` dependency is unavailable, we skip generating
    tokenizer.json and the artifact will still work with the slow tokenizer
    (`CTokTokenizer`) via vocab.json.
    """
    try:
        from tokenizers import Tokenizer  # type: ignore
        from tokenizers.models import WordPiece  # type: ignore
        from tokenizers.processors import TemplateProcessing  # type: ignore
    except Exception:
        print(
            "[WARN] python package `tokenizers` not found; skipping tokenizer.json. "
            "Install `tokenizers` to enable CTokTokenizerFast.",
            file=sys.stderr,
        )
        return

    unk = "[UNK]"
    if unk not in token_to_id:
        raise ValueError("[UNK] must be in vocab")
    model = WordPiece(vocab=token_to_id, unk_token=unk, continuing_subword_prefix="")
    tok = Tokenizer(model)

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
    *,
    outdir: str,
    token_to_id: Dict[str, int],
    boundaries: Set[str],
    vocab_size_requested: int,
    max_len: int,
    fmt: str,
    text_key: str,
    label_key: Optional[str],
    model_max_length: int,
    lowercase: bool,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # vocab.json (debug / slow tokenizer)
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=True, indent=2)

    meta = {
        "match_special_tokens": False,
        "artifact_version": "ctok-fast-v1",
        "pipeline_locked": True,
        "lowercase": lowercase,
        "hygiene": hygiene_cfg.to_dict(),
        "pretokenizer": pretok_cfg.to_dict(),
        "hygiene_metrics": hygiene.vocab_hygiene_metrics(token_to_id.keys(), hygiene_cfg.typed_tokens),
        "build": {
            "trainer": "unigram_viterbi_prune",
            "format": fmt,
            "text_key": text_key,
            "label_key": label_key,
            "vocab_size_requested": vocab_size_requested,
            "vocab_size_actual": len(token_to_id),
            "max_len": max_len,
            "boundaries": sorted(list(boundaries)),
            "pretokenizer": "generic" if pretok_cfg.enabled else "none",
            "lowercase": lowercase,
        },
    }
    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)

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

    write_fast_tokenizer_json(outdir, token_to_id)

    # Copy code for trust_remote_code use.
    here = Path(__file__).resolve().parent
    for fn in ["tokenization_ctok.py", "tokenization_ctok_fast.py", "hygiene.py", "pretokenize.py"]:
        src = here / fn
        if src.exists():
            shutil.copy(str(src), os.path.join(outdir, fn))

    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# CTok Fast Tokenizer Artifact\n\n"
            "Load with: AutoTokenizer.from_pretrained(path, trust_remote_code=True)\n\n"
            "Files:\n"
            "- tokenizer.json: Rust backend (WordPiece greedy longest-match, empty continuation prefix)\n"
            "- vocab.json: token->id (debug / slow tokenizer)\n"
            "- ctok_meta.json: build metadata\n"
            "- tokenizer_config.json, special_tokens_map.json: Transformers integration\n"
        )


# -----------------------------
# Main build
# -----------------------------
def build_ctok_unigram_from_corpus(
    *,
    corpus: str,
    fmt: str,
    text_key: str,
    label_key: Optional[str],
    outdir: str,
    vocab_size: int,
    max_len: int,
    boundaries_str: str,
    lowercase: bool,
    pretokenizer_mode: str,
    no_hygiene: bool,
    model_max_length: int,
    max_samples: Optional[int],
) -> None:
    locked = _Locked()
    random.seed(locked.rng_seed)

    boundaries = parse_boundaries(boundaries_str)
    # ensure whitespace boundaries
    boundaries |= {" ", "\t", "\n", "\r"}

    hygiene_cfg = hygiene.default_hygiene_config()
    hygiene_cfg.enabled = not no_hygiene
    if not hygiene_cfg.enabled:
        hygiene_cfg.typed_tokens = []
        hygiene_cfg.patterns = []

    pretok_cfg = pretokenize.default_pretokenizer_config()
    pretok_cfg.enabled = pretokenizer_mode != "none"
    if not pretok_cfg.enabled:
        pretok_cfg.patterns = []

    cache_path = _build_or_load_cache(
        outdir=outdir,
        fmt=fmt,
        corpus_path=corpus,
        max_samples=max_samples,
        text_key=text_key,
        label_key=label_key,
        lowercase=lowercase,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        locked=locked,
    )

    top_words, sample_words, observed_chars = _collect_top_words_and_samples(
        cache_path,
        boundaries=boundaries,
        typed_tokens=hygiene_cfg.typed_tokens,
        max_len=max_len,
        locked=locked,
    )
    subwords = _derive_subwords_from_words(
        top_words,
        boundaries=boundaries,
        typed_tokens=hygiene_cfg.typed_tokens,
        max_len=max_len,
        vocab_size=vocab_size,
        locked=locked,
    )

    # Required tokens
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    required: List[str] = []
    required.extend(special)
    required.extend(sorted(set(hygiene_cfg.typed_tokens)))
    # Add boundaries as single-char tokens.
    required.extend(sorted(boundaries))
    # Add ASCII base chars for robustness.
    required.extend(sorted(hygiene.ascii_base_chars()))
    required_set = set(required)

    # Seed token pool
    seed_tokens: List[str] = []
    seed_counts: Dict[str, int] = {}
    for t in required:
        seed_tokens.append(t)
        seed_counts[t] = max(seed_counts.get(t, 0), 10)

    for w, c in top_words:
        if w in required_set:
            continue
        seed_tokens.append(w)
        seed_counts[w] = max(seed_counts.get(w, 0), c)

    for sw, c in subwords:
        if sw in required_set:
            continue
        seed_tokens.append(sw)
        seed_counts[sw] = max(seed_counts.get(sw, 0), c)

    # Final pruning to vocab_size using Unigram hard-EM on sample words.
    final_tokens = _prune_unigram_vocab(
        required=required_set,
        initial_tokens=seed_tokens,
        initial_counts=seed_counts,
        sample_words=sample_words,
        vocab_size=vocab_size,
        max_len=max_len,
        locked=locked,
    )

    # Build token_to_id (stable ordering)
    # Ensure specials first, then the rest in their existing order.
    ordered: List[str] = []
    for t in special:
        if t in final_tokens and t not in ordered:
            ordered.append(t)
    for t in final_tokens:
        if t not in ordered:
            ordered.append(t)

    token_to_id = {t: i for i, t in enumerate(ordered)}

    write_artifact(
        outdir=outdir,
        token_to_id=token_to_id,
        boundaries=boundaries,
        vocab_size_requested=vocab_size,
        max_len=max_len,
        fmt=fmt,
        text_key=text_key,
        label_key=label_key,
        model_max_length=model_max_length,
        lowercase=lowercase,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
    )

    # Helpful summary
    print(f"Wrote CTok artifact to: {outdir}")
    print(f"Vocab size: {len(token_to_id)} (requested {vocab_size})")
    print(f"Cache: {cache_path} ({os.path.getsize(cache_path)/1024/1024:.1f} MB)")


def build_ctok_from_samples(
    *,
    samples: Sequence[Tuple[Optional[str], str]],
    text_key: str,
    label_key: Optional[str],
    outdir: str,
    args: argparse.Namespace,
) -> None:
    locked = _Locked()
    random.seed(locked.rng_seed)

    boundaries = parse_boundaries(getattr(args, "boundaries", "=&?:/\\n\\t <>\\\"'"))
    boundaries |= {" ", "\t", "\n", "\r"}

    hygiene_cfg = hygiene.default_hygiene_config()
    hygiene_cfg.enabled = not bool(getattr(args, "no_hygiene", False))
    if not hygiene_cfg.enabled:
        hygiene_cfg.typed_tokens = []
        hygiene_cfg.patterns = []

    pretok_cfg = pretokenize.default_pretokenizer_config()
    pretok_cfg.enabled = str(getattr(args, "pretokenizer", "none")) != "none"
    if not pretok_cfg.enabled:
        pretok_cfg.patterns = []

    max_samples = getattr(args, "max_samples", 0)
    max_samples = None if max_samples is None or max_samples <= 0 else int(max_samples)

    cache_path = _build_or_load_cache_from_samples(
        outdir=outdir,
        samples=samples,
        max_samples=max_samples,
        lowercase=bool(getattr(args, "lowercase", False)),
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        locked=locked,
    )

    top_words, sample_words, observed_chars = _collect_top_words_and_samples(
        cache_path,
        boundaries=boundaries,
        typed_tokens=hygiene_cfg.typed_tokens,
        max_len=int(getattr(args, "max_len", 12)),
        locked=locked,
    )
    subwords = _derive_subwords_from_words(
        top_words,
        boundaries=boundaries,
        typed_tokens=hygiene_cfg.typed_tokens,
        max_len=int(getattr(args, "max_len", 12)),
        vocab_size=int(getattr(args, "vocab_size", 8192)),
        locked=locked,
    )

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    required: List[str] = []
    required.extend(sorted(hygiene_cfg.typed_tokens))
    required.extend(sorted(observed_chars))
    required.extend(sorted(boundaries))

    final_tokens = _prune_and_score_subwords(
        subwords=subwords,
        sample_words=sample_words,
        special=special,
        required=required,
        vocab_size=int(getattr(args, "vocab_size", 8192)),
        boundaries=boundaries,
        locked=locked,
    )

    ordered: List[str] = []
    for t in special:
        if t in final_tokens and t not in ordered:
            ordered.append(t)
    for t in final_tokens:
        if t not in ordered:
            ordered.append(t)

    token_to_id = {t: i for i, t in enumerate(ordered)}
    write_artifact(
        outdir=outdir,
        token_to_id=token_to_id,
        boundaries=boundaries,
        vocab_size_requested=int(getattr(args, "vocab_size", 8192)),
        max_len=int(getattr(args, "max_len", 12)),
        fmt=str(getattr(args, "format", "samples")),
        text_key=text_key,
        label_key=label_key,
        model_max_length=int(getattr(args, "model_max_length", 512)),
        lowercase=bool(getattr(args, "lowercase", False)),
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
    )

    print(f"Wrote CTok artifact to: {outdir}")
    print(f"Vocab size: {len(token_to_id)} (requested {getattr(args, 'vocab_size', 8192)})")
    print(f"Cache: {cache_path} ({os.path.getsize(cache_path)/1024/1024:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Path to corpus (txt/tsv/jsonl/parquet) or parquet directory")
    ap.add_argument("--format", default="parquet", choices=["txt", "tsv", "jsonl", "parquet"])
    ap.add_argument("--text_key", default="text", help="For jsonl/parquet: text field")
    ap.add_argument("--label_key", default="label", help="For jsonl/parquet: label field; set empty to disable")

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--boundaries", type=str, default="=&?:/\\n\\t <>\\\"'", help="Boundary characters (supports escapes)")
    ap.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0=all)")

    ap.add_argument("--no_hygiene", action="store_true", help="Disable hygiene replacements")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before hygiene/pretokenization")
    ap.add_argument("--pretokenizer", choices=["none", "generic"], default="generic")
    ap.add_argument("--model_max_length", type=int, default=512)

    args = ap.parse_args()
    max_samples = None if args.max_samples <= 0 else int(args.max_samples)
    label_key = args.label_key if args.label_key else None

    build_ctok_unigram_from_corpus(
        corpus=args.corpus,
        fmt=args.format,
        text_key=args.text_key,
        label_key=label_key,
        outdir=args.outdir,
        vocab_size=int(args.vocab_size),
        max_len=int(args.max_len),
        boundaries_str=str(args.boundaries),
        lowercase=bool(args.lowercase),
        pretokenizer_mode=str(args.pretokenizer),
        no_hygiene=bool(args.no_hygiene),
        model_max_length=int(args.model_max_length),
        max_samples=max_samples,
    )


if __name__ == "__main__":
    main()

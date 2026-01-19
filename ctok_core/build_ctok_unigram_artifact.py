from __future__ import annotations

"""Build a *standard* Hugging Face tokenizer artifact (no trust_remote_code).

This script produces a directory containing a `tokenizer.json` that embeds:
  - normalizer (Rust `tokenizers`)
  - pre_tokenizer (Rust `tokenizers`)
  - model: Unigram
  - post_processor: TemplateProcessing ([CLS] ... [SEP])

The resulting artifact loads with:

  AutoTokenizer.from_pretrained(outdir)

and exposes backend components:

  tok.backend_tokenizer.normalizer.normalize_str(...)
  tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(...)

Design notes
------------
* We compile CTok's hygiene + pretokenize rules into Rust as much as possible.
* Numeric handling is fully Rust: protect allowlist -> bucketize -> restore.
* Training uses a lightweight Unigram-style induction with pruning.

This is intentionally minimal but production-usable.
"""

import argparse
import gzip
import json
import math
import os
import multiprocessing as mp
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# Local imports (from repo) — these are *build-time only*.
try:
    import hygiene
    import pretokenize
except Exception as e:  # pragma: no cover
    raise RuntimeError("This builder expects hygiene.py and pretokenize.py alongside it.") from e


# -----------------------------
# Corpus readers
# -----------------------------
def iter_txt(path: str, max_samples: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_tsv(path: str, max_samples: Optional[int]) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            yield parts[1]
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_jsonl(path: str, max_samples: Optional[int], text_key: str) -> Iterator[str]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            x = obj.get(text_key)
            if x is None:
                continue
            yield str(x)
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_parquet(path: str, max_samples: Optional[int], text_key: str, *, waf_join: bool = False) -> Iterator[str]:
    # Build-time convenience (pyarrow preferred).
    try:
        import pyarrow.dataset as ds  # type: ignore

        dataset = ds.dataset(path, format="parquet")

        # Fast path for WAF dataset: join multiple columns into one string at Arrow level.
        if waf_join:
            import pyarrow as pa  # type: ignore
            import pyarrow.compute as pc  # type: ignore

            cols = ["method", "url", "protocol", "headers", "body"]
            scanner = dataset.scanner(columns=cols)
            n = 0
            for batch in scanner.to_batches():
                rb = batch
                m = pc.fill_null(rb.column(0).cast(pa.string()), "")
                u = pc.fill_null(rb.column(1).cast(pa.string()), "")
                p = pc.fill_null(rb.column(2).cast(pa.string()), "")
                h = pc.fill_null(rb.column(3).cast(pa.string()), "")
                b = pc.fill_null(rb.column(4).cast(pa.string()), "")

                # Element-wise concatenation in Arrow (avoids Python loops).
                t1 = pc.binary_join_element_wise([pa.array(["<METHOD> "] * len(rb)), m, pa.array(["\n"] * len(rb))], "")
                t2 = pc.binary_join_element_wise([pa.array(["<URL> "] * len(rb)), u, pa.array(["\n"] * len(rb))], "")
                t3 = pc.binary_join_element_wise([pa.array(["<PROT> "] * len(rb)), p, pa.array(["\n"] * len(rb))], "")
                t4 = pc.binary_join_element_wise([pa.array(["<HDR>\n"] * len(rb)), h, pa.array(["\n"] * len(rb))], "")
                t5 = pc.binary_join_element_wise([pa.array(["<BODY>\n"] * len(rb)), b, pa.array(["\n"] * len(rb))], "")
                text_arr = pc.binary_join_element_wise([t1, t2, t3, t4, t5], "")

                for x in text_arr.to_pylist():
                    if not x:
                        continue
                    yield x
                    n += 1
                    if max_samples is not None and n >= max_samples:
                        return
            return

        scanner = dataset.scanner(columns=[text_key])
        n = 0
        for batch in scanner.to_batches():
            table = batch.to_pydict()
            for x in table[text_key]:
                if x is None:
                    continue
                yield str(x)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return
        return
    except Exception:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path, columns=[text_key])
        if max_samples is not None:
            df = df.head(max_samples)
        for x in df[text_key].tolist():
            if x is None:
                continue
            yield str(x)


def corpus_iter(fmt: str, path: str, max_samples: Optional[int], text_key: str, *, waf_join: bool = False) -> Iterator[str]:
    if fmt == "txt":
        return iter_txt(path, max_samples)
    if fmt == "tsv":
        return iter_tsv(path, max_samples)
    if fmt == "jsonl":
        return iter_jsonl(path, max_samples, text_key=text_key)
    if fmt == "parquet":
        return iter_parquet(path, max_samples, text_key=text_key, waf_join=waf_join)
    raise ValueError(f"Unknown --format: {fmt}")


# -----------------------------
# Rust-compatible numeric normalization
# -----------------------------
@dataclass(frozen=True)
class NumNormSpec:
    allowed: Sequence[str]
    protect_prefix: str = "__ALLOW__"
    protect_suffix: str = "__"
    num3: str = "__NUM3__"
    num4: str = "__NUM4__"
    num5p: str = "__NUM5P__"


def _rust_style_number_norm_py(text: str, spec: NumNormSpec) -> str:
    """Python mirror of the Rust pipeline we encode in tokenizer.json.

    Steps:
      1) protect allowlist numbers by wrapping them
      2) bucketize 3/4/5+ digit numbers
      3) restore allowlist numbers
    """
    import re

    # Protect: wrap exact allowlist matches.
    # Note: we use word boundaries; this is close to Rust `regex` behavior.
    # Build a compact alternation.
    allowed = sorted(set(spec.allowed), key=lambda s: (-len(s), s))
    if allowed:
        pat = r"\b(" + "|".join(map(re.escape, allowed)) + r")\b"
        text = re.sub(pat, spec.protect_prefix + r"\1" + spec.protect_suffix, text)

    # Bucketize.
    text = re.sub(r"\b\d{5,}\b", spec.num5p, text)
    text = re.sub(r"\b\d{4}\b", spec.num4, text)
    text = re.sub(r"\b\d{3}\b", spec.num3, text)

    # Restore.
    text = re.sub(re.escape(spec.protect_prefix) + r"(\d+)" + re.escape(spec.protect_suffix), r"\1", text)
    return text


# -----------------------------
# Preprocess (build-time)
# -----------------------------
def preprocess(text: str, *, lowercase: bool, num_spec: NumNormSpec) -> str:
    if lowercase:
        text = text.lower()

    # Apply regex hygiene patterns (without the python number normalization, which we re-do).
    cfg = hygiene.default_hygiene_config()
    out = text
    for rgx, repl in cfg.compiled_patterns():
        out = rgx.sub(repl, out)
    out = _rust_style_number_norm_py(out, num_spec)

    # Apply pretokenize (spacing) as a build-time approximation of the Rust pre-tokenizer.
    out = pretokenize.apply_pretokenize(out, pretokenize.default_pretokenizer_config())
    return out


# -----------------------------
# Lightweight Unigram induction
# -----------------------------
@dataclass(frozen=True)
class TrainSpec:
    rng_seed: int = 0
    sample_words: int = 200_000
    max_word_len: int = 200
    max_len: int = 16
    num_workers: int = 0  # 0 => auto (cpu_count), 1 => single-process
    prune_iters: int = 6
    prune_frac: float = 0.20
    smoothing: float = 1e-4


def _reservoir_sample_words(texts: Iterable[str], *, spec: TrainSpec) -> List[str]:
    rng = random.Random(spec.rng_seed)
    sample: List[str] = []
    seen = 0
    for t in texts:
        for w in t.split():
            if not w:
                continue
            if len(w) > spec.max_word_len:
                w = w[: spec.max_word_len]
            if len(w) < 2:
                continue
            seen += 1
            if len(sample) < spec.sample_words:
                sample.append(w)
            else:
                j = rng.randint(0, seen - 1)
                if j < spec.sample_words:
                    sample[j] = w
    return sample


def _collect_substrings(words: Sequence[str], *, max_len: int) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for w in words:
        n = len(w)
        if n < 2:
            continue
        lim = min(n, 80)
        ww = w[:lim]
        for i in range(len(ww)):
            for j in range(i + 2, min(len(ww), i + max_len) + 1):
                cnt[ww[i:j]] += 1
    return cnt


class _TrieNode:
    __slots__ = ("children", "token")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.token: Optional[str] = None


class _Trie:
    """Trie for enumerating *all* token matches starting at a position.

    This avoids O(n * max_len) substring creation + set lookups for Viterbi.
    """

    def __init__(self, tokens: Sequence[str]):
        self.root = _TrieNode()
        for t in tokens:
            if not t:
                continue
            node = self.root
            for ch in t:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = _TrieNode()
                    node.children[ch] = nxt
                node = nxt
            node.token = t

    def matches(self, s: str, i: int, max_len: int) -> Iterator[Tuple[str, int]]:
        node = self.root
        j = i
        lim = min(len(s), i + max_len)
        while j < lim:
            nxt = node.children.get(s[j])
            if nxt is None:
                return
            node = nxt
            j += 1
            if node.token is not None:
                yield node.token, j


def _viterbi_tokenize_counts(word: str, trie: _Trie, token_cost: Dict[str, float], max_len: int) -> Counter[str]:
    """Viterbi over a single word using trie enumeration."""
    n = len(word)
    dp = [math.inf] * (n + 1)
    bp: List[Optional[Tuple[int, str]]] = [None] * (n + 1)  # (prev, tok)
    dp[0] = 0.0

    for i in range(n):
        if dp[i] == math.inf:
            continue
        for tok, j in trie.matches(word, i, max_len=max_len):
            c = token_cost[tok]
            nd = dp[i] + c
            if nd < dp[j]:
                dp[j] = nd
                bp[j] = (i, tok)

    if dp[n] == math.inf:
        return Counter()
    out: Counter[str] = Counter()
    k = n
    while k > 0 and bp[k] is not None:
        prev, tok = bp[k]
        out[tok] += 1
        k = prev
    return out


# ---- multiprocessing helpers (fork-friendly) ----
_MP_TRIE: Optional[_Trie] = None
_MP_COST: Optional[Dict[str, float]] = None
_MP_MAX_LEN: int = 16


def _mp_init(trie: _Trie, cost: Dict[str, float], max_len: int) -> None:
    global _MP_TRIE, _MP_COST, _MP_MAX_LEN
    _MP_TRIE = trie
    _MP_COST = cost
    _MP_MAX_LEN = max_len


def _mp_viterbi_counts(words: Sequence[str]) -> Counter[str]:
    assert _MP_TRIE is not None and _MP_COST is not None
    out: Counter[str] = Counter()
    for w in words:
        out.update(_viterbi_tokenize_counts(w, _MP_TRIE, _MP_COST, max_len=_MP_MAX_LEN))
    return out


def train_unigram(
    texts: Iterable[str],
    *,
    vocab_size: int,
    required_tokens: Sequence[str],
    max_len: int,
    spec: TrainSpec,
) -> Tuple[List[Tuple[str, float]], Dict[str, int]]:
    """Train a Unigram vocab and return (vocab_with_scores, token_to_id).

    Returns:
      vocab_with_scores: List[(token, logprob)]
      token_to_id: token -> id (aligned with vocab list order)
    """

    # 1) sample words
    sample_words = _reservoir_sample_words(texts, spec=spec)
    if not sample_words:
        raise ValueError("No words found after preprocessing.")

    # 2) candidate pool: substrings + required
    cand = _collect_substrings(sample_words, max_len=max_len)

    # Keep top candidates.
    # Heuristic: candidates ~= vocab_size * 50 (cap)
    k = min(max(vocab_size * 50, 200_000), 1_000_000)
    top = [t for t, _ in cand.most_common(k)]

    # Required tokens first.
    required = list(dict.fromkeys(required_tokens))
    tokens = list(dict.fromkeys(required + top))

    # Initialize counts and probs.
    init_counts = Counter()
    for w in sample_words:
        init_counts[w] += 1
    # naive prior: substring frequency
    for t in top:
        init_counts[t] += cand[t]

    def probs_from_counts(ts: Sequence[str], counts: Counter[str]) -> Dict[str, float]:
        sm = spec.smoothing
        tot = sum(float(counts.get(t, 0.0)) for t in ts) + sm * len(ts)
        return {t: (float(counts.get(t, 0.0)) + sm) / tot for t in ts}

    probs = probs_from_counts(tokens, init_counts)

    # 3) prune iterations
    target = vocab_size
    for _ in range(spec.prune_iters):
        if len(tokens) <= target:
            break

        token_cost = {t: -math.log(max(probs.get(t, 1e-12), 1e-12)) for t in tokens}
        trie = _Trie(tokens)

        # Viterbi counting is the main bottleneck. Parallelize across words.
        # We rely on Linux 'fork' to share the trie/cost dict cheaply.
        nw = spec.num_workers
        if nw <= 0:
            try:
                nw = max((os.cpu_count() or 2) - 1, 1)
            except Exception:
                nw = 1

        if nw == 1 or len(sample_words) < 50_000:
            vcounts: Counter[str] = Counter()
            for w in sample_words:
                vcounts.update(_viterbi_tokenize_counts(w, trie, token_cost, max_len=max_len))
        else:
            # Chunk the words to reduce IPC overhead.
            chunk = max(len(sample_words) // (nw * 8), 10_000)
            chunks = [sample_words[i : i + chunk] for i in range(0, len(sample_words), chunk)]

            ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
            with ctx.Pool(processes=nw, initializer=_mp_init, initargs=(trie, token_cost, max_len)) as pool:
                parts = pool.map(_mp_viterbi_counts, chunks)
            vcounts = Counter()
            for c in parts:
                vcounts.update(c)

        # update probs
        probs = probs_from_counts(tokens, vcounts)

        # removable ranking: low expected utility first
        removable = []
        for t in tokens:
            if t in required:
                continue
            c = vcounts.get(t, 0)
            saving = max(len(t) - 1, 1)
            score = float(c) * saving - 0.4 * hygiene.junk_score(t)
            removable.append((score, t))
        if not removable:
            break
        removable.sort(key=lambda x: (x[0], x[1]))

        need_remove = len(tokens) - target
        batch = max(int(len(tokens) * spec.prune_frac), 1)
        batch = min(batch, need_remove)
        to_remove = {t for _, t in removable[:batch]}
        tokens = [t for t in tokens if t not in to_remove]

    # Final logprobs.
    probs = probs  # last updated
    vocab = [(t, math.log(max(probs.get(t, 1e-12), 1e-12))) for t in tokens]

    # Token ids: stable order.
    token_to_id = {t: i for i, (t, _) in enumerate(vocab)}
    return vocab, token_to_id


# -----------------------------
# Tokenizer.json construction
# -----------------------------
def _make_allowlist_regex(nums: Sequence[str]) -> str:
    # Use a bounded alternation; sort by length to keep regex engine happy.
    uniq = sorted(set(nums), key=lambda s: (-len(s), s))
    return r"\\b(" + "|".join(map(lambda s: s.replace("\\", "\\\\"), uniq)) + r")\\b"


def build_tokenizer_json(
    *,
    vocab: List[Tuple[str, float]],
    token_to_id: Dict[str, int],
    lowercase: bool,
    num_spec: NumNormSpec,
    out_path: str,
) -> None:
    """Write a standard `tokenizer.json` with Rust normalizer/pre_tokenizer/model/post."""

    try:
        from tokenizers import Regex, Tokenizer
        from tokenizers.models import Unigram
        from tokenizers.normalizers import Lowercase, NFKC, Replace, Sequence as NormSeq
        from tokenizers.pre_tokenizers import Sequence as PreSeq
        from tokenizers.pre_tokenizers import Split, Whitespace
        from tokenizers.processors import TemplateProcessing
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Please `pip install tokenizers` to build tokenizer.json") from e

    # ---- model (Unigram) ----
    # tokenizers Unigram expects vocab as List[(token, score)] where score is typically logprob.
    unk = "[UNK]"
    if unk not in token_to_id:
        raise ValueError("[UNK] must be present in vocab")
    model = Unigram(vocab=vocab, unk_id=token_to_id[unk])
    tok = Tokenizer(model)

    # ---- normalizer ----
    # Start with NFKC for consistent unicode representation.
    norms = [NFKC()]
    if lowercase:
        norms.append(Lowercase())

    # Hygiene regex rules (rust-compatible) from hygiene.default_hygiene_config()
    hcfg = hygiene.default_hygiene_config()
    for p in hcfg.patterns:
        # Skip lookaround/backref patterns (Rust regex doesn't support lookaround).
        pat = p.pattern
        if "(?<" in pat or "(?=" in pat or "(?!" in pat or "(?<!" in pat or "(?<=" in pat or "\\1" in pat:
            continue
        norms.append(Replace(Regex(pat), p.replacement))

    # Numeric allowlist protect -> bucketize -> restore
    allow_pat = _make_allowlist_regex(num_spec.allowed)
    norms.append(Replace(Regex(allow_pat), num_spec.protect_prefix + "$1" + num_spec.protect_suffix))
    norms.append(Replace(Regex(r"\\b\\d{5,}\\b"), num_spec.num5p))
    norms.append(Replace(Regex(r"\\b\\d{4}\\b"), num_spec.num4))
    norms.append(Replace(Regex(r"\\b\\d{3}\\b"), num_spec.num3))
    # Restore protected allowlist numbers.
    restore_pat = Regex(r"__ALLOW__(\\d+)__") if (num_spec.protect_prefix, num_spec.protect_suffix) == ("__ALLOW__", "__") else Regex(
        (repr(num_spec.protect_prefix)[1:-1] + r"(\\d+)" + repr(num_spec.protect_suffix)[1:-1]).replace("\\\\", "\\")
    )
    norms.append(Replace(restore_pat, "$1"))

    tok.normalizer = NormSeq(norms)

    # ---- pre_tokenizer ----
    # CTok's pretokenize.py is effectively "isolate structural delimiters".
    # We implement an equivalent Rust pre-tokenizer using Split(..., isolated)
    # followed by Whitespace() to drop spacing.
    # Order matters: longer operators first.
    def _split_lit(lit: str) -> Split:
        # Convert literal to a safe regex.
        return Split(Regex(repr(lit)[1:-1]), behavior="isolated")

    pre: List[object] = []
    # Percent-hex first so '%' isn't separated prematurely.
    pre.append(Split(Regex(r"(%[0-9A-Fa-f]{2})"), behavior="isolated"))

    pre_ops = [
        "<!--",
        "-->",
        "<![CDATA[",
        "]]>",
        "</",
        "<?",
        "?>",
        "/>",
        "../",
        "..\\",
        "//",
        "\\\\",
        "&&",
        "||",
        "==",
        "!=",
        "<=",
        ">=",
    ]
    for op in pre_ops:
        pre.append(_split_lit(op))

    # Single-character delimiters.
    for ch in ["<", ">", "?", "&", "=", ":", ";", ",", "/", "\\", "(", ")", "[", "]", "{", "}", "|"]:
        pre.append(_split_lit(ch))

    # Finally drop whitespace (spaces/newlines/tabs).
    pre.append(Whitespace())
    tok.pre_tokenizer = PreSeq(pre)

    # ---- post_processor ----
    cls = "[CLS]"
    sep = "[SEP]"
    if cls in token_to_id and sep in token_to_id:
        tok.post_processor = TemplateProcessing(
            single=f"{cls} $A {sep}",
            pair=f"{cls} $A {sep} $B {sep}",
            special_tokens=[(cls, token_to_id[cls]), (sep, token_to_id[sep])],
        )

    tok.save(out_path)


def write_hf_artifact(
    *,
    outdir: str,
    tokenizer_json_path: str,
    model_max_length: int,
    meta: Dict[str, object],
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # tokenizer.json already written.
    # Minimal configs for AutoTokenizer without remote code.
    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "model_max_length": int(model_max_length),
                "padding_side": "right",
                "truncation_side": "right",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(outdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["txt", "tsv", "jsonl", "parquet"], default="txt")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--waf_join", action="store_true", help="For WAF parquet with columns method/url/protocol/headers/body: build text on the fly")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab_size", type=int, default=2048)
    ap.add_argument("--max_len", type=int, default=16)
    ap.add_argument("--sample_words", type=int, default=200000)
    ap.add_argument("--prune_iters", type=int, default=6)
    ap.add_argument("--prune_frac", type=float, default=0.20)
    ap.add_argument("--num_workers", type=int, default=0, help="0=auto, 1=single-process")
    ap.add_argument("--lowercase", action="store_true")
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(list(argv) if argv is not None else None)

    # Numeric allowlist (reuse your set).
    num_spec = NumNormSpec(allowed=sorted(hygiene.ALLOWED_NUMBERS))

    # Stream and preprocess corpus; training performs its own reservoir sampling,
    # so we do NOT materialize the whole corpus in memory.
    it = corpus_iter(args.format, args.corpus, args.max_samples, text_key=args.text_key, waf_join=bool(args.waf_join))

    def _preprocessed() -> Iterator[str]:
        for x in it:
            px = preprocess(x, lowercase=args.lowercase, num_spec=num_spec)
            if px:
                yield px

    # Required tokens: special + typed + buckets
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    typed = hygiene.default_hygiene_config().typed_tokens
    required = special + typed + [num_spec.num3, num_spec.num4, num_spec.num5p]

    # Also include single ASCII chars to ensure coverage.
    for ch in hygiene.ascii_base_chars():
        required.append(ch)

    train_spec = TrainSpec(
        rng_seed=args.seed,
        max_len=args.max_len,
        sample_words=int(args.sample_words),
        prune_iters=int(args.prune_iters),
        prune_frac=float(args.prune_frac),
        num_workers=int(args.num_workers),
    )
    vocab, token_to_id = train_unigram(
        _preprocessed(),
        vocab_size=args.vocab_size,
        required_tokens=required,
        max_len=args.max_len,
        spec=train_spec,
    )

    # Write tokenizer.json
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    tok_json = os.path.join(outdir, "tokenizer.json")
    build_tokenizer_json(
        vocab=vocab,
        token_to_id=token_to_id,
        lowercase=bool(args.lowercase),
        num_spec=num_spec,
        out_path=tok_json,
    )

    meta = {
        "artifact_version": "ctok-unigram-standard-v1",
        "no_trust_remote_code": True,
        "model": "unigram",
        "max_len": int(args.max_len),
        "vocab_size": int(args.vocab_size),
        "lowercase": bool(args.lowercase),
        "num_norm": {
            "protect_prefix": num_spec.protect_prefix,
            "protect_suffix": num_spec.protect_suffix,
            "num3": num_spec.num3,
            "num4": num_spec.num4,
            "num5p": num_spec.num5p,
            "allowed_count": len(num_spec.allowed),
        },
        "build": {
            "format": args.format,
            "corpus": os.path.abspath(args.corpus),
            "text_key": args.text_key,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
    }

    write_hf_artifact(outdir=outdir, tokenizer_json_path=tok_json, model_max_length=args.model_max_length, meta=meta)

    # Optional: also dump a simple vocab.json for inspection.
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)

    print(f"Wrote standard tokenizer artifact to: {outdir}")
    print(f"Vocab size: {len(token_to_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

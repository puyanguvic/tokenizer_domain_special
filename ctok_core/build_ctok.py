from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import hygiene
import pretokenize

try:
    from tqdm import tqdm as _tqdm
except Exception:
    _tqdm = None


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


def parse_escaped_chars(s: str) -> Set[str]:
    """Parse a boundary string with escapes (e.g. "=&?:/\\n\\t \"'")."""
    decoded = s.encode("utf-8").decode("unicode_escape")
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


def iter_tsv(path: str, max_samples: Optional[int]) -> Iterable[Tuple[str, str]]:
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


def iter_jsonl(path: str, text_key: str, label_key: Optional[str], max_samples: Optional[int]) -> Iterable[Tuple[Optional[str], str]]:
    import json as _json

    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            x = str(obj[text_key])
            y = str(obj[label_key]) if label_key and label_key in obj else None
            yield y, x
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def iter_parquet(path: str, text_key: str, label_key: Optional[str], max_samples: Optional[int]) -> Iterable[Tuple[Optional[str], str]]:
    """Stream rows from parquet (single file or directory)."""
    try:
        import pyarrow.dataset as ds
    except Exception:
        ds = None

    if ds is not None:
        dataset = ds.dataset(path, format="parquet")
        cols = [text_key] + ([label_key] if label_key else [])
        scanner = dataset.scan(columns=cols)
        n = 0
        for batch in scanner.to_batches():
            text_col = batch.column(text_key)
            label_col = batch.column(label_key) if label_key else None
            for i in range(batch.num_rows):
                x = text_col[i].as_py()
                if x is None:
                    continue
                y = label_col[i].as_py() if label_col is not None else None
                yield (None if y is None else str(y)), str(x)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return
        return

    # Fallback: pandas (loads selected cols into memory)
    import pandas as pd

    df = pd.read_parquet(path, columns=[text_key] + ([label_key] if label_key else []))
    if max_samples is not None:
        df = df.head(max_samples)
    for _, row in df.iterrows():
        x = row[text_key]
        if x is None:
            continue
        y = row[label_key] if label_key else None
        yield (None if y is None else str(y)), str(x)


def collect_base_chars(
    pairs: Iterable[Tuple[Optional[str], str]],
    max_chars: int,
    add_ascii: bool = True,
    extra_tokens: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
) -> Set[str]:
    chars: Set[str] = set()
    if add_ascii:
        chars.update(hygiene.ascii_base_chars())
    if extra_tokens:
        chars.update(extra_tokens)
    for idx, (_, x) in enumerate(pairs):
        for ch in x:
            chars.add(ch)
            if len(chars) >= max_chars:
                return chars
        if max_samples is not None and (idx + 1) >= max_samples:
            break
    return chars


def collect_candidates(
    pairs: Iterable[Tuple[Optional[str], str]],
    boundaries: Set[str],
    max_len: int,
    min_freq: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
) -> Counter[str]:
    cnt: Counter[str] = Counter()
    total = len(pairs) if hasattr(pairs, "__len__") else None
    iterator = _progress(pairs, total=total, desc="Collecting candidates", unit="samples")
    for _, text in iterator:
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

    # freq filter
    for k in list(cnt.keys()):
        if cnt[k] < min_freq:
            del cnt[k]
    return cnt


def estimate_gain(tok: str, freq: int) -> int:
    # Approx: if base is 1-char tokens, adding tok length L saves (L-1) per occurrence.
    L = len(tok)
    return freq * max(L - 1, 0)


def estimate_mi_scores(
    pairs: Iterable[Tuple[Optional[str], str]],
    candidates: Sequence[str],
    max_samples: int,
) -> Dict[str, float]:
    """Cheap MI proxy: treat token as a substring indicator; count per label.

    This is *not* your probe-based directed distortion, but a light semantic signal for early sweeps.
    """
    cand_set = set(candidates)
    label_counts: Counter[str] = Counter()
    token_label_counts: Dict[str, Counter[str]] = {c: Counter() for c in cand_set}

    n = 0
    total = len(pairs) if hasattr(pairs, "__len__") else None
    iterator = _progress(pairs, total=total, desc="Estimating MI", unit="samples")
    for y, x in iterator:
        if y is None:
            continue
        y = str(y)
        label_counts[y] += 1
        # crude: substring test
        for c in cand_set:
            if c in x:
                token_label_counts[c][y] += 1
        n += 1
        if n >= max_samples:
            break

    total = sum(label_counts.values())
    if total == 0:
        return {c: 0.0 for c in cand_set}

    # global p(y)
    py = {y: label_counts[y] / total for y in label_counts}

    def kl(p: Dict[str, float], q: Dict[str, float]) -> float:
        import math

        out = 0.0
        for y, pv in p.items():
            if pv <= 0:
                continue
            qv = q.get(y, 1e-12)
            out += pv * (math.log(pv + 1e-12) - math.log(qv + 1e-12))
        return out

    scores: Dict[str, float] = {}
    for c in cand_set:
        cy = token_label_counts[c]
        ctot = sum(cy.values())
        if ctot == 0:
            scores[c] = 0.0
            continue
        p_y_given_c = {y: cy[y] / ctot for y in cy}
        scores[c] = kl(p_y_given_c, py)
    return scores


def build_vocab(
    base_chars: Set[str],
    candidates: Counter[str],
    vocab_size: int,
    special_tokens: List[str],
    required_tokens: Sequence[str],
    semantic_mode: str,
    lambda_sem: float,
    mi_scores: Optional[Dict[str, float]],
    junk_penalty_beta: float = 0.0,
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {}
    cur = 0
    for st in special_tokens:
        token_to_id[st] = cur
        cur += 1

    for tok in sorted(set(required_tokens)):
        if tok not in token_to_id:
            token_to_id[tok] = cur
            cur += 1

    for ch in sorted(base_chars):
        if ch not in token_to_id:
            token_to_id[ch] = cur
            cur += 1

    base_size = len(token_to_id)
    if vocab_size <= base_size:
        raise ValueError(f"vocab_size={vocab_size} too small for base_size={base_size}")

    scored: List[Tuple[float, str]] = []
    for tok, f in candidates.items():
        if tok in token_to_id:
            continue
        g = float(estimate_gain(tok, f))
        s = g
        if semantic_mode == "mi" and mi_scores is not None:
            s = g + lambda_sem * float(mi_scores.get(tok, 0.0))
        if junk_penalty_beta > 0:
            s -= junk_penalty_beta * hygiene.junk_score(tok)
        scored.append((s, tok))

    scored.sort(reverse=True)
    for s, tok in scored:
        if len(token_to_id) >= vocab_size:
            break
        if s <= 0:
            continue
        token_to_id[tok] = cur
        cur += 1

    return token_to_id


def collect_doc_stats(
    pairs: Iterable[Tuple[Optional[str], str]],
    candidates: Set[str],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
) -> Tuple[Counter[str], Dict[str, int]]:
    doc_freq: Counter[str] = Counter()
    max_in_doc: Dict[str, int] = {}
    for _, text in pairs:
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


def filter_candidates(
    candidates: Counter[str],
    pairs: Iterable[Tuple[Optional[str], str]],
    boundaries: Set[str],
    max_len: int,
    allow_boundary_at_ends: bool,
    max_chars_per_sample: int,
    filter_value_fragments: bool,
    typed_tokens: Sequence[str],
    min_doc_freq: int,
    max_doc_concentration: float,
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
        pairs,
        candidates=set(filtered.keys()),
        boundaries=boundaries,
        max_len=max_len,
        allow_boundary_at_ends=allow_boundary_at_ends,
        max_chars_per_sample=max_chars_per_sample,
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


def build_tokenizer_json_wordpiece(token_to_id: Dict[str, int], out_path: str):
    """Create a Rust-backed tokenizer.json using WordPiece greedy longest-match.

    Trick: set continuing_subword_prefix="" so WordPiece acts like CTok's greedy matching.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.processors import TemplateProcessing

    model = WordPiece(vocab=token_to_id, unk_token="[UNK]", continuing_subword_prefix="")
    tok = Tokenizer(model)

    # No pre-tokenizer: run on the full string (so it can match across spaces if vocab allows)
    # You can add a normalizer here if you want NFKC etc; keep it minimal for determinism.

    cls_id = token_to_id["[CLS]"]
    sep_id = token_to_id["[SEP]"]

    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    tok.save(out_path)


def write_artifact(
    outdir: str,
    token_to_id: Dict[str, int],
    boundaries: Set[str],
    build_args: Dict[str, object],
    emit_code: bool,
    hygiene_cfg: hygiene.HygieneConfig,
    pretok_cfg: pretokenize.PreTokenizerConfig,
    hygiene_metrics: Dict[str, float],
):
    os.makedirs(outdir, exist_ok=True)

    # vocab.json (useful for inspection + slow fallback)
    with open(os.path.join(outdir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)

    # tokenizer.json for fast tokenizer
    build_tokenizer_json_wordpiece(token_to_id, os.path.join(outdir, "tokenizer.json"))

    meta = {
        "artifact_version": "ctok-fast-v1",
        "match_rule": "left-to-right longest-match (WordPiece greedy, continuing_subword_prefix='')",
        "boundary_chars": sorted(list(boundaries)),
        "model_max_length": int(build_args.get("model_max_length", 512)),
        "pipeline_locked": True,
        "lowercase": bool(build_args.get("lowercase", False)),
        "hygiene": hygiene_cfg.to_dict(),
        "pretokenizer": pretok_cfg.to_dict(),
        "hygiene_metrics": hygiene_metrics,
        "build": build_args,
    }
    with open(os.path.join(outdir, "ctok_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # AutoTokenizer mapping: slow + fast (tuple/list required)
    tok_cfg = {
        "tokenizer_class": "CTokTokenizerFast",
        "auto_map": {
            "AutoTokenizer": [
                "tokenization_ctok.CTokTokenizer",
                "tokenization_ctok_fast.CTokTokenizerFast",
            ]
        },
        "model_max_length": int(build_args.get("model_max_length", 512)),
        "padding_side": "right",
        "truncation_side": "right",
    }
    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=False, indent=2)

    sp_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(outdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(sp_map, f, ensure_ascii=False, indent=2)

    if emit_code:
        # Copy the tokenizer implementation files into the artifact folder
        here = Path(__file__).resolve().parent
        shutil.copy2(str(here / "tokenization_ctok.py"), os.path.join(outdir, "tokenization_ctok.py"))
        shutil.copy2(str(here / "tokenization_ctok_fast.py"), os.path.join(outdir, "tokenization_ctok_fast.py"))
        shutil.copy2(str(here / "hygiene.py"), os.path.join(outdir, "hygiene.py"))
        shutil.copy2(str(here / "pretokenize.py"), os.path.join(outdir, "pretokenize.py"))

    with open(os.path.join(outdir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# CTok Tokenizer Artifact (Fast)\n\n"
            "Files:\n"
            "- tokenizer.json: Rust-backed WordPiece tokenizer configured for greedy longest-match\n"
            "- vocab.json: token->id mapping (also used by slow fallback)\n"
            "- ctok_meta.json: build metadata + boundary chars\n"
            "- tokenizer_config.json + special_tokens_map.json: Transformers loading\n\n"
            "Load with:\n"
            "```python\n"
            "from transformers import AutoTokenizer\n"
            "tok = AutoTokenizer.from_pretrained('PATH', trust_remote_code=True)\n"
            "```\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="path to corpus file or directory")
    ap.add_argument("--format", choices=["txt", "tsv", "jsonl", "parquet"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_freq", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--max_chars_per_sample", type=int, default=4096)
    ap.add_argument("--boundaries", type=str, default="=&?:/\\n\\t <>\\\"'", help="boundary chars (supports escapes)")
    ap.add_argument("--no_boundary_ends", action="store_true")
    ap.add_argument("--text_key", type=str, default="text")
    ap.add_argument("--label_key", type=str, default="label")
    ap.add_argument("--semantic_mode", choices=["none", "mi"], default="none")
    ap.add_argument("--lambda_sem", type=float, default=0.0)
    ap.add_argument("--mi_max_samples", type=int, default=20000)
    ap.add_argument("--mi_top_k", type=int, default=50000)
    ap.add_argument("--max_base_chars", type=int, default=5000)
    ap.add_argument("--base_chars_max_samples", type=int, default=200000)
    ap.add_argument("--no_ascii_base", action="store_true")
    ap.add_argument("--model_max_length", type=int, default=512)
    ap.add_argument("--emit_code", action="store_true")
    ap.add_argument("--no_hygiene", action="store_true")
    ap.add_argument("--lowercase", action="store_true", help="Lowercase text before hygiene/pretokenization")
    ap.add_argument("--pretokenizer", choices=["none", "generic"], default="none")
    ap.add_argument("--no_filter_value_fragments", action="store_true")
    ap.add_argument("--min_doc_freq", type=int, default=1)
    ap.add_argument("--max_doc_concentration", type=float, default=1.0)
    ap.add_argument("--junk_penalty_beta", type=float, default=0.5)

    args = ap.parse_args()

    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    boundaries = parse_escaped_chars(args.boundaries)

    # Readers
    def reader() -> Iterable[Tuple[Optional[str], str]]:
        if args.format == "txt":
            return iter_txt(args.corpus, args.max_samples)
        if args.format == "tsv":
            return iter_tsv(args.corpus, args.max_samples)
        if args.format == "jsonl":
            lk = args.label_key if args.label_key else None
            return iter_jsonl(args.corpus, args.text_key, lk, args.max_samples)
        if args.format == "parquet":
            lk = args.label_key if args.label_key else None
            return iter_parquet(args.corpus, args.text_key, lk, args.max_samples)
        raise ValueError("unsupported")

    # We need to iterate multiple times; materialize a sample list (bounded).
    pairs = list(reader())
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
        pairs = [(y, x.lower()) for y, x in pairs]
    if hygiene_cfg.enabled:
        pairs = [(y, hygiene.apply_hygiene(x, hygiene_cfg)) for y, x in _progress(pairs, desc="Applying hygiene", unit="samples")]
    if pretok_cfg.enabled:
        pairs = [(y, pretokenize.apply_pretokenize(x, pretok_cfg)) for y, x in _progress(pairs, desc="Applying pretokenizer", unit="samples")]

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # Base chars: ASCII + observed chars
    base_chars = collect_base_chars(
        pairs,
        max_chars=args.max_base_chars,
        add_ascii=not args.no_ascii_base,
        extra_tokens=hygiene_cfg.typed_tokens,
        max_samples=args.base_chars_max_samples,
    )
    # Ensure boundary chars present in base
    base_chars.update(boundaries)

    # Candidates
    candidates = collect_candidates(
        pairs,
        boundaries=boundaries,
        max_len=args.max_len,
        min_freq=args.min_freq,
        allow_boundary_at_ends=not args.no_boundary_ends,
        max_chars_per_sample=args.max_chars_per_sample,
    )
    candidates = filter_candidates(
        candidates=candidates,
        pairs=pairs,
        boundaries=boundaries,
        max_len=args.max_len,
        allow_boundary_at_ends=not args.no_boundary_ends,
        max_chars_per_sample=args.max_chars_per_sample,
        filter_value_fragments=not args.no_filter_value_fragments,
        typed_tokens=hygiene_cfg.typed_tokens,
        min_doc_freq=args.min_doc_freq,
        max_doc_concentration=args.max_doc_concentration,
    )

    mi_scores = None
    if args.semantic_mode == "mi":
        # restrict to top-k by frequency for tractability
        top = [t for t, _ in candidates.most_common(args.mi_top_k)]
        mi_scores = estimate_mi_scores(pairs, top, max_samples=args.mi_max_samples)

    vocab = build_vocab(
        base_chars=base_chars,
        candidates=candidates,
        vocab_size=args.vocab_size,
        special_tokens=special,
        required_tokens=hygiene_cfg.typed_tokens,
        semantic_mode=args.semantic_mode,
        lambda_sem=args.lambda_sem,
        mi_scores=mi_scores,
        junk_penalty_beta=args.junk_penalty_beta,
    )

    build_args = {
        "format": args.format,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(vocab),
        "max_len": args.max_len,
        "min_freq": args.min_freq,
        "max_samples": args.max_samples,
        "max_chars_per_sample": args.max_chars_per_sample,
        "semantic_mode": args.semantic_mode,
        "lambda_sem": args.lambda_sem,
        "model_max_length": args.model_max_length,
        "no_boundary_ends": bool(args.no_boundary_ends),
        "no_ascii_base": bool(args.no_ascii_base),
        "max_base_chars": args.max_base_chars,
        "base_chars_max_samples": args.base_chars_max_samples,
        "text_key": args.text_key,
        "label_key": args.label_key,
        "pretokenizer": args.pretokenizer,
        "lowercase": bool(args.lowercase),
    }

    build_args.update(
        {
            "hygiene_enabled": hygiene_cfg.enabled,
            "filter_value_fragments": not args.no_filter_value_fragments,
            "min_doc_freq": args.min_doc_freq,
            "max_doc_concentration": args.max_doc_concentration,
            "junk_penalty_beta": args.junk_penalty_beta,
        }
    )

    write_artifact(
        args.outdir,
        vocab,
        boundaries,
        build_args,
        emit_code=args.emit_code,
        hygiene_cfg=hygiene_cfg,
        pretok_cfg=pretok_cfg,
        hygiene_metrics=hygiene.vocab_hygiene_metrics(vocab.keys(), hygiene_cfg.typed_tokens),
    )

    print(f"Wrote CTok FAST artifact to: {args.outdir}")
    print(f"Vocab size: {len(vocab)} (requested {args.vocab_size})")
    print(f"Candidates kept: {len(candidates)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import common first to bootstrap repo imports when running from a checkout.
from experiments.common import levenshtein, results_root, save_json

from cit_tokenizers.corpus import load_dataset_by_name, resolve_dataset_key


def _ws_normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _json_sort_keys(s: str) -> str:
    """Try to parse JSON and dump with sorted keys; if parsing fails return input."""
    try:
        obj = json.loads(s)
        return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        return s


def _add_delim_spaces(s: str) -> str:
    # Insert spaces around common delimiters, a benign formatting drift.
    s = re.sub(r"([,:;=\[\]{}()<>])", r" \1 ", s)
    return _ws_normalize(s)


def _shuffle_html_attrs(s: str, rng: random.Random) -> str:
    # Very light-weight attribute shuffler for patterns like: <tag a="1" b="2">
    def repl(m: re.Match[str]) -> str:
        tag = m.group(1)
        attrs = m.group(2).strip()
        if not attrs:
            return m.group(0)
        parts = re.split(r"\s+", attrs)
        parts = [p for p in parts if p]
        if len(parts) <= 1:
            return m.group(0)
        rng.shuffle(parts)
        return f"<{tag} {' '.join(parts)}>"

    return re.sub(r"<([A-Za-z0-9:_-]+)\s+([^>]+)>", repl, s)


@dataclass
class E3Config:
    dataset: str
    split: str = "val"
    max_samples: int = 2000
    seed: int = 42


def run_e3(*, cfg: E3Config, tokenizer, tokenizer_name: str) -> Path:
    rng = random.Random(cfg.seed)

    ds_key = resolve_dataset_key(cfg.dataset)
    ds = load_dataset_by_name(ds_key, split=cfg.split)
    n = min(int(cfg.max_samples), len(ds))
    rows = [ds[i] for i in range(n)]

    def _get_text(row: Any) -> str:
        if isinstance(row, dict):
            for k in ("formatted", "text", "body", "url", "content"):
                if k in row and row[k] is not None:
                    v = str(row[k]).strip()
                    if v:
                        return v
        return str(row)

    texts = [_get_text(r) for r in rows]

    # Domain-appropriate benign drifts.
    transforms: List[Tuple[str, Callable[[str], str]]] = [
        ("ws", _ws_normalize),
        ("delim_spaces", _add_delim_spaces),
    ]
    if ds_key in {"waf"}:
        transforms.append(("json_sort_keys", _json_sort_keys))
    if ds_key in {"phish_html"}:
        transforms.append(("html_attr_shuffle", lambda s: _shuffle_html_attrs(s, rng)))

    stats: Dict[str, Dict[str, float]] = {}

    base_ids = tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

    for name, fn in transforms:
        pert = [fn(t) for t in texts]
        pert_ids = tokenizer(pert, truncation=False, add_special_tokens=False)["input_ids"]

        # Token length jitter and token-level edit distance.
        lens0 = [len(x) for x in base_ids]
        lens1 = [len(x) for x in pert_ids]
        len_abs = [abs(a - b) for a, b in zip(lens0, lens1)]
        ed = [levenshtein(a, b) for a, b in zip(base_ids, pert_ids)]

        stats[name] = {
            "n": float(n),
            "avg_len": float(sum(lens0) / max(1, n)),
            "avg_len_pert": float(sum(lens1) / max(1, n)),
            "avg_abs_len_jitter": float(sum(len_abs) / max(1, n)),
            "avg_edit_distance": float(sum(ed) / max(1, n)),
        }

    outdir = results_root() / ds_key / "e3_robustness" / tokenizer_name
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "dataset": ds_key,
            "split": cfg.split,
            "tokenizer": tokenizer_name,
            "max_samples": int(n),
            "transforms": stats,
        },
        outdir / "metrics.json",
    )
    return outdir

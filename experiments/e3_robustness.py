from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Import common first to bootstrap repo imports when running from a checkout.
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.common import (
    cosine_distance,
    detect_label_key,
    get_text,
    infer_num_labels,
    levenshtein,
    remap_labels_to_contiguous,
    results_root,
    save_json,
    to_int_label,
)

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
    with_probe: bool = True
    probe_dim: int = 64
    probe_epochs: int = 2
    lr: float = 5e-3
    label_key: Optional[str] = None


class _TokDataset(Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class MeanPoolProbe(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_labels: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        x = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        return self.head(x), x


def run_e3(*, cfg: E3Config, tokenizer, tokenizer_name: str) -> Path:
    rng = random.Random(cfg.seed)

    ds_key = resolve_dataset_key(cfg.dataset)
    ds = load_dataset_by_name(ds_key, split=cfg.split)
    n = min(int(cfg.max_samples), len(ds))
    rows = [ds[i] for i in range(n)]
    texts = [get_text(r, dataset_key=ds_key) for r in rows]

    label_key = cfg.label_key or (detect_label_key(rows[0]) if isinstance(rows[0], dict) else None)
    raw_int: list[int] = []
    if label_key is not None and isinstance(rows[0], dict):
        raw_int = [to_int_label(r.get(label_key)) for r in rows]  # type: ignore[arg-type]
        raw_int, mapping = remap_labels_to_contiguous(raw_int)
        num_labels = infer_num_labels(raw_int)
    else:
        mapping = {}
        num_labels = 0

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

    probe_stats: Dict[str, Dict[str, float]] = {}
    if cfg.with_probe and raw_int:
        # Train a tiny probe on the base texts and measure stability under perturbations.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
        model = MeanPoolProbe(vocab_size=vocab_size, dim=int(cfg.probe_dim), num_labels=int(num_labels)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
        loss_fn = nn.CrossEntropyLoss()

        enc_base = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        y = torch.tensor(raw_int, dtype=torch.long)
        dl = DataLoader(_TokDataset(enc_base, y), batch_size=64, shuffle=True)
        for _ in range(int(cfg.probe_epochs)):
            model.train()
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids=input_ids, attention_mask=attn)
                loss = loss_fn(logits, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            logits0, emb0 = model(enc_base["input_ids"].to(device), enc_base["attention_mask"].to(device))
            p0 = torch.softmax(logits0, dim=-1)
            pred0 = torch.argmax(p0, dim=-1)

        for name, fn in transforms:
            pert = [fn(t) for t in texts]
            enc1 = tokenizer(pert, truncation=True, padding=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                logits1, emb1 = model(enc1["input_ids"].to(device), enc1["attention_mask"].to(device))
                p1 = torch.softmax(logits1, dim=-1)
                pred1 = torch.argmax(p1, dim=-1)
                flip = float((pred0 != pred1).float().mean().item())
                conf0 = float(p0.max(dim=-1).values.mean().item())
                conf1 = float(p1.max(dim=-1).values.mean().item())
                # Average symmetric KL between predictive distributions.
                kl01 = torch.sum(p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log()), dim=-1).mean()
                kl10 = torch.sum(p1 * (p1.clamp_min(1e-12).log() - p0.clamp_min(1e-12).log()), dim=-1).mean()
                skl = float(0.5 * (kl01 + kl10).item())
                rep_drift = cosine_distance(emb0, emb1)
            probe_stats[name] = {
                "flip_rate": float(flip),
                "avg_maxprob_base": float(conf0),
                "avg_maxprob_pert": float(conf1),
                "avg_sym_kl": float(skl),
                "avg_cosine_distance": float(rep_drift),
            }

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
            "probe": probe_stats,
            "with_probe": bool(cfg.with_probe),
            "label_key": label_key,
            "num_labels": int(num_labels) if raw_int else 0,
            "label_mapping": mapping,
        },
        outdir / "metrics.json",
    )
    return outdir

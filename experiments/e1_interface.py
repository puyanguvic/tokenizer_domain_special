from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Import common first to bootstrap repo imports when running from a checkout.
from experiments.common import detect_label_key, results_root, save_csv, save_json, to_int_label

from cit_tokenizers.corpus import load_dataset_by_name, resolve_dataset_key


@dataclass
class E1Config:
    dataset: str
    split: str = "val"
    max_samples: int = 20000
    batch_size: int = 64
    max_length: int = 512
    probe_dim: int = 64
    probe_epochs: int = 2
    lr: float = 5e-3
    seed: int = 42
    label_key: Optional[str] = None


class _TokDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.enc = encodings
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


class MeanPoolProbe(nn.Module):
    """A lightweight probe to estimate label cross-entropy from tokenized sequences.

    This is intentionally small and used only as an interface diagnostic.
    """

    def __init__(self, vocab_size: int, dim: int, num_labels: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.emb(input_ids)
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        x = (x * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        return self.head(x)


@torch.no_grad()
def _avg_len(attention_mask: torch.Tensor) -> float:
    return float(attention_mask.sum(dim=1).float().mean().item())


def run_e1(
    *,
    cfg: E1Config,
    tokenizer,
    tokenizer_name: str,
) -> Path:
    """Run E1 for a single tokenizer. Returns the result directory."""

    torch.manual_seed(cfg.seed)

    ds_key = resolve_dataset_key(cfg.dataset)
    ds = load_dataset_by_name(ds_key, split=cfg.split)
    if len(ds) == 0:
        raise ValueError(f"Empty dataset split: {ds_key}:{cfg.split}")

    label_key = cfg.label_key or detect_label_key(ds[0])
    n = min(cfg.max_samples, len(ds))
    rows = [ds[i] for i in range(n)]
    texts = [str(r.get("text") or r.get("body") or r.get("url") or r.get("content") or "") for r in rows]
    # Prefer dataset-specific formatter if available: the corpus export uses it.
    # Here we keep it simple: if a formatted field exists, use it.
    if "formatted" in rows[0]:
        texts = [str(r.get("formatted", "")) for r in rows]

    labels = torch.tensor([to_int_label(r.get(label_key)) for r in rows], dtype=torch.long)

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=int(cfg.max_length),
        return_tensors="pt",
    )
    avg_len = _avg_len(enc["attention_mask"])

    # Probe training (estimate distortion as best achievable label CE).
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    model = MeanPoolProbe(vocab_size=vocab_size, dim=int(cfg.probe_dim), num_labels=2)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr))
    loss_fn = nn.CrossEntropyLoss()

    dl = DataLoader(_TokDataset(enc, labels), batch_size=int(cfg.batch_size), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for _ in range(int(cfg.probe_epochs)):
        model.train()
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_n = 0
        for batch in DataLoader(_TokDataset(enc, labels), batch_size=256, shuffle=False):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * int(y.shape[0])
            total_n += int(y.shape[0])
        ce = total_loss / max(1, total_n)

    # Distortion proxy: teacher-aligned CE (T=1).
    distortion = float(ce)
    if math.isnan(distortion) or math.isinf(distortion):
        distortion = 1e9

    outdir = results_root() / ds_key / "e1_interface" / tokenizer_name
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "dataset": ds_key,
            "split": cfg.split,
            "tokenizer": tokenizer_name,
            "max_samples": int(n),
            "max_length": int(cfg.max_length),
            "avg_len": avg_len,
            "distortion_ce": distortion,
            "label_key": label_key,
        },
        outdir / "metrics.json",
    )
    save_csv([(avg_len, distortion)], outdir / "rd_points.csv", header=["rate", "distortion_ce"])
    return outdir

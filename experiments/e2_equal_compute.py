from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertConfig, BertForSequenceClassification

# Import common first to bootstrap repo imports when running from a checkout.
from experiments.common import (
    auroc_binary,
    detect_label_key,
    get_text,
    infer_num_labels,
    remap_labels_to_contiguous,
    results_root,
    save_json,
    to_int_label,
)

from cit_tokenizers.corpus import load_dataset_by_name, resolve_dataset_key


@dataclass
class E2Config:
    dataset: str
    train_split: str = "train"
    val_split: str = "val"
    max_train_samples: int = 200000
    max_val_samples: int = 50000
    batch_size: int = 16
    max_length: int = 512
    total_tokens: int = 5_000_000
    lr: float = 5e-5
    weight_decay: float = 0.01
    seed: int = 42
    label_key: Optional[str] = None

    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024


class _HFClsDataset(Dataset):
    def __init__(self, enc: Dict[str, torch.Tensor], labels: torch.Tensor):
        self.enc = enc
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


def _basic_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, *, num_labels: int) -> Dict[str, float]:
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
    acc = float((y_true == y_pred).float().mean().item())
    # Macro-F1 across labels.
    f1s: List[float] = []
    for c in range(int(num_labels)):
        tp = int(((y_true == c) & (y_pred == c)).sum().item())
        fp = int(((y_true != c) & (y_pred == c)).sum().item())
        fn = int(((y_true == c) & (y_pred != c)).sum().item())
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        f1s.append(float(f1))
    macro_f1 = float(sum(f1s) / max(1, len(f1s)))
    return {"acc": acc, "macro_f1": macro_f1}


@torch.no_grad()
def _avg_len(attention_mask: torch.Tensor) -> float:
    return float(attention_mask.sum(dim=1).float().mean().item())


def run_e2(*, cfg: E2Config, tokenizer, tokenizer_name: str, model_family: str = "bert_small") -> Path:
    torch.manual_seed(cfg.seed)

    ds_key = resolve_dataset_key(cfg.dataset)
    train_ds = load_dataset_by_name(ds_key, split=cfg.train_split)
    val_ds = load_dataset_by_name(ds_key, split=cfg.val_split)

    if len(train_ds) == 0:
        raise ValueError(f"Empty dataset split: {ds_key}:{cfg.train_split}")
    if len(val_ds) == 0:
        raise ValueError(f"Empty dataset split: {ds_key}:{cfg.val_split}")

    label_key = cfg.label_key or detect_label_key(train_ds[0])

    n_tr = min(cfg.max_train_samples, len(train_ds))
    n_va = min(cfg.max_val_samples, len(val_ds))

    train_rows = [train_ds[i] for i in range(n_tr)]
    val_rows = [val_ds[i] for i in range(n_va)]
    train_texts = [get_text(r, dataset_key=ds_key) for r in train_rows]
    val_texts = [get_text(r, dataset_key=ds_key) for r in val_rows]

    raw_train = [to_int_label(r.get(label_key)) for r in train_rows]
    raw_val = [to_int_label(r.get(label_key)) for r in val_rows]
    raw_train, mapping = remap_labels_to_contiguous(raw_train)
    # Apply the same mapping to val labels.
    raw_val = [mapping.get(x, 0) for x in raw_val]
    num_labels = infer_num_labels(raw_train)

    y_train = torch.tensor(raw_train, dtype=torch.long)
    y_val = torch.tensor(raw_val, dtype=torch.long)

    enc_train = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=int(cfg.max_length),
        return_tensors="pt",
    )
    enc_val = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=int(cfg.max_length),
        return_tensors="pt",
    )

    avg_len_train = _avg_len(enc_train["attention_mask"])
    avg_len_val = _avg_len(enc_val["attention_mask"])

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    bert_cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=int(cfg.hidden_size),
        num_hidden_layers=int(cfg.num_hidden_layers),
        num_attention_heads=int(cfg.num_attention_heads),
        intermediate_size=int(cfg.intermediate_size),
        max_position_embeddings=int(cfg.max_length) + 2,
        type_vocab_size=2,
        num_labels=int(num_labels),
    )
    model = BertForSequenceClassification(bert_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dl = DataLoader(_HFClsDataset(enc_train, y_train), batch_size=int(cfg.batch_size), shuffle=True)

    tokens_seen = 0
    steps = 0
    t0 = time.time()
    model.train()

    while tokens_seen < int(cfg.total_tokens):
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            tokens_seen += int(attn.sum().item())

            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = loss_fn(logits, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            steps += 1
            if tokens_seen >= int(cfg.total_tokens):
                break
        if tokens_seen >= int(cfg.total_tokens):
            break

    train_seconds = time.time() - t0
    tokens_per_sec = float(tokens_seen / max(1e-9, train_seconds))
    peak_mem_bytes = float(torch.cuda.max_memory_allocated() if device.type == "cuda" else 0.0)

    model.eval()
    with torch.no_grad():
        val_dl = DataLoader(_HFClsDataset(enc_val, y_val), batch_size=64, shuffle=False)
        total_loss = 0.0
        total_n = 0
        preds = []
        prob_pos: list[float] = []
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            loss = loss_fn(logits, labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total_n += int(labels.shape[0])
            p = torch.softmax(logits, dim=-1).cpu()
            if int(num_labels) == 2:
                prob_pos.extend(p[:, 1].tolist())
            preds.append(torch.argmax(p, dim=-1).cpu())
        y_pred = torch.cat(preds, dim=0)
        metrics = _basic_metrics(y_val, y_pred, num_labels=int(num_labels))
        if int(num_labels) == 2:
            metrics["auroc"] = float(auroc_binary(raw_val, prob_pos))
        metrics["val_loss"] = float(total_loss / max(1, total_n))

    outdir = results_root() / ds_key / "e2_equal_compute" / model_family / tokenizer_name
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "dataset": ds_key,
            "tokenizer": tokenizer_name,
            "model_family": model_family,
            "train_split": cfg.train_split,
            "val_split": cfg.val_split,
            "label_key": label_key,
            "max_length": int(cfg.max_length),
            "batch_size": int(cfg.batch_size),
            "total_tokens_budget": int(cfg.total_tokens),
            "tokens_seen": int(tokens_seen),
            "steps": int(steps),
            "train_seconds": float(train_seconds),
            "tokens_per_sec": float(tokens_per_sec),
            "peak_mem_bytes": float(peak_mem_bytes),
            "avg_len_train": float(avg_len_train),
            "avg_len_val": float(avg_len_val),
            "metrics": metrics,
            "num_labels": int(num_labels),
            "label_mapping": mapping,
        },
        outdir / "metrics.json",
    )
    return outdir

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os, json
from ...contract import Contract, ContractConfig
from ...data import iter_text
from ...hf_artifact import save_hf_tokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def train_unigram_hygiene(
    corpus: str,
    outdir: str,
    vocab_size: int,
    contract_cfg: ContractConfig,
    fmt: str = "txt",
    text_key: str = "text",
    max_samples: int | None = None,
    min_frequency: int = 10,
    model_max_length: int = 512,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    contract = Contract(contract_cfg)

    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.UnigramTrainer(
        vocab_size=int(vocab_size),
        # tokenizers' UnigramTrainer uses "show_progress", "special_tokens"
        special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"],
    )

    def gen():
        for s in iter_text(corpus, fmt=fmt, text_key=text_key, max_samples=max_samples):
            yield contract.apply(s)

    tokenizer.train_from_iterator(gen(), trainer=trainer)

    save_hf_tokenizer(tokenizer, outdir, model_max_length=model_max_length)

    with open(os.path.join(outdir, "cit_contract.json"), "w", encoding="utf-8") as f:
        f.write(contract_cfg.to_json())

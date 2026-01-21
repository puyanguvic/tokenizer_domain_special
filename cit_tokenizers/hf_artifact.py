from __future__ import annotations
from typing import Dict, Optional, Any
import json, os

try:  # optional dependency
    from tokenizers import Tokenizer  # type: ignore
except Exception:  # pragma: no cover
    Tokenizer = object  # type: ignore

DEFAULT_SPECIAL_TOKENS = {
    "unk_token": "[UNK]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "mask_token": "[MASK]",
}

def save_hf_tokenizer(
    tokenizer: Tokenizer,
    outdir: str,
    model_max_length: int = 512,
    special_tokens: Optional[Dict[str, str]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a `tokenizers.Tokenizer` as a directory loadable by `transformers.AutoTokenizer`.

    Writes:
      - tokenizer.json
      - tokenizer_config.json
      - special_tokens_map.json
      - (optional) added_tokens.json
    """
    os.makedirs(outdir, exist_ok=True)
    tokenizer.save(os.path.join(outdir, "tokenizer.json"))

    st = dict(DEFAULT_SPECIAL_TOKENS)
    if special_tokens:
        st.update(special_tokens)

    tok_cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": int(model_max_length),
        "padding_side": "right",
        "truncation_side": "right",
    }
    if extra_config:
        tok_cfg.update(extra_config)

    with open(os.path.join(outdir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tok_cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(outdir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

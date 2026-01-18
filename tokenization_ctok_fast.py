from __future__ import annotations

from transformers import PreTrainedTokenizerFast


class CTokTokenizerFast(PreTrainedTokenizerFast):
    """Fast CTok tokenizer.

    Build-time produces tokenizer.json using tokenizers' WordPiece backend configured to:
      - greedy longest-match segmentation
      - continuing_subword_prefix="" (no "##"), so matching is identical at any position

    This yields CTok's deterministic left-to-right longest-match behavior with Rust speed.
    """

    vocab_files_names = {"tokenizer_file": "tokenizer.json", "meta_file": "ctok_meta.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file: str, meta_file: str | None = None, **kwargs):
        # Ensure special tokens are set (and consistent with build).
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")

        super().__init__(tokenizer_file=tokenizer_file, **kwargs)

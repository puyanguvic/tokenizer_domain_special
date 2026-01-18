from __future__ import annotations

import json
import os

from transformers import PreTrainedTokenizerFast

def _import_hygiene():
    try:
        import hygiene  # type: ignore

        return hygiene
    except ImportError:
        import importlib.util
        import sys

        here = os.path.dirname(__file__)
        path = os.path.join(here, "hygiene.py")
        if not os.path.exists(path):
            raise
        spec = importlib.util.spec_from_file_location("ctok_hygiene", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules["ctok_hygiene"] = module
        spec.loader.exec_module(module)
        return module


hygiene = _import_hygiene()


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

        meta: dict = {}
        if meta_file is not None and os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        self.hygiene_cfg = hygiene.HygieneConfig.from_dict(meta.get("hygiene", {})) if meta.get("hygiene") else hygiene.HygieneConfig(enabled=False)

    def _apply_hygiene(self, text: str) -> str:
        return hygiene.apply_hygiene(text, self.hygiene_cfg)

    def tokenize(self, text: str, **kwargs):
        return super().tokenize(self._apply_hygiene(text), **kwargs)

    def _encode_plus(self, text, text_pair=None, **kwargs):
        if isinstance(text, str):
            text = self._apply_hygiene(text)
        if isinstance(text_pair, str):
            text_pair = self._apply_hygiene(text_pair)
        return super()._encode_plus(text, text_pair=text_pair, **kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        processed = []
        for item in batch_text_or_text_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                if isinstance(a, str):
                    a = self._apply_hygiene(a)
                if isinstance(b, str):
                    b = self._apply_hygiene(b)
                processed.append((a, b))
            elif isinstance(item, str):
                processed.append(self._apply_hygiene(item))
            else:
                processed.append(item)
        return super()._batch_encode_plus(processed, **kwargs)

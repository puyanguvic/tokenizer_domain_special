from __future__ import annotations

import json
import os
import re

from transformers import PreTrainedTokenizerFast

def _build_hygiene_shim():
    class _HygienePattern:
        def __init__(self, name: str, pattern: str, replacement: str, flags: int = 0):
            self.name = name
            self.pattern = pattern
            self.replacement = replacement
            self.flags = flags

        def compile(self):
            return re.compile(self.pattern, self.flags)

    class _HygieneConfig:
        def __init__(self, enabled: bool = True, typed_tokens=None, patterns=None):
            self.enabled = enabled
            self.typed_tokens = typed_tokens or []
            self.patterns = patterns or []

        @staticmethod
        def from_dict(data: dict) -> "_HygieneConfig":
            patterns = []
            for p in data.get("patterns", []):
                patterns.append(
                    _HygienePattern(
                        name=str(p.get("name", "")),
                        pattern=str(p.get("pattern", "")),
                        replacement=str(p.get("replacement", "")),
                        flags=int(p.get("flags", 0)),
                    )
                )
            return _HygieneConfig(
                enabled=bool(data.get("enabled", True)),
                typed_tokens=list(data.get("typed_tokens", [])),
                patterns=patterns,
            )

    def _apply_hygiene(text: str, cfg: _HygieneConfig) -> str:
        if not cfg.enabled:
            return text
        out = text
        for p in cfg.patterns:
            out = p.compile().sub(p.replacement, out)
        return out

    class _Shim:
        HygieneConfig = _HygieneConfig
        apply_hygiene = staticmethod(_apply_hygiene)

    return _Shim()


def _import_hygiene():
    import importlib.util
    import sys

    here = os.path.dirname(__file__)
    path = os.path.join(here, "hygiene.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("ctok_hygiene", path)
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load hygiene module from local file")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ctok_hygiene"] = module
        spec.loader.exec_module(module)
        return module

    try:
        from . import hygiene  # type: ignore

        return hygiene
    except Exception:
        return _build_hygiene_shim()


hygiene = _import_hygiene()


def _build_pretokenize_shim():
    class _PreTokenizePattern:
        def __init__(self, name: str, pattern: str, replacement: str, flags: int = 0):
            self.name = name
            self.pattern = pattern
            self.replacement = replacement
            self.flags = flags

        def compile(self):
            return re.compile(self.pattern, self.flags)

    class _PreTokenizerConfig:
        def __init__(self, enabled: bool = True, patterns=None):
            self.enabled = enabled
            self.patterns = patterns or []

        @staticmethod
        def from_dict(data: dict) -> "_PreTokenizerConfig":
            patterns = []
            for p in data.get("patterns", []):
                patterns.append(
                    _PreTokenizePattern(
                        name=str(p.get("name", "")),
                        pattern=str(p.get("pattern", "")),
                        replacement=str(p.get("replacement", "")),
                        flags=int(p.get("flags", 0)),
                    )
                )
            return _PreTokenizerConfig(
                enabled=bool(data.get("enabled", True)),
                patterns=patterns,
            )

    def _apply_pretokenize(text: str, cfg: _PreTokenizerConfig) -> str:
        if not cfg.enabled:
            return text
        out = text
        for p in cfg.patterns:
            out = p.compile().sub(p.replacement, out)
        return out

    class _Shim:
        PreTokenizerConfig = _PreTokenizerConfig
        apply_pretokenize = staticmethod(_apply_pretokenize)

    return _Shim()


def _import_pretokenize():
    import importlib.util
    import sys

    here = os.path.dirname(__file__)
    path = os.path.join(here, "pretokenize.py")
    if os.path.exists(path):
        spec = importlib.util.spec_from_file_location("ctok_pretokenize", path)
        if spec is None or spec.loader is None:
            raise ImportError("Failed to load pretokenize module from local file")
        module = importlib.util.module_from_spec(spec)
        sys.modules["ctok_pretokenize"] = module
        spec.loader.exec_module(module)
        return module

    try:
        from . import pretokenize  # type: ignore

        return pretokenize
    except Exception:
        return _build_pretokenize_shim()


pretokenize = _import_pretokenize()


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
        if meta_file is None or not os.path.exists(meta_file):
            raise ValueError("ctok_meta.json is required to ensure pipeline symmetry.")
        tok_dir = os.path.dirname(os.path.abspath(tokenizer_file))
        meta_dir = os.path.dirname(os.path.abspath(meta_file))
        if tok_dir != meta_dir:
            raise ValueError("ctok_meta.json must live alongside tokenizer.json to keep pipeline locked.")
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("pipeline_locked") and ("hygiene" not in meta or "pretokenizer" not in meta):
            raise ValueError("Locked pipeline requires hygiene/pretokenizer config in ctok_meta.json.")
        self.lowercase: bool = bool(meta.get("lowercase", False))
        self.hygiene_cfg = hygiene.HygieneConfig.from_dict(meta.get("hygiene", {})) if meta.get("hygiene") else hygiene.HygieneConfig(enabled=False)
        self.pretok_cfg = pretokenize.PreTokenizerConfig.from_dict(meta.get("pretokenizer", {})) if meta.get("pretokenizer") else pretokenize.PreTokenizerConfig(enabled=False)

    def _apply_hygiene(self, text: str) -> str:
        return hygiene.apply_hygiene(text, self.hygiene_cfg)

    def _apply_pretokenize(self, text: str) -> str:
        return pretokenize.apply_pretokenize(text, self.pretok_cfg)

    def _apply_lowercase(self, text: str) -> str:
        return text.lower() if self.lowercase else text

    def tokenize(self, text: str, **kwargs):
        text = self._apply_lowercase(text)
        text = self._apply_pretokenize(self._apply_hygiene(text))
        return super().tokenize(text, **kwargs)

    def _encode_plus(self, text, text_pair=None, **kwargs):
        if isinstance(text, str):
            text = self._apply_lowercase(text)
            text = self._apply_pretokenize(self._apply_hygiene(text))
        if isinstance(text_pair, str):
            text_pair = self._apply_lowercase(text_pair)
            text_pair = self._apply_pretokenize(self._apply_hygiene(text_pair))
        return super()._encode_plus(text, text_pair=text_pair, **kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        processed = []
        for item in batch_text_or_text_pairs:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                a, b = item
                if isinstance(a, str):
                    a = self._apply_lowercase(a)
                    a = self._apply_pretokenize(self._apply_hygiene(a))
                if isinstance(b, str):
                    b = self._apply_lowercase(b)
                    b = self._apply_pretokenize(self._apply_hygiene(b))
                processed.append((a, b))
            elif isinstance(item, str):
                item = self._apply_lowercase(item)
                processed.append(self._apply_pretokenize(self._apply_hygiene(item)))
            else:
                processed.append(item)
        return super()._batch_encode_plus(processed, **kwargs)

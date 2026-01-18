from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

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


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"]
    token_id: Optional[int] = None

    def __init__(self):
        self.children = {}
        self.token_id = None


class TrieMatcher:
    """Left-to-right greedy longest-match over a fixed vocabulary.

    Determinism:
      - Longest match wins.
      - If the same string is inserted twice (shouldn't happen), smallest token_id wins.
    """

    def __init__(self, token_to_id: Dict[str, int]):
        self.root = TrieNode()
        self._build(token_to_id)

    def _build(self, token_to_id: Dict[str, int]) -> None:
        for tok, tid in token_to_id.items():
            node = self.root
            for ch in tok:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            if node.token_id is None or tid < node.token_id:
                node.token_id = tid

    def longest_match(self, s: str, i: int) -> Tuple[Optional[int], int]:
        node = self.root
        best_id: Optional[int] = None
        best_len = 0
        j = i
        while j < len(s):
            ch = s[j]
            nxt = node.children.get(ch)
            if nxt is None:
                break
            node = nxt
            j += 1
            if node.token_id is not None:
                best_id = node.token_id
                best_len = j - i
        return best_id, best_len


class CTokTokenizer(PreTrainedTokenizer):
    """Slow (Python) CTok tokenizer.

    This is a reference / fallback implementation. The fast path uses tokenizers' WordPiece
    backend (see tokenization_ctok_fast.py), but both share the same vocabulary semantics:
    deterministic greedy longest-match.
    """

    vocab_files_names = {"vocab_file": "vocab.json", "meta_file": "ctok_meta.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, meta_file: Optional[str] = None, **kwargs):
        with open(vocab_file, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        if not isinstance(token_to_id, dict):
            raise ValueError("vocab.json must be a dict {token: id}.")

        meta: Dict[str, Any] = {}
        if meta_file is not None and os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        pad = kwargs.get("pad_token", "[PAD]")
        unk = kwargs.get("unk_token", "[UNK]")
        cls = kwargs.get("cls_token", "[CLS]")
        sep = kwargs.get("sep_token", "[SEP]")
        mask = kwargs.get("mask_token", "[MASK]")

        super().__init__(
            pad_token=pad,
            unk_token=unk,
            cls_token=cls,
            sep_token=sep,
            mask_token=mask,
            **kwargs,
        )

        self.token_to_id: Dict[str, int] = {str(k): int(v) for k, v in token_to_id.items()}
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

        self.match_special_tokens: bool = bool(meta.get("match_special_tokens", False))
        self.hygiene_cfg = hygiene.HygieneConfig.from_dict(meta.get("hygiene", {})) if meta.get("hygiene") else hygiene.HygieneConfig(enabled=False)

        # sanity: specials must exist
        for st in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
            if st not in self.token_to_id:
                raise ValueError(f"Special token {st} missing from vocab.json")

        self._pad_id = self.token_to_id[self.pad_token]
        self._unk_id = self.token_to_id[self.unk_token]
        self._cls_id = self.token_to_id[self.cls_token]
        self._sep_id = self.token_to_id[self.sep_token]
        self._mask_id = self.token_to_id[self.mask_token]

        match_vocab = dict(self.token_to_id)
        if not self.match_special_tokens:
            for st in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
                match_vocab.pop(st, None)

        self.matcher = TrieMatcher(match_vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        text = hygiene.apply_hygiene(text, self.hygiene_cfg)
        toks: List[str] = []
        i = 0
        while i < len(text):
            tid, ln = self.matcher.longest_match(text, i)
            if tid is None or ln <= 0:
                ch = text[i : i + 1]
                toks.append(ch if ch in self.token_to_id else self.unk_token)
                i += 1
            else:
                toks.append(self.id_to_token[tid])
                i += ln
        return toks

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, self._unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join([t for t in tokens if t not in self.all_special_tokens])

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self._cls_id] + token_ids_0 + [self._sep_id]
        return [self._cls_id] + token_ids_0 + [self._sep_id] + token_ids_1 + [self._sep_id]

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False) -> List[int]:
        special = {self._cls_id, self._sep_id, self._pad_id, self._mask_id}
        if already_has_special_tokens:
            return [1 if t in special else 0 for t in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        path = os.path.join(save_directory, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=True, indent=2)
        return (path,)

    # Optional helper for diagnostics
    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        text = hygiene.apply_hygiene(text, self.hygiene_cfg)
        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []
        i = 0
        while i < len(text):
            tid, ln = self.matcher.longest_match(text, i)
            if tid is None or ln <= 0:
                ch = text[i : i + 1]
                ids.append(self._convert_token_to_id(ch))
                offsets.append((i, i + 1))
                i += 1
            else:
                ids.append(tid)
                offsets.append((i, i + ln))
                i += ln
        return ids, offsets

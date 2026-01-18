from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer


# =========================
# Byte-string representation
# =========================

def to_bytestr(text: str) -> str:
    """unicode -> UTF-8 bytes -> latin-1 str (1 char == 1 byte)."""
    return text.encode("utf-8").decode("latin-1")


def from_bytestr(bs: str) -> str:
    """latin-1 str (1 char == 1 byte) -> UTF-8 decode (lossy for invalid sequences)."""
    return bs.encode("latin-1").decode("utf-8", errors="replace")


# =========================
# Trie matcher: left-to-right longest match
# Deterministic tie-break: smaller token_id on duplicate strings
# =========================

@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"]
    token_id: Optional[int] = None

    def __init__(self) -> None:
        self.children = {}
        self.token_id = None


class TrieMatcher:
    def __init__(self, token_to_id: Dict[str, int]) -> None:
        self.root = TrieNode()
        self._build(token_to_id)

    def _build(self, token_to_id: Dict[str, int]) -> None:
        for tok, tid in token_to_id.items():
            node = self.root
            for ch in tok:
                node = node.children.setdefault(ch, TrieNode())
            if node.token_id is None or tid < node.token_id:
                node.token_id = tid

    def longest_match(self, s: str, i: int) -> Tuple[Optional[int], int]:
        """Return (token_id, length) of the longest match starting at i."""
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


# =========================
# CTokTokenizer (slow, but AutoTokenizer-loadable)
# =========================

class CTokTokenizer(PreTrainedTokenizer):
    """CTok runtime: deterministic left-to-right longest-match tokenizer.

    This class is intentionally minimal and "slow" (Python). You can later swap
    out the matcher with a Rust extension without changing this interface.

    Expected files in a pretrained directory:
      - vocab.json
      - ctok_meta.json
      - tokenizer_config.json
      - special_tokens_map.json
      - (this file) tokenization_ctok.py

    Load with:
      AutoTokenizer.from_pretrained(path_or_repo, trust_remote_code=True)
    """

    vocab_files_names = {"vocab_file": "vocab.json", "meta_file": "ctok_meta.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        meta_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Load vocab (token -> id)
        with open(vocab_file, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        if not isinstance(token_to_id, dict):
            raise ValueError("vocab.json must be a dict: {token: id}")

        # Load meta (optional)
        meta: Dict[str, Any] = {}
        if meta_file is not None and os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # Special tokens (defaults)
        pad_token = kwargs.pop("pad_token", "[PAD]")
        unk_token = kwargs.pop("unk_token", "[UNK]")
        cls_token = kwargs.pop("cls_token", "[CLS]")
        sep_token = kwargs.pop("sep_token", "[SEP]")
        mask_token = kwargs.pop("mask_token", "[MASK]")

        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            **kwargs,
        )

        self.token_to_id: Dict[str, int] = {str(k): int(v) for k, v in token_to_id.items()}
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

        # Meta knobs
        self.meta: Dict[str, Any] = meta
        self.use_bytestr: bool = bool(meta.get("use_bytestr", True))
        self.match_special_tokens: bool = bool(meta.get("match_special_tokens", False))

        # Verify specials exist
        for st in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
            if st not in self.token_to_id:
                raise ValueError(f"Special token {st} missing from vocab.json")

        self._pad_id = self.token_to_id[self.pad_token]
        self._unk_id = self.token_to_id[self.unk_token]
        self._cls_id = self.token_to_id[self.cls_token]
        self._sep_id = self.token_to_id[self.sep_token]
        self._mask_id = self.token_to_id[self.mask_token]

        # Build matcher excluding specials (recommended)
        match_vocab = dict(self.token_to_id)
        if not self.match_special_tokens:
            for st in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
                match_vocab.pop(st, None)
        self.matcher = TrieMatcher(match_vocab)

    # ---------- required overrides ----------
    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _tokenize(self, text: str) -> List[str]:
        s = to_bytestr(text) if self.use_bytestr else text
        toks: List[str] = []

        i = 0
        while i < len(s):
            tid, ln = self.matcher.longest_match(s, i)
            if tid is None or ln <= 0:
                unit = s[i : i + 1]
                toks.append(unit if unit in self.token_to_id else self.unk_token)
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
        # Mainly for debugging; CTok does not require exact invertibility.
        s = "".join([t for t in tokens if t not in self.all_special_tokens])
        return from_bytestr(s) if self.use_bytestr else s

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self._cls_id] + token_ids_0 + [self._sep_id]
        return [self._cls_id] + token_ids_0 + [self._sep_id] + token_ids_1 + [self._sep_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        specials = {self._cls_id, self._sep_id, self._pad_id, self._mask_id}
        if already_has_special_tokens:
            return [1 if t in specials else 0 for t in token_ids_0]
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)

        vocab_name = (filename_prefix + "-" if filename_prefix else "") + "vocab.json"
        vocab_path = os.path.join(save_directory, vocab_name)
        with open(vocab_path, "w", encoding="utf-8") as f:
            # ensure_ascii=True keeps control bytes safely escaped
            json.dump(self.token_to_id, f, ensure_ascii=True, indent=2)

        # Also persist meta if present
        meta_name = (filename_prefix + "-" if filename_prefix else "") + "ctok_meta.json"
        meta_path = os.path.join(save_directory, meta_name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta or {"use_bytestr": True}, f, ensure_ascii=True, indent=2)

        return (vocab_path, meta_path)

    # ---------- optional: offsets (byte offsets in internal byte-string) ----------
    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        s = to_bytestr(text) if self.use_bytestr else text
        ids: List[int] = []
        offsets: List[Tuple[int, int]] = []

        i = 0
        while i < len(s):
            tid, ln = self.matcher.longest_match(s, i)
            if tid is None or ln <= 0:
                unit = s[i : i + 1]
                ids.append(self._convert_token_to_id(unit))
                offsets.append((i, i + 1))
                i += 1
            else:
                ids.append(tid)
                offsets.append((i, i + ln))
                i += ln

        return ids, offsets

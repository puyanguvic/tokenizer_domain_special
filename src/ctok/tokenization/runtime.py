"""Deterministic runtime tokenization."""

from __future__ import annotations

from dataclasses import dataclass

from ctok.tokenization.vocab import Vocabulary


@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"]
    terminal_id: int | None = None

    def __init__(self) -> None:
        self.children = {}
        self.terminal_id = None


class TokenizerRuntime:
    """Left-to-right, longest-match tokenizer with deterministic behavior."""

    def __init__(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        self._token_to_id = vocab.token_to_id()
        self._unk_id = vocab.special_ids().get("unk") if vocab.special_tokens else None
        self._root = _TrieNode()
        for token_id, token in enumerate(vocab.tokens):
            node = self._root
            for ch in token:
                node = node.children.setdefault(ch, _TrieNode())
            node.terminal_id = token_id

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        idx = 0
        while idx < len(text):
            node = self._root
            last_id: int | None = None
            last_pos = idx
            cursor = idx
            while cursor < len(text):
                ch = text[cursor]
                if ch not in node.children:
                    break
                node = node.children[ch]
                cursor += 1
                if node.terminal_id is not None:
                    last_id = node.terminal_id
                    last_pos = cursor
            if last_id is None:
                char = text[idx]
                token_id = self._token_to_id.get(char)
                if token_id is None:
                    if self._unk_id is not None:
                        ids.append(self._unk_id)
                        idx += 1
                        continue
                    raise ValueError(f"No token found for character {char!r}; add base charset tokens.")
                ids.append(token_id)
                idx += 1
            else:
                ids.append(last_id)
                idx = last_pos
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab.token_for(token_id) for token_id in ids)

    def tokenize(self, text: str) -> list[str]:
        return [self._vocab.token_for(token_id) for token_id in self.encode(text)]


class BoundaryAwareTokenizerRuntime(TokenizerRuntime):
    """Tokenizer that prevents tokens from crossing boundary characters."""

    def __init__(self, vocab: Vocabulary, boundary_chars: set[str]) -> None:
        super().__init__(vocab)
        if not boundary_chars:
            raise ValueError("boundary_chars must be non-empty for boundary-aware tokenization.")
        self._boundary_chars = set(boundary_chars)

    @property
    def boundary_chars(self) -> set[str]:
        return set(self._boundary_chars)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        start = 0
        for idx, ch in enumerate(text):
            if ch not in self._boundary_chars:
                continue
            if start < idx:
                ids.extend(super().encode(text[start:idx]))
            token_id = self._token_to_id.get(ch)
            if token_id is None:
                if self._unk_id is not None:
                    ids.append(self._unk_id)
                    start = idx + 1
                    continue
                raise ValueError(f"Boundary character {ch!r} is missing from the vocabulary.")
            ids.append(token_id)
            start = idx + 1
        if start < len(text):
            ids.extend(super().encode(text[start:]))
        return ids

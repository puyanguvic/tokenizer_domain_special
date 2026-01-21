from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class CompiledMatcher:
    """Compiled greedy longest-match matcher.

    This is a small, dependency-free compiler/runtime that provides deterministic
    left-to-right longest-match tokenization with fixed tie-breaking.

    Complexity: O(n * L_max) where L_max is the maximum token length in the
    vocabulary (typically <= 24/32 in our intended regime). Because L_max is a
    *build-time constant* stored in the artifact, runtime is predictable and
    effectively linear in the input length.

    Representation:
      - trie_next[state][char] -> next_state
      - accept[state] -> token_id (the token that ends at this state)
    """

    trie_next: List[Dict[str, int]]
    accept: List[int]
    max_token_len: int
    tie_break: str = "longer_then_lower_id"  # reserved for future

    def to_json(self) -> str:
        return json.dumps(
            {
                "trie_next": self.trie_next,
                "accept": self.accept,
                "max_token_len": self.max_token_len,
                "tie_break": self.tie_break,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(s: str) -> "CompiledMatcher":
        obj = json.loads(s)
        return CompiledMatcher(
            trie_next=[{k: int(v) for k, v in d.items()} for d in obj["trie_next"]],
            accept=[int(x) for x in obj["accept"]],
            max_token_len=int(obj["max_token_len"]),
            tie_break=str(obj.get("tie_break", "longer_then_lower_id")),
        )

    def encode_greedy(self, text: str, unk_id: int, vocab_id_for_char: Optional[Dict[str, int]] = None) -> List[int]:
        """Greedy longest-match tokenization.

        Args:
            text: input string (already passed through contract/hygiene).
            unk_id: id for [UNK].
            vocab_id_for_char: optional mapping used to fall back to single
              character tokens if present; otherwise use unk_id.
        """

        out: List[int] = []
        n = len(text)
        i = 0
        while i < n:
            state = 0
            best_id = -1
            best_len = 0
            # bounded walk
            limit = min(n, i + self.max_token_len)
            j = i
            while j < limit:
                ch = text[j]
                nxt = self.trie_next[state].get(ch)
                if nxt is None:
                    break
                state = nxt
                tok_id = self.accept[state]
                if tok_id >= 0:
                    best_id = tok_id
                    best_len = (j - i) + 1
                j += 1

            if best_id >= 0:
                out.append(best_id)
                i += best_len
                continue

            # fallback: single character token if present, else UNK
            if vocab_id_for_char is not None:
                cid = vocab_id_for_char.get(text[i])
                out.append(cid if cid is not None else unk_id)
            else:
                out.append(unk_id)
            i += 1
        return out


def compile_trie(tokens: Iterable[Tuple[str, int]]) -> CompiledMatcher:
    """Compile a trie matcher from (token_string, token_id).

    Notes:
      - Special tokens should be excluded (except typed symbols if they are
        emitted by the contract/hygiene).
      - If multiple tokens map to the same string, the lower id wins.
    """

    trie_next: List[Dict[str, int]] = [{}]
    accept: List[int] = [-1]
    max_len = 1

    for s, tid in tokens:
        if not s:
            continue
        max_len = max(max_len, len(s))
        state = 0
        for ch in s:
            nxt = trie_next[state].get(ch)
            if nxt is None:
                nxt = len(trie_next)
                trie_next[state][ch] = nxt
                trie_next.append({})
                accept.append(-1)
            state = nxt
        # accept state
        prev = accept[state]
        if prev < 0 or tid < prev:
            accept[state] = tid

    return CompiledMatcher(trie_next=trie_next, accept=accept, max_token_len=max_len)

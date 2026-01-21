"""Transformers-compatible tokenizer for CIT.

This module lets a CIT artifact directory be loaded by Hugging Face:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/path/to/cit_artifact", trust_remote_code=True)
```

It intentionally implements the minimal API surface needed for encoder-only
classification and benchmarking.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer

from cit_tokenizers.cit.runtime import CITArtifact, CITRuntime


class CITTokenizer(PreTrainedTokenizer):
    """A deterministic tokenizer backed by a compiled CIT matcher."""

    vocab_files_names = {"vocab_file": "cit_artifact.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        model_max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        with open(vocab_file, "r", encoding="utf-8") as f:
            self._artifact = CITArtifact.loads(f.read())
        self._runtime = CITRuntime(self._artifact)
        self.vocab = self._artifact.vocab
        self.ids_to_tokens = {i: t for i, t in enumerate(self._artifact.id_to_token)}
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("mask_token", "[MASK]")
        super().__init__(model_max_length=model_max_length, **kwargs)

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:  # type: ignore[override]
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:  # type: ignore[override]
        ids = self._runtime.encode(text)
        return [self.ids_to_tokens.get(i, self.unk_token) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:  # type: ignore[override]
        return self.vocab.get(token, self.vocab.get(self.unk_token, 1))

    def _convert_id_to_token(self, index: int) -> str:  # type: ignore[override]
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:  # type: ignore[override]
        # CIT does not guarantee invertibility; best-effort join for debugging.
        return "".join(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return [0] * (len(token_ids_0) + 2)
        return [0] * (len(token_ids_0) + 2) + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str, ...]:  # type: ignore[override]
        os.makedirs(save_directory, exist_ok=True)
        artifact_path = os.path.join(save_directory, (filename_prefix or "") + "cit_artifact.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            f.write(self._artifact.dumps())
        return (artifact_path,)

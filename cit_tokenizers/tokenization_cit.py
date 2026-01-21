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

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer

from .cit.runtime import CITArtifact


class CITTokenizer(PreTrainedTokenizer):
    """A deterministic tokenizer backed by a compiled CIT matcher."""

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        contract_file: str,
        matcher_file: str,
        model_max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        self._artifact = CITArtifact.load(vocab_file=vocab_file, contract_file=contract_file, matcher_file=matcher_file)
        self.vocab = self._artifact.vocab
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        super().__init__(
            model_max_length=model_max_length,
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:  # type: ignore[override]
        return dict(self.vocab)

    def _tokenize(self, text: str) -> List[str]:  # type: ignore[override]
        return self._artifact.tokenize(text)

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
        vocab_path = os.path.join(save_directory, (filename_prefix or "") + "vocab.json")
        contract_path = os.path.join(save_directory, (filename_prefix or "") + "contract.json")
        matcher_path = os.path.join(save_directory, (filename_prefix or "") + "matcher.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(self._artifact.contract_cfg.to_dict(), f, ensure_ascii=False, indent=2)
        with open(matcher_path, "w", encoding="utf-8") as f:
            json.dump(self._artifact.matcher.to_dict(), f, ensure_ascii=False)
        return (vocab_path, contract_path, matcher_path)

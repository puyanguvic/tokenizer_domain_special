# CTok Tokenizer Artifact

This directory is directly loadable by Transformers.

## Load
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('./THIS_DIR', trust_remote_code=True)
```

## Files
- vocab.json: token->id (includes specials + 256 bytes + induced tokens)
- ctok_meta.json: build metadata (boundaries etc.)
- tokenizer_config.json + special_tokens_map.json: loading config
- tokenization_ctok.py: runtime implementation

## Build objective
This artifact was built with a compression-first surrogate (lambda_sem=0).

To match the paper more closely, enable label-aware semantic scoring or plug in a probe.

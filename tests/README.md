# CTok (No `trust_remote_code`) — Unigram + Rust `tokenizers` pipeline

This package produces a **standard** Hugging Face tokenizer artifact that can be loaded with:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/path/to/ctok_artifact")
print(type(tok.backend_tokenizer))
print(tok.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))
```

Key properties:

* **No custom Python tokenizer class** is needed at runtime.
* `tokenizer.json` contains **normalizer**, **pre_tokenizer**, **Unigram model**, and **post_processor**.
* The “hygiene + pretokenize” rules are **compiled into Rust** as much as possible.
* Numeric bucketing is implemented fully in Rust normalizer using an allowlist-protect → bucketize → restore approach.

## Files

* `build_ctok_unigram_artifact.py`: Build an artifact from a corpus (txt/tsv/jsonl/parquet).
* `demo_no_remote_code.py`: Smoke-test for HF loading and backend components.

## Output artifact layout

The output directory contains:

* `tokenizer.json`
* `tokenizer_config.json`
* `special_tokens_map.json`
* `ctok_meta.json` (metadata for reproducibility)

Because the artifact is a standard `tokenizer.json`, **no** `tokenization_*.py` is shipped and
`trust_remote_code=True` is **not** required.

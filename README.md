# CTok Fast Tokenizer (final)

This package builds a **fast** CTok tokenizer artifact from a corpus and loads it via Transformers.

## Why this fixes your error
`tokenizer_config.json` uses `auto_map` as a **2-tuple/list**: `[slow_ref, fast_ref]`.
Transformers' `AutoTokenizer` expects this format for dynamic tokenizers on the Hub. See HF discussion. 

## Quickstart

```bash
pip install -U transformers tokenizers

bash demo_build_and_load.sh
```

## Build from Parquet

```bash
python build_ctok_from_corpus.py \
  --corpus /path/to/train.parquet \
  --format parquet \
  --text_key text \
  --label_key label \
  --outdir ctok_http_8k \
  --vocab_size 8192 \
  --max_len 12 \
  --min_freq 50 \
  --semantic_mode mi --lambda_sem 50.0 \
  --use_ascii_base \
  --emit_code
```

Load:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./ctok_http_8k", trust_remote_code=True)
```

Notes:
- Runtime is **fast** because the artifact includes `tokenizer.json` built with `tokenizers` (Rust).
- Segmentation is greedy longest-match (WordPiece) with `continuing_subword_prefix=""`.

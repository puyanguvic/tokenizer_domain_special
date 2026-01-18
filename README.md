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

## Experiments

Experiment 1: build CTok tokenizer from dataset (see `run_ctok_experiment.py` or `run_ctok_experiment_hydra.py`).
```bash
python run_ctok_experiment_hydra.py \
  dataset=logfit-project/HDFS_v1 \
  semantic_mode=mi \
  lambda_sem=50.0 \
  use_ascii_base=true \
  emit_code=true
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --use_ascii_base \
  --emit_code
If a corpus already exists and you want a full rebuild, add `--force_corpus`.
```

Experiment 2: fine-tune a classifier using a CTok tokenizer with a BERT model
(see `run_ctok_experiment2.py` or `run_ctok_experiment2_hydra.py`).
```bash
python run_ctok_experiment2_hydra.py \
  dataset=logfit-project/HDFS_v1 \
  tokenizer_path=tokenizers/logfit-project__HDFS_v1_train \
  model_name=google/bert_uncased_L-4_H-256_A-4
python run_ctok_experiment2.py \
  --dataset logfit-project/HDFS_v1 \
  --tokenizer_path tokenizers/logfit-project__HDFS_v1_train \
  --model_name google/bert_uncased_L-4_H-256_A-4
```
Use `value_randomization_eval=true` to run a simple robustness check that randomizes
high-cardinality values in the eval split.

CTok hygiene (default on) replaces high-cardinality values with typed tokens (e.g., `<IPV4>`, `<UUID>`) at build
and runtime, and logs hygiene metrics in `ctok_meta.json`. You can disable it with `--no_hygiene` and control
candidate filtering with `--no_filter_value_fragments`, `--min_doc_freq`, and `--max_doc_concentration`.

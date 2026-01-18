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
  --pretokenizer generic \
  --no_boundary_ends \
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
  emit_code=true \
  pretokenizer=generic \
  no_boundary_ends=true
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --use_ascii_base \
  --emit_code \
  --pretokenizer generic \
  --no_boundary_ends
By default, this builds directly from the HF dataset (no jsonl export). If you want to export and reuse a jsonl
corpus, add `--write_corpus` (and use `--force_corpus` to rebuild an existing corpus).

Common build knobs:
- `--max_chars_per_sample` limits per-sample scan length (default 4096)
- `--boundaries` controls token boundary characters
- `--no_boundary_ends` disables adding boundary-prefixed/suffixed tokens
- `--max_base_chars` caps the base character set size
- `--pretokenizer generic` enables structure-aware pre-tokenization (HTML/HTTP-friendly)
- `--junk_penalty_beta` penalizes high-entropy/value-like fragments (default 0.5)

Performance tips for large corpora:
- Prefer direct HF dataset builds (default) to avoid Arrow->jsonl conversion.
- Set `--num_workers 0` to auto-use all cores (used by candidate collection and doc stats).
- Use `--max_samples` or `--sample_ratio` for quick iterations.
- Tighten `--max_chars_per_sample` to reduce scanning cost on long log lines.

Common command patterns:
```bash
# Fast smoke test (small sample, fewer workers)
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --max_samples 50000 \
  --num_workers 4 \
  --pretokenizer generic \
  --use_ascii_base

# Full build with MI and all cores
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --semantic_mode mi \
  --lambda_sem 50.0 \
  --num_workers 0 \
  --pretokenizer generic \
  --use_ascii_base

# Export and reuse jsonl corpus on disk
python run_ctok_experiment.py \
  --dataset logfit-project/HDFS_v1 \
  --write_corpus \
  --force_corpus
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

CTok hygiene (default on) replaces high-cardinality values with typed tokens (e.g., `__IPV4__`, `__UUID__`) at build
and runtime, and logs hygiene metrics in `ctok_meta.json`. You can disable it with `--no_hygiene` and control
candidate filtering with `--no_filter_value_fragments`, `--min_doc_freq`, and `--max_doc_concentration`. The
default filter also treats long pure-numeric strings as value fragments to keep vocab focused on structure.

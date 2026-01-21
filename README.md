# CIT Tokenizers

This package provides:

- **CIT**: Controlled Interface Tokenization (contract-constrained tokenizer interface).
- **Parallel, hygiene-aligned baselines**:
  - `BPE+Hygiene` (`cit-train-bpeh`)
  - `WordPiece+Hygiene` (`cit-train-wordpieceh`)
  - `Unigram+Hygiene` (`cit-train-unigramh`)

All methods share the **same deterministic interface contract** (JSON role-aware serialization + typed hygiene + integrity rules),
and differ only in **vocabulary induction / segmentation model**.

## Install (editable)

```bash
pip install -e .
```

## Quickstart

1) Train a baseline WordPiece+Hygiene tokenizer:

```bash
cit-train-wordpieceh --corpus path/to/corpus.txt --outdir tokenizers/wp_hygiene_demo --vocab-size 8192
```

2) Load with Hugging Face:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizers/wp_hygiene_demo", use_fast=True)
print(tok.tokenize("GET /index.php?id=123&x=1"))
```

## Running experiments

This repo supports a simple experiment workflow.
Downloaded datasets and cache live under `datasets/` (e.g., `datasets/hf_cache`), and tokenizer outputs should be placed in `tokenizers/` with a descriptive name.

One-command experiment script (exports + trains):

```bash
python experiments/run_experiment.py --dataset hdfs --algorithm cit --name hdfs_cit_tokenizer
```

For structured HTTP/WAF logs, add `--cit-preset http` to enable symbol-aware boundaries and character fallback.

Manual steps (export a Hugging Face dataset to JSONL, then train):

```bash
python - <<'PY'
from cit_tokenizers.corpus import export_dataset_corpus
path = export_dataset_corpus("hdfs", split="train", cache_dir="datasets/hf_cache")
print(path)
PY

cit-train-cit \
  --corpus datasets/corpus/logfit-project__HDFS_v1_train.jsonl \
  --format jsonl \
  --text-key text \
  --outdir tokenizers/hdfs_cit_tokenizer \
  --vocab-size 8192

cit-train-wordpieceh \
  --corpus datasets/corpus/logfit-project__HDFS_v1_train.jsonl \
  --format jsonl \
  --text-key text \
  --outdir tokenizers/hdfs_wordpiece_tokenizer \
  --vocab-size 8192
```

## CIT core code map (paper ↔ code)

If you are looking for the *core* CIT implementation (as described in the paper):

* **Interface contract (serialization + typed hygiene + integrity)**
  - `cit_tokenizers/contract.py` (Contract + config)
  - `cit_tokenizers/hygiene.py` (typed tokens, numeric bucketing, priority order)
* **Distortion-aware vocabulary induction (greedy gain–distortion)**
  - `cit_tokenizers/cit/trainer.py` (`CITTrainer`, candidate extraction, greedy selection)
  - Note: `label_probe` mode is included as an optional stub; you can keep `distortion_mode="none"` if you are only building artifacts.
* **Deterministic compilation + runtime matcher (longest-match policy)**
  - `cit_tokenizers/cit/compiler.py` (trie compiler)
  - `cit_tokenizers/cit/runtime.py` (artifact loader + greedy tokenizer runtime)
* **Hugging Face loading (AutoTokenizer)**
  - `cit_tokenizers/tokenization_cit.py` (Python tokenizer class)

Parallel baselines live under `cit_tokenizers/baselines/` and are intentionally kept separate so that ablations (e.g., **BPE+Hygiene**) do not interfere with CIT's core code.

## Corpus formats

The trainers accept `.txt`, `.jsonl`, and `.parquet` (optional dependency: `pyarrow`).
Use `--format` / `--text-key` to specify the field containing the raw string.

## Contract

The contract is applied **identically at build time and runtime**:

`X -> serialize/typed hygiene -> X' -> subword segmentation -> tokens`

See `cit_tokenizers/contract.py`.



## Parallel Baselines (for Ablations)

This repo intentionally keeps **CIT** and all ablation baselines **side-by-side** (no code-path entanglement):

- `cit_tokenizers/cit/` — CIT (Controlled Interface Tokenization)
- `cit_tokenizers/baselines/bpe_hygiene/` — **BPE + Hygiene** (same hygiene/serialization, frequency-driven BPE)
- `cit_tokenizers/baselines/wordpiece_hygiene/` — **WordPiece + Hygiene**
- `cit_tokenizers/baselines/unigram_hygiene/` — **Unigram + Hygiene**

Each baseline exposes a `*Trainer` API and a CLI entrypoint, and each produces a Hugging Face–loadable tokenizer
artifact in `outdir/` (including `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`).

### Quick commands

```bash
# 1) Train CIT
cit-train-cit --corpus data.txt --outdir tokenizers/cit_demo --vocab-size 8192 --format txt

# 2) Train baseline: BPE + Hygiene
cit-train-bpeh --corpus data.txt --outdir tokenizers/bpe_hyg_demo --vocab-size 8192 --format txt

# 3) Train baseline: WordPiece + Hygiene
cit-train-wordpieceh --corpus data.txt --outdir tokenizers/wp_hyg_demo --vocab-size 8192 --format txt

# 4) Train baseline: Unigram + Hygiene
cit-train-unigramh --corpus data.txt --outdir tokenizers/uni_hyg_demo --vocab-size 8192 --format txt
```

### Loading with Transformers

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizers/cit_demo")
print(tok.tokenize("GET /index.html?x=1 HTTP/1.1"))
```

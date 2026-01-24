# CIT Tokenizers

This package provides:

- **CIT**: Controlled Interface Tokenization (contract-constrained tokenizer interface).
- **Parallel, hygiene-aligned baselines**:
  - `BPE+Hygiene` (via `cit train bpeh`)
  - `WordPiece+Hygiene` (via `cit train wordpieceh`)
  - `Unigram+Hygiene` (via `cit train unigramh`)

All methods share the **same deterministic interface contract** (JSON role-aware serialization + typed hygiene + integrity rules),
and differ only in **vocabulary induction / segmentation model**.

## Install (editable)

```bash
pip install -e .
```

## Quickstart

1) Train a CIT tokenizer:

```bash
cit train cit --corpus path/to/corpus.txt --outdir tokenizers/cit_demo --vocab-size 8192 --preset waf
```

2) Load with Transformers:

```python
from cit_tokenizers import CITTokenizer
tok = CITTokenizer.from_pretrained("tokenizers/cit_demo")
print(tok.tokenize("GET /index.html?x=1 HTTP/1.1"))
```

Note: `CITTokenizer` loads data-only artifacts without `trust_remote_code`; `AutoTokenizer.from_pretrained(...)` is not supported.

## Train a CIT tokenizer

Example (WAF corpus):

```bash
cit train cit \
  --preset waf \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --outdir tokenizers/waf_cit_tokenizer \
  --vocab-size 8192 \
  --min-freq 10 \
  --seed 0
```

Training cleans noisy blobs (e.g., base64/hex) by default; disable with `--no-clean`.
For `--preset http|waf`, structured HTTP parsing is enabled by default (see `--structured-input`).

Explicitly enable structured parsing:

```bash
cit train cit \
  --preset waf \
  --structured-input http \
  --structured-max-len 4096 \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --outdir tokenizers/waf_cit_tokenizer_struct \
  --vocab-size 8192 \
  --min-freq 10 \
  --seed 0
```

Structured HTTP preprocessing:
- Emits tags like `<METHOD>`, `<URL>`, `<HDR>`, `<BODY>` and splits query/cookie/body values into key/value tokens.
- High-entropy values are replaced with placeholders such as `<HEX>`, `<B64>`, `<HASH_32>`, `<UUID>`, `<URLENC>`, `<BYTES>`, and numeric buckets.
- Cookie keys keep readable prefixes while high-entropy suffixes are normalized (e.g., `comment_author_<hash>` -> `comment_author_<HEX>`).
- Fallback on parse failure: `<RAW> ... <TRUNC>` with a hard length cap.

Optional: pre-clean a corpus and write it back out:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --out datasets/corpus/puyang2025__waf_data_v2_train_clean.jsonl
```

Optional: pre-clean + structured HTTP parsing:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --structured-input http \
  --out datasets/corpus/puyang2025__waf_data_v2_train_struct.jsonl
```

Cookie key normalization demo:

```bash
cit clean \
  --format jsonl \
  --corpus datasets/corpus/puyang2025__waf_data_v2_train.jsonl \
  --text-key text \
  --structured-input http \
  --out datasets/corpus/puyang2025__waf_data_v2_train_struct_cookiekeys.jsonl
```

Common arguments:

- `--corpus`: Path to corpus file (required).
- `--format`: Corpus format: `txt`, `jsonl`, or `parquet` (default: `txt`).
- `--text-key`: Text field for `jsonl`/`parquet` (default: `text`).
- `--max-samples`: Optional cap on samples read from the corpus.
- `--outdir`: Output artifact directory (required).
- `--vocab-size`: Target vocabulary size (default: `8192`).
- `--min-freq`: Minimum token frequency (default: `10`).
- `--preset`: Domain preset: `default`, `http`, or `waf` (default: `default`).
- `--seed`: RNG seed (default: `0`).
- `--lambda-rd`: Regularization strength for candidate scoring (default: `0.0`).
- `--distortion-mode`: Distortion proxy: `none` or `boundary_penalty` (default: `none`).
- `--boundary-penalty`: Penalty scalar for `boundary_penalty` mode (default: `1.0`).
- `--config`: Path to a `CITBuildConfig` JSON file (overrides per-flag trainer/contract).

Contract options:

- `--contract-json`: Path to a `ContractConfig` JSON file.
- `--no-json-serialization`: Disable JSON role-aware serialization.
- `--no-typed-hygiene`: Disable typed hygiene.
- `--no-numeric-buckets`: Disable numeric bucket tokens.
- `--long-num-min-digits`: Minimum digit count for long-number buckets (default: `6`).
- `--structured-input`: Structured preprocessing mode: `none`, `http`, or `waf` (default: auto for `http|waf` preset).
- `--structured-max-len`: Hard cap for structured parsing input length (default: `4096`).

Global:

- `--log-level`: Logging level (e.g., `INFO`, `DEBUG`).

## Package layout (v0.7.0)

The library is structured as a small, production-oriented SDK:

- `cit_tokenizers.interface`: contract, value-aware hygiene, serialization
- `cit_tokenizers.cit`: CIT trainer, compiler, runtime matcher
- `cit_tokenizers.artifacts`: artifact IO helpers
- `cit_tokenizers.io`: corpus/dataset IO

## CLI recipes

### Export a CIT artifact as a HuggingFace-style folder (no remote code)

```bash
cit export-hf \
  --artifact-dir tokenizers/cit_demo \
  --outdir tokenizers/cit_demo_hf \
  --model-max-length 512
```

Load:

```python
from cit_tokenizers import CITTokenizer
tok = CITTokenizer.from_pretrained("tokenizers/cit_demo_hf")
```

### Pull a HuggingFace dataset split to parquet

Requires optional dependencies:

```bash
pip install -e '.[hf]'
```

Then:

```bash
cit dataset pull \
  --dataset imdb \
  --split train \
  --text-key text \
  --label-key label \
  --out data/imdb_train.parquet \
  --max-samples 10000 \
  --shuffle
```

## Development

Install (editable) with dev tooling:

```bash
pip install -e '.[dev]'
pre-commit install
```

Run checks locally:

```bash
ruff check .
ruff format .
pytest
```

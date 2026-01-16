# Controlled Tokenization (ctok)

Controlled Tokenization (ctok) is a deterministic, grammar-guided tokenizer system for
structured and semi-structured detection workloads. The repository provides a clean
runtime tokenizer, a configurable vocabulary induction pipeline, and scaffolding for
experiments and diagnostics aligned with the paper.

## Quickstart

### Prerequisites

- Python 3.10+ available on PATH.
- `uv` installed (see https://github.com/astral-sh/uv).

### Setup

```bash
./setup.sh
source .venv/bin/activate
```

If you want a different venv path or extras:

```bash
VENV_DIR=/tmp/ctok-venv ./setup.sh
./setup.sh --extras dev,hf,transformers
./setup.sh --no-extras
```

### Build and run

```bash
ctok build --corpus data/raw/example.txt --output artifacts/tokenizers/ctok_v1 --config configs/tokenizers/ctok.yaml
ctok encode --artifact artifacts/tokenizers/ctok_v1 --input data/raw/example.txt --format ids
ctok eval --artifact artifacts/tokenizers/ctok_v1 --corpus data/raw/example.txt
```

Optional: add a label-aligned file to enable the label-entropy distortion proxy:

```bash
ctok build --corpus data/raw/example.txt --labels data/raw/example.labels \
  --output artifacts/tokenizers/ctok_v1 --config configs/tokenizers/ctok.yaml
```

For boundary-respecting candidates (boundary-healing baseline), add:

```bash
ctok build --corpus data/raw/example.txt --output artifacts/tokenizers/ctok_v1 \
  --config configs/tokenizers/ctok.yaml --boundary-aware
```

### Python usage

```python
from ctok.tokenization import CtokTokenizer

tokenizer = CtokTokenizer.from_pretrained("artifacts/tokenizers/ctok_v1")
batch = tokenizer(["GET /", "POST /admin"], padding=True, truncation=True, max_length=128)
print(batch["input_ids"])
```

### Transformers integration

Install the optional dependency:

```bash
pip install -e ".[transformers]"
```

```python
from ctok.tokenization.hf import CtokHFTokenizer

hf_tokenizer = CtokHFTokenizer.from_pretrained("artifacts/tokenizers/ctok_v1")
```

### Fine-tuning RoBERTa

```bash
pip install -e ".[transformers,hf]"
python scripts/finetune_roberta.py \
  --dataset-config configs/datasets/waf_http.yaml \
  --model-config configs/models/roberta_base.yaml \
  --tokenizer artifacts/tokenizers/ctok_v1 \
  --output results/runs/roberta_waf_http
```

## Layout

- `src/ctok/`: core package (tokenization runtime, induction, probes, diagnostics).
- `configs/`: configuration files for datasets, tokenizers, models, and experiments.
- `scripts/`: one-off helper scripts (data prep, export, end-to-end runs).
- `docs/`: project documentation and paper-facing notes.
- `artifacts/`, `results/`, `data/`: generated outputs (gitignored except README files).

## Notes

The current implementation focuses on deterministic runtime tokenization and a
reference induction loop that follows the gain--distortion structure in the paper.
Specialized probes, diagnostics, and model wrappers are stubbed for extension.
See `docs/experiments.md` for the experiment plan and config templates.
Hugging Face datasets are configured via `configs/datasets/*.yaml` using `hf_dataset`.
Model templates include DistilRoBERTa, RoBERTa-base, and RoBERTa-large configs.
Experiments are intended to run a dataset x model-size matrix across these three sizes.

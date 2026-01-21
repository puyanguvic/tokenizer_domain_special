# Reproducibility Notes for Parallel Baselines

This project keeps ablation baselines in `cit_tokenizers/baselines/*` to avoid structural coupling with the CIT core.

## What is held constant across CIT vs baselines

All *Hygiene* baselines share the **same deterministic interface contract** components as CIT:

- **Normalization** (minimal, structure-preserving)
- **Value-aware hygiene** (typed tokens + integrity rules)
- **Structured serialization** (e.g., JSON key/value markers) where enabled

Therefore, differences between:

- `CIT` vs `BPE+Hygiene` isolate the value of **distortion-aware induction** and **finite-state compilation** under the same constraints.
- `BPE+Hygiene` vs `WordPiece+Hygiene` vs `Unigram+Hygiene` isolate the **trainer objective / induction algorithm** under the same constraint set.

## Baseline definitions

### BPE + Hygiene
- Uses the same hygiene/serialization pass `pi`.
- Trains BPE merges using a frequency objective (no distortion signal).
- Produces a deterministic runtime segmenter and HF artifact.

### WordPiece + Hygiene
- Uses the same `pi`.
- Trains WordPiece with a standard likelihood-ish objective (implemented via `tokenizers` WordPiece trainer).
- Deterministic runtime.

### Unigram + Hygiene
- Uses the same `pi`.
- Trains a Unigram language-model tokenizer (SentencePiece-style) via `tokenizers` Unigram trainer.
- Runtime uses deterministic Viterbi (no sampling).

## Required ablation in the paper

Add a baseline named **`BPE+Hygiene`** (and optionally `WordPiece+Hygiene`, `Unigram+Hygiene`) to ensure reviewers cannot attribute gains solely to regex-like preprocessing.

Suggested naming in the paper:

- `BPE` (raw)
- `BPE+Hyg` (same hygiene/serialization as CIT)
- `WordPiece+Hyg`
- `Unigram+Hyg`
- `CIT` (ours)

## Exact command capture

For every run, log:

- the CLI command line
- the generated `outdir/config.json` (or `cit_config.json`)
- git commit SHA
- library versions (`python -m pip freeze`)


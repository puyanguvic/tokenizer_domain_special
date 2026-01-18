#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/_demo_artifact_fast"

python - <<'PY'
from pathlib import Path
p = Path("_demo_corpus.tsv")
rows = [
    (0, "GET /index.html?x=1&y=2"),
    (1, "POST /login?user=admin&pass=123"),
    (0, "GET /static/app.js"),
]
with p.open("w", encoding="utf-8") as f:
    for y, x in rows:
        f.write(f"{y}\t{x}\n")
print("Wrote", p)
PY

python "$ROOT/build_ctok.py" \
  --corpus "$ROOT/_demo_corpus.tsv" --format tsv \
  --outdir "$OUT" \
  --vocab_size 2048 --max_len 12 --min_freq 1 \
  --semantic_mode mi --lambda_sem 10.0 --mi_top_k 5000 --mi_max_samples 1000 \
  --emit_code

python - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./_demo_artifact_fast", trust_remote_code=True)
print("Loaded:", type(tok))
print(tok.tokenize("GET /index.html?x=1&y=2"))
enc = tok("GET /index.html?x=1&y=2", truncation=True, padding="max_length", max_length=32)
print(enc["input_ids"])
PY

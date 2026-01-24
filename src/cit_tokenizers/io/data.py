from __future__ import annotations
from typing import Iterable, Optional, Union, Dict, Any
import json, os

def iter_text(
    corpus_path: str,
    fmt: str = "txt",
    text_key: str = "text",
    max_samples: Optional[int] = None,
    clean: bool = False,
):
    n = 0
    fmt = fmt.lower()
    if clean:
        from .clean import clean_text
    if fmt == "txt":
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.rstrip("\n")
                if not s:
                    continue
                if clean:
                    s = clean_text(s)
                yield s
                n += 1
                if max_samples and n >= max_samples:
                    return
    elif fmt == "jsonl":
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and text_key in obj:
                    s = str(obj[text_key])
                else:
                    s = json.dumps(obj, ensure_ascii=False)
                if clean:
                    s = clean_text(s)
                yield s
                n += 1
                if max_samples and n >= max_samples:
                    return
    elif fmt == "parquet":
        try:
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("parquet support requires `pyarrow`. pip install pyarrow") from e
        table = pq.read_table(corpus_path, columns=[text_key])
        col = table.column(text_key).to_pylist()
        for s in col:
            if s is None:
                continue
            out = str(s)
            if clean:
                out = clean_text(out)
            yield out
            n += 1
            if max_samples and n >= max_samples:
                return
    else:
        raise ValueError(f"Unknown format: {fmt}. Use txt|jsonl|parquet.")

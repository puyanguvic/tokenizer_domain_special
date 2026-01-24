from __future__ import annotations

import json
import re
from typing import Optional

from ..interface.http_struct import structure_http

MIN_B64_LEN = 24
MIN_HEX_LEN = 16
MIN_NOISE_LEN = 32
NOISE_RATIO_THRESHOLD = 0.85

_B64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")
_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_WS_RE = re.compile(r"\s+")


def _is_hash_token(tok: str) -> bool:
    return 32 <= len(tok) <= 64 and _HEX_RE.fullmatch(tok) is not None


def _is_hex_token(tok: str) -> bool:
    return len(tok) >= MIN_HEX_LEN and _HEX_RE.fullmatch(tok) is not None


def _is_b64_token(tok: str) -> bool:
    if len(tok) < MIN_B64_LEN:
        return False
    if _B64_CHARS_RE.fullmatch(tok) is None:
        return False
    # Only allow padding at the end.
    if "=" in tok[:-2]:
        return False
    # Require some character diversity typical for base64 blobs.
    if re.search(r"[0-9+/=a-z]", tok) is None:
        return False
    stripped = re.sub(r"[+/=_-]", "", tok)
    if stripped and re.fullmatch(r"[A-Z]+", stripped) is not None:
        return False
    return True


def classify_noise(text: str) -> Optional[str]:
    s = text.strip()
    if not s:
        return None
    if _is_hash_token(s):
        return "<HASH>"
    if _is_hex_token(s):
        return "<HEX>"
    if _is_b64_token(s):
        return "<B64>"

    tokens = _WS_RE.split(s)
    if len(tokens) <= 1:
        return None

    total_len = sum(len(t) for t in tokens)
    if total_len < MIN_NOISE_LEN:
        return None

    counts = {"<HASH>": 0, "<HEX>": 0, "<B64>": 0}
    for t in tokens:
        if _is_hash_token(t):
            counts["<HASH>"] += len(t)
        elif _is_hex_token(t):
            counts["<HEX>"] += len(t)
        elif _is_b64_token(t):
            counts["<B64>"] += len(t)

    noise_len = sum(counts.values())
    if noise_len == 0:
        return None
    if noise_len / total_len >= NOISE_RATIO_THRESHOLD:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    return None


def clean_text(text: str) -> str:
    placeholder = classify_noise(text)
    return placeholder if placeholder is not None else text


def _normalize_for_txt(text: str) -> str:
    return text.replace("\r", " ").replace("\n", " ")


def clean_corpus(
    corpus_path: str,
    out_path: str,
    *,
    fmt: str = "txt",
    text_key: str = "text",
    max_samples: Optional[int] = None,
    out_format: Optional[str] = None,
    structured_input: Optional[str] = None,
    structured_max_len: Optional[int] = None,
) -> None:
    fmt = fmt.lower()
    out_fmt = (out_format or fmt).lower()
    n = 0
    structured = (structured_input or "none").lower() in ("http", "waf")
    max_len = int(structured_max_len) if structured_max_len is not None else 4096

    def should_stop() -> bool:
        return max_samples is not None and n >= max_samples

    if out_fmt not in {"txt", "jsonl", "parquet"}:
        raise ValueError(f"Unknown output format: {out_fmt}. Use txt|jsonl|parquet.")

    if fmt == "txt":
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f_in, open(
            out_path, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                s = line.rstrip("\n")
                if not s:
                    continue
                cleaned = clean_text(s)
                if structured:
                    cleaned = structure_http(cleaned, max_len=max_len)
                if out_fmt == "txt":
                    f_out.write(_normalize_for_txt(cleaned) + "\n")
                elif out_fmt == "jsonl":
                    f_out.write(json.dumps({text_key: cleaned}, ensure_ascii=False) + "\n")
                else:
                    raise ValueError("parquet output is not supported for txt input.")
                n += 1
                if should_stop():
                    return
        return

    if fmt == "jsonl":
        with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f_in, open(
            out_path, "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and text_key in obj:
                    cleaned = clean_text(str(obj[text_key]))
                    if structured:
                        cleaned = structure_http(cleaned, max_len=max_len)
                    obj[text_key] = cleaned
                    if out_fmt == "jsonl":
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    elif out_fmt == "txt":
                        f_out.write(_normalize_for_txt(cleaned) + "\n")
                    else:
                        raise ValueError("parquet output is not supported for jsonl input.")
                else:
                    raw = json.dumps(obj, ensure_ascii=False)
                    cleaned = clean_text(raw)
                    if structured:
                        cleaned = structure_http(cleaned, max_len=max_len)
                    if out_fmt == "jsonl":
                        f_out.write(json.dumps({text_key: cleaned}, ensure_ascii=False) + "\n")
                    elif out_fmt == "txt":
                        f_out.write(_normalize_for_txt(cleaned) + "\n")
                    else:
                        raise ValueError("parquet output is not supported for jsonl input.")
                n += 1
                if should_stop():
                    return
        return

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as e:
            raise RuntimeError("parquet support requires `pyarrow`. pip install pyarrow") from e
        table = pq.read_table(corpus_path)
        if text_key not in table.column_names:
            raise KeyError(f"Missing text_key column: {text_key}")
        if max_samples is not None:
            table = table.slice(0, max_samples)
        col = table.column(text_key).to_pylist()
        cleaned_col = [clean_text("" if s is None else str(s)) for s in col]
        if structured:
            cleaned_col = [structure_http(s, max_len=max_len) for s in cleaned_col]
        if out_fmt == "parquet":
            idx = table.schema.get_field_index(text_key)
            table = table.set_column(idx, text_key, pa.array(cleaned_col))
            pq.write_table(table, out_path)
            return
        with open(out_path, "w", encoding="utf-8") as f_out:
            for s in cleaned_col:
                if out_fmt == "txt":
                    f_out.write(_normalize_for_txt(s) + "\n")
                elif out_fmt == "jsonl":
                    f_out.write(json.dumps({text_key: s}, ensure_ascii=False) + "\n")
                else:
                    raise ValueError(f"Unknown output format: {out_fmt}. Use txt|jsonl|parquet.")
        return

    raise ValueError(f"Unknown format: {fmt}. Use txt|jsonl|parquet.")

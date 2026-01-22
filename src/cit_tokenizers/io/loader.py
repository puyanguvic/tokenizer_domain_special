from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

DATASET_REGISTRY: Dict[str, str] = {
    "hdfs": "logfit-project/HDFS_v1",
    "phish_html": "puyang2025/phish_html",
    "phishing_email": "puyang2025/seven-phishing-email-datasets",
    "waf": "puyang2025/waf_data_v2",
}

DATASET_ALIASES: Dict[str, str] = {
    "hdfs_v1": "hdfs",
    "hdfs-v1": "hdfs",
    "phish-html": "phish_html",
    "phishing-email": "phishing_email",
    "waf_v2": "waf",
    "waf-v2": "waf",
}

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "corpus"


def _normalize_dataset_key(dataset_key: str) -> str:
    return dataset_key.strip().lower().replace(" ", "_")


def list_datasets() -> Iterable[str]:
    return sorted(DATASET_REGISTRY.keys())


def resolve_dataset_key(dataset_key: str) -> str:
    if not isinstance(dataset_key, str) or not dataset_key.strip():
        raise ValueError("dataset_key must be a non-empty string")
    normalized = _normalize_dataset_key(dataset_key)
    if normalized in DATASET_REGISTRY:
        return normalized
    if normalized in DATASET_ALIASES:
        return DATASET_ALIASES[normalized]
    for key, path in DATASET_REGISTRY.items():
        if dataset_key == path:
            return key
    raise ValueError(
        f"Unknown dataset_key '{dataset_key}'. Available keys: {', '.join(list_datasets())}"
    )


def resolve_dataset_path(dataset_key: str) -> str:
    return DATASET_REGISTRY[resolve_dataset_key(dataset_key)]


def _to_str(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_to_str(v) for v in value if v is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _pick_first(row: Mapping[str, object], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            val = _to_str(row[key]).strip()
            if val:
                return val
    return ""


def _fallback_text(row: Mapping[str, object]) -> str:
    for key, val in row.items():
        if key.lower() in {"label", "anomaly", "target", "class"}:
            continue
        text = _to_str(val).strip()
        if text:
            return text
    return ""


def format_waf(row: Mapping[str, object]) -> str:
    method = _pick_first(row, ["method"])
    url = _pick_first(row, ["url"])
    proto = _pick_first(row, ["protocol"])
    headers = _pick_first(row, ["headers"])
    body = _pick_first(row, ["body"])
    return (
        f"<METHOD> {method}\n"
        f"<URL> {url}\n"
        f"<PROT> {proto}\n"
        f"<HDR>\n{headers}\n"
        f"<BODY>\n{body}\n"
    ).strip()


def format_hdfs(row: Mapping[str, object]) -> str:
    content = _pick_first(row, ["content"])
    date = _pick_first(row, ["date"])
    time = _pick_first(row, ["time"])
    level = _pick_first(row, ["level"])
    component = _pick_first(row, ["component"])
    block_id = _pick_first(row, ["block_id"])
    parts = []
    if date or time:
        parts.append(f"<TS> {date} {time}".strip())
    if level:
        parts.append(f"<LEVEL> {level}")
    if component:
        parts.append(f"<COMP> {component}")
    if block_id:
        parts.append(f"<BLK> {block_id}")
    if content:
        parts.append(f"<MSG> {content}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


def format_phish_html(row: Mapping[str, object]) -> str:
    html = _pick_first(row, ["text"])
    path = _pick_first(row, ["path"])
    year = _pick_first(row, ["year"])
    parts = []
    if path:
        parts.append(f"<PATH> {path}")
    if year:
        parts.append(f"<YEAR> {year}")
    if html:
        parts.append(f"<HTML>\n{html}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


def format_phishing_email(row: Mapping[str, object]) -> str:
    subject = _pick_first(row, ["subject"])
    sender = _pick_first(row, ["sender"])
    receiver = _pick_first(row, ["receiver"])
    body = _pick_first(row, ["text"])
    date = _pick_first(row, ["date"])
    dataset_name = _pick_first(row, ["dataset_name"])
    urls = _pick_first(row, ["urls"])
    parts = []
    if subject:
        parts.append(f"<SUBJ> {subject}")
    if sender:
        parts.append(f"<FROM> {sender}")
    if receiver:
        parts.append(f"<TO> {receiver}")
    if date:
        parts.append(f"<DATE> {date}")
    if dataset_name:
        parts.append(f"<DATASET> {dataset_name}")
    if urls:
        parts.append(f"<URLS> {urls}")
    if body:
        parts.append(f"<BODY>\n{body}")
    if parts:
        return "\n".join(parts).strip()
    return _fallback_text(row)


DATASET_FORMATTERS: Dict[str, Callable[[Mapping[str, object]], str]] = {
    "hdfs": format_hdfs,
    "phish_html": format_phish_html,
    "phishing_email": format_phishing_email,
    "waf": format_waf,
}

DEFAULT_TEXT_KEYS: Dict[str, Sequence[str]] = {
    "hdfs": ("content", "text"),
    "phish_html": ("text", "html"),
    "phishing_email": ("text", "body", "subject"),
    "waf": ("body", "url", "method"),
}


def load_dataset_by_name(
    dataset_key: str,
    split: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    **kwargs,
):
    from datasets import load_dataset as hf_load_dataset

    dataset_path = resolve_dataset_path(dataset_key)
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    def _load(split_name: Optional[str]):
        return hf_load_dataset(dataset_path, split=split_name, cache_dir=str(cache_root), **kwargs)

    if split is None:
        return _load(None)

    try:
        return _load(split)
    except ValueError as exc:
        msg = str(exc)
        if "Unknown split" not in msg:
            raise
        normalized = str(split).strip().lower()
        if normalized in {"val", "valid", "validation", "dev"}:
            for alt in ("validation", "test"):
                if alt == normalized:
                    continue
                try:
                    return _load(alt)
                except ValueError as alt_exc:
                    if "Unknown split" not in str(alt_exc):
                        raise
        raise


def iter_dataset_texts(
    dataset_key: str,
    split: str = "train",
    *,
    text_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_formatter: bool = True,
    **kwargs,
) -> Iterable[str]:
    dataset = load_dataset_by_name(
        dataset_key,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )
    key = resolve_dataset_key(dataset_key)
    formatter = DATASET_FORMATTERS.get(key) if use_formatter else None
    if text_key:
        text_keys = (text_key,)
    else:
        text_keys = DEFAULT_TEXT_KEYS.get(key, ("text",))

    n = 0
    for row in dataset:
        if formatter is not None:
            text = formatter(row)
        else:
            text = _pick_first(row, text_keys) or _fallback_text(row)
        text = text.strip() if isinstance(text, str) else str(text).strip()
        if not text:
            continue
        yield text
        n += 1
        if max_samples is not None and n >= max_samples:
            return


def export_dataset_corpus(
    dataset_key: str,
    split: str = "train",
    *,
    out_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    output_text_key: str = "text",
    text_key: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
    streaming: bool = False,
    use_formatter: bool = True,
    **kwargs,
) -> Path:
    out_root = Path(out_dir) if out_dir is not None else DEFAULT_CORPUS_DIR
    out_root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        dataset_path = resolve_dataset_path(dataset_key)
        safe_name = dataset_path.replace("/", "__")
        filename = f"{safe_name}_{split}.jsonl"
    output_path = out_root / filename

    with output_path.open("w", encoding="utf-8") as f:
        for text in iter_dataset_texts(
            dataset_key,
            split=split,
            text_key=text_key,
            cache_dir=cache_dir,
            max_samples=max_samples,
            streaming=streaming,
            use_formatter=use_formatter,
            **kwargs,
        ):
            row = {output_text_key: text}
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    return output_path


__all__ = [
    "list_datasets",
    "resolve_dataset_key",
    "resolve_dataset_path",
    "load_dataset_by_name",
    "iter_dataset_texts",
    "export_dataset_corpus",
]

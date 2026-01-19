from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

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


def list_datasets() -> Iterable[str]:
    return sorted(DATASET_REGISTRY.keys())


def resolve_dataset_path(dataset_key: str) -> str:
    if not isinstance(dataset_key, str) or not dataset_key.strip():
        raise ValueError("dataset_key must be a non-empty string")
    normalized = dataset_key.strip().lower().replace(" ", "_")
    if normalized in DATASET_REGISTRY:
        return DATASET_REGISTRY[normalized]
    if normalized in DATASET_ALIASES:
        return DATASET_REGISTRY[DATASET_ALIASES[normalized]]
    if dataset_key in DATASET_REGISTRY.values():
        return dataset_key
    raise ValueError(
        f"Unknown dataset_key '{dataset_key}'. Available keys: {', '.join(list_datasets())}"
    )


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
    return hf_load_dataset(dataset_path, split=split, cache_dir=str(cache_root), **kwargs)


__all__ = ["list_datasets", "resolve_dataset_path", "load_dataset_by_name"]

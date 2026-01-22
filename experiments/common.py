from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


# -----------------------------------------------------------------------------
# Repo import bootstrap
# -----------------------------------------------------------------------------
# These experiment scripts are intended to be runnable directly from a repo
# checkout (e.g., `python experiments/run_all.py`) without requiring an
# editable install. We therefore ensure the repo root and src/ are on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def results_root() -> Path:
    return repo_root() / "results"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_jsonable, ensure_ascii=False)


def save_csv(rows: Iterable[Sequence[Any]], path: Path, header: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def detect_label_key(example: Mapping[str, Any]) -> str:
    """Heuristic label key detector across common security datasets."""
    candidates = [
        "label",
        "labels",
        "target",
        "y",
        "class",
        "attack",
        "is_attack",
        "anomaly",
        "is_anomaly",
    ]
    lower_map: Dict[str, str] = {str(k).lower(): str(k) for k in example.keys()}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    raise KeyError(
        "Could not detect label key. Please pass --label-key explicitly. "
        f"Available keys: {sorted(example.keys())}"
    )


def to_int_label(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int,)):
        return int(v)
    if isinstance(v, float):
        return int(v)
    s = str(v).strip().lower()
    if s in {"0", "false", "benign", "normal", "ham"}:
        return 0
    if s in {"1", "true", "malicious", "attack", "anomaly", "phish", "spam"}:
        return 1
    try:
        return int(float(s))
    except Exception:
        # fallback: map unknown string to 1 (conservative for security)
        return 1


def levenshtein(a: List[int], b: List[int]) -> int:
    """Token-level Levenshtein distance (DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]

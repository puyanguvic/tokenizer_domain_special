from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import math


def get_text(row: Any, *, dataset_key: str | None = None) -> str:
    """Best-effort text extraction.

    Prefer the same dataset-specific formatter used for corpus export so
    experiments reflect the deployed interface contract.
    """

    if isinstance(row, dict):
        if "formatted" in row and row["formatted"] is not None:
            v = str(row["formatted"]).strip()
            if v:
                return v

        if dataset_key is not None:
            try:
                from cit_tokenizers.io.loader import DATASET_FORMATTERS

                fmt = DATASET_FORMATTERS.get(str(dataset_key))
                if fmt is not None:
                    v = str(fmt(row)).strip()
                    if v:
                        return v
            except Exception:
                # Formatter is a convenience; never fail experiments because of it.
                pass

        for k in ("text", "body", "url", "content", "html"):
            if k in row and row[k] is not None:
                v = str(row[k]).strip()
                if v:
                    return v
        return json.dumps(row, ensure_ascii=True)

    return str(row).strip()


def infer_num_labels(int_labels: List[int]) -> int:
    uniq = sorted(set(int_labels))
    # Common case: binary.
    if uniq == [0, 1]:
        return 2
    # If labels are already 0..K-1, use that.
    if uniq and uniq[0] == 0 and uniq[-1] == (len(uniq) - 1):
        return len(uniq)
    # Otherwise map to contiguous.
    return len(uniq) if uniq else 2


def remap_labels_to_contiguous(int_labels: List[int]) -> Tuple[List[int], Dict[int, int]]:
    uniq = sorted(set(int_labels))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    return [mapping[x] for x in int_labels], mapping


def softmax(logits):
    import torch

    return torch.softmax(logits, dim=-1)


def cosine_distance(a, b) -> float:
    import torch

    a = a / (a.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    b = b / (b.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    sim = (a * b).sum(dim=-1)
    return float((1.0 - sim).mean().item())


def auroc_binary(y_true: List[int], y_score: List[float]) -> float:
    """Compute AUROC for binary labels without sklearn.

    Uses rank-based Mann–Whitney U statistic.
    """

    if len(y_true) != len(y_score) or not y_true:
        return float("nan")
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average ranks for ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    sum_ranks_pos = sum(ranks[idx] for idx, (_, y) in enumerate(pairs) if y == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    if math.isnan(auc) or math.isinf(auc):
        return float("nan")
    return float(auc)


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

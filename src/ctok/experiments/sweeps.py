"""Experiment sweeps for paper-style evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ctok.datasets.config import DatasetConfig, DatasetExample
from ctok.datasets.io import load_dataset
from ctok.experiments.tokenizers import TokenizerAdapter, TokenizerSpec, build_tokenizer_artifact, load_tokenizer_artifact
from ctok.tokenization.boundary import DEFAULT_BOUNDARY_CHARS, normalize_boundary_chars
from ctok.utils.serialization import write_json


@dataclass(frozen=True)
class LoadedSplits:
    train: list[DatasetExample]
    eval: list[DatasetExample]
    train_split: str
    eval_split: str
    train_total: int | None = None
    eval_total: int | None = None


def run_sweep(
    *,
    dataset_config: DatasetConfig,
    tokenizer_specs: list[TokenizerSpec],
    vocab_sizes: list[int],
    artifacts_dir: Path,
    results_dir: Path,
    seed: int = 0,
    train_split: str = "train",
    eval_split: str = "validation",
    max_train_samples: int | None = 20_000,
    max_eval_samples: int | None = 5_000,
    max_text_chars: int | None = 8192,
    force_tokenizers: bool = False,
    prefix_samples_per_example: int = 2,
    robust_samples: int = 1_000,
) -> dict[str, Any]:
    """Run E1/E3-style sweeps and write raw metrics under results_dir.

    Returns the metrics dict (also written to disk).
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    splits = _load_splits(
        dataset_config,
        train_split=train_split,
        eval_split=eval_split,
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )
    train_examples = splits.train
    eval_examples = splits.eval

    train_examples = _normalize_labels(train_examples, dataset_config)
    eval_examples = _normalize_labels(eval_examples, dataset_config)

    dataset_stats = _dataset_stats(dataset_config.name, splits)

    if max_text_chars is not None:
        train_examples = _truncate_text(train_examples, max_text_chars)
        eval_examples = _truncate_text(eval_examples, max_text_chars)

    rd_records: list[dict[str, Any]] = []
    robustness_records: list[dict[str, Any]] = []
    main_records: list[dict[str, Any]] = []

    for vocab_size in vocab_sizes:
        for base_spec in tokenizer_specs:
            spec = replace(base_spec, vocab_size=vocab_size)
            artifact_dir = artifacts_dir / dataset_config.name / f"{spec.name}_{vocab_size}"
            aligned_labels: list[str] | None = None
            if all(ex.label is not None for ex in train_examples):
                aligned_labels = [str(ex.label) for ex in train_examples]
            build_tokenizer_artifact(
                spec,
                corpus=[ex.text for ex in train_examples],
                labels=aligned_labels,
                output_dir=artifact_dir,
                force=force_tokenizers,
            )
            tokenizer = load_tokenizer_artifact(artifact_dir)

            lengths = _length_stats(tokenizer, [ex.text for ex in eval_examples])
            distortion = _estimate_prefix_log_loss(
                tokenizer,
                train_examples=train_examples,
                eval_examples=eval_examples,
                seed=seed,
                samples_per_example=prefix_samples_per_example,
            )

            rd_records.append(
                {
                    "dataset": dataset_config.name,
                    "tokenizer": spec.name,
                    "vocab_size": vocab_size,
                    "rate_mean": lengths["mean"],
                    "rate_p95": lengths["p95"],
                    "distortion_log_loss": distortion,
                    "eval_count": lengths["count"],
                }
            )

            tok_s = _tokenization_throughput(tokenizer, [ex.text for ex in eval_examples])
            metric = _probe_full_sequence_metric(tokenizer, train_examples=train_examples, eval_examples=eval_examples, seed=seed)
            main_records.append(
                {
                    "dataset": dataset_config.name,
                    "tokenizer": spec.name,
                    "vocab_size": vocab_size,
                    "accuracy": metric.get("accuracy"),
                    "f1": metric.get("f1"),
                    "auroc": metric.get("auroc"),
                    "avg_len": lengths["mean"],
                    "p95_len": lengths["p95"],
                    "tokens_per_sec": tok_s,
                }
            )

            robustness = _robustness_metrics(
                tokenizer,
                examples=eval_examples[: max(0, robust_samples)] if robust_samples else eval_examples,
                seed=seed,
            )
            robustness_records.append(
                {
                    "dataset": dataset_config.name,
                    "tokenizer": spec.name,
                    "vocab_size": vocab_size,
                    **robustness,
                }
            )

    payload: dict[str, Any] = {
        "dataset": dataset_config.name,
        "dataset_stats": dataset_stats,
        "rate_distortion": rd_records,
        "main": main_records,
        "robustness": robustness_records,
    }
    write_json(results_dir / f"{dataset_config.name}_sweep.json", payload)
    return payload


def default_tokenizer_specs(config_root: Path) -> list[TokenizerSpec]:
    """Build default tokenizer specs from configs/tokenizers/*.yaml."""

    import yaml

    tokenizers_dir = config_root / "configs" / "tokenizers"
    specs: list[TokenizerSpec] = []
    for path in sorted(tokenizers_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or path.stem)
        kind = str(raw.get("type") or name)
        vocab_size = int(raw.get("vocab_size") or 16_000)
        boundary_mode = str(raw.get("boundary_mode") or ("aware" if raw.get("boundary_aware") else "none"))
        boundary_chars = normalize_boundary_chars(raw.get("boundary_chars")) if raw.get("boundary_chars") is not None else None
        if boundary_mode == "aware" and not boundary_chars:
            boundary_chars = DEFAULT_BOUNDARY_CHARS
        base_charset = str(raw.get("base_charset") or "byte")
        min_freq = int(raw.get("min_freq") or 2)
        min_len = int(raw.get("min_len") or 2)
        max_len = int(raw.get("max_len") or 8)
        lambda_weight = float(raw.get("lambda_weight") or 0.0)
        special_tokens = raw.get("special_tokens")
        if special_tokens is not None and not isinstance(special_tokens, dict):
            special_tokens = None
        specs.append(
            TokenizerSpec(
                name=name,
                kind=kind,
                vocab_size=vocab_size,
                boundary_mode=boundary_mode,
                boundary_chars=boundary_chars,
                base_charset=base_charset,
                min_freq=min_freq,
                min_len=min_len,
                max_len=max_len,
                lambda_weight=lambda_weight,
                special_tokens=dict(special_tokens) if special_tokens else None,
            )
        )
    return specs


def _load_splits(
    dataset_config: DatasetConfig,
    *,
    train_split: str,
    eval_split: str,
    seed: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> LoadedSplits:
    if dataset_config.hf_dataset:
        available = _hub_parquet_splits(dataset_config.hf_dataset)
        if available:
            resolved_train = _resolve_split(train_split, available, fallback_order=["train"])
            resolved_eval = _resolve_split(
                eval_split,
                available,
                fallback_order=["validation", "eval", "test", "heldout"],
            )
            train = _load_hub_parquet(dataset_config, resolved_train, max_train_samples)
            eval_ = _load_hub_parquet(dataset_config, resolved_eval, max_eval_samples)
            return LoadedSplits(
                train=train,
                eval=eval_,
                train_split=resolved_train,
                eval_split=resolved_eval,
                train_total=_hub_parquet_num_rows(dataset_config.hf_dataset, resolved_train),
                eval_total=_hub_parquet_num_rows(dataset_config.hf_dataset, resolved_eval),
            )

    try:
        train_cfg = replace(dataset_config, hf_split=train_split, max_samples=max_train_samples)
        eval_cfg = replace(dataset_config, hf_split=eval_split, max_samples=max_eval_samples)
        train = load_dataset(train_cfg)
        eval_ = load_dataset(eval_cfg)
        return LoadedSplits(train=train, eval=eval_, train_split=train_split, eval_split=eval_split)
    except Exception:
        # Fallback: load a single split and create a deterministic split.
        base = load_dataset(replace(dataset_config, max_samples=None))
        rng = _rng(seed)
        indices = list(range(len(base)))
        rng.shuffle(indices)
        split_point = int(0.8 * len(indices))
        train_idx = indices[:split_point]
        eval_idx = indices[split_point:]
        train = [base[i] for i in train_idx]
        eval_ = [base[i] for i in eval_idx]
        if max_train_samples is not None:
            train = train[:max_train_samples]
        if max_eval_samples is not None:
            eval_ = eval_[:max_eval_samples]
        return LoadedSplits(train=train, eval=eval_, train_split="train", eval_split="eval")


def _resolve_split(requested: str, available: set[str], *, fallback_order: list[str]) -> str:
    if requested in available:
        return requested
    for name in fallback_order:
        if name in available:
            return name
    return sorted(available)[0]


def _hub_parquet_splits(dataset_id: str) -> set[str]:
    snapshot = _latest_hub_snapshot(dataset_id)
    if snapshot is None:
        return set()
    return {path.stem for path in snapshot.glob("*.parquet")}


def _load_hub_parquet(
    dataset_config: DatasetConfig,
    split: str,
    max_samples: int | None,
) -> list[DatasetExample]:
    snapshot = _latest_hub_snapshot(dataset_config.hf_dataset or "")
    if snapshot is None:
        raise FileNotFoundError(f"No hub snapshot found for {dataset_config.hf_dataset}")
    parquet_path = snapshot / f"{split}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet split {split!r} for {dataset_config.hf_dataset}")

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    columns: list[str] = []
    if dataset_config.sequence_field:
        columns.append(dataset_config.sequence_field)
    elif dataset_config.text_fields:
        columns.extend(dataset_config.text_fields)
    else:
        columns.append(dataset_config.text_field)
    if dataset_config.label_field:
        columns.append(dataset_config.label_field)
    if dataset_config.group_field:
        columns.append(dataset_config.group_field)

    examples: list[DatasetExample] = []
    for batch in pf.iter_batches(batch_size=1024, columns=columns):
        data = batch.to_pydict()
        size = len(next(iter(data.values()))) if data else 0
        for idx in range(size):
            row = {key: values[idx] for key, values in data.items()}
            text = _resolve_text(row, dataset_config)
            label = row.get(dataset_config.label_field) if dataset_config.label_field else None
            meta = {dataset_config.group_field: row.get(dataset_config.group_field)} if dataset_config.group_field else None
            examples.append(DatasetExample(text=text, label=label, meta=meta))
            if max_samples is not None and len(examples) >= max_samples:
                return examples
    return examples


def _hub_parquet_num_rows(dataset_id: str, split: str) -> int | None:
    snapshot = _latest_hub_snapshot(dataset_id)
    if snapshot is None:
        return None
    path = snapshot / f"{split}.parquet"
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def _latest_hub_snapshot(dataset_id: str) -> Path | None:
    if "/" not in dataset_id:
        return None
    namespace, name = dataset_id.split("/", 1)
    base = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{namespace}--{name}" / "snapshots"
    if not base.is_dir():
        return None
    snapshots = [path for path in base.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]


def _resolve_text(payload: dict[str, Any], config: DatasetConfig) -> str:
    if config.sequence_field and config.sequence_field in payload:
        sequence = payload[config.sequence_field]
        if isinstance(sequence, list):
            return config.sequence_delimiter.join(str(item) for item in sequence)
        return str(sequence)
    if config.text_fields:
        parts: list[str] = []
        for field in config.text_fields:
            value = payload.get(field)
            if value is None:
                continue
            parts.append(str(value))
        if parts:
            return config.field_delimiter.join(parts)
    if config.text_field in payload:
        return str(payload[config.text_field])
    available = ", ".join(sorted(payload.keys()))
    raise KeyError(f"Missing text field '{config.text_field}'. Available: {available}")


def _dataset_stats(name: str, splits: LoadedSplits) -> dict[str, Any]:
    def avg_bytes(examples: list[DatasetExample]) -> float:
        if not examples:
            return 0.0
        return sum(len(ex.text.encode("utf-8", errors="replace")) for ex in examples) / len(examples)

    return {
        "name": name,
        "train_split": splits.train_split,
        "eval_split": splits.eval_split,
        "train_count": len(splits.train),
        "eval_count": len(splits.eval),
        "train_total": splits.train_total,
        "eval_total": splits.eval_total,
        "train_avg_bytes": avg_bytes(splits.train),
        "eval_avg_bytes": avg_bytes(splits.eval),
    }


def _normalize_labels(examples: list[DatasetExample], dataset_config: DatasetConfig) -> list[DatasetExample]:
    mapping = dataset_config.label_mapping or {}
    normalized: list[DatasetExample] = []
    for ex in examples:
        if ex.label is None:
            normalized.append(ex)
            continue
        if isinstance(ex.label, int):
            normalized.append(ex)
            continue
        label_str = str(ex.label)
        if mapping:
            if label_str not in mapping:
                raise ValueError(f"Unknown label {label_str!r} for dataset {dataset_config.name}")
            label = mapping[label_str]
        else:
            label = int(label_str)
        normalized.append(DatasetExample(text=ex.text, label=label, meta=ex.meta))
    return normalized


def _truncate_text(examples: list[DatasetExample], max_chars: int) -> list[DatasetExample]:
    if max_chars <= 0:
        return examples
    truncated: list[DatasetExample] = []
    for ex in examples:
        text = ex.text[:max_chars]
        truncated.append(DatasetExample(text=text, label=ex.label, meta=ex.meta))
    return truncated


def _length_stats(tokenizer: TokenizerAdapter, texts: list[str]) -> dict[str, Any]:
    lengths: list[int] = []
    for text in texts:
        if not text:
            continue
        lengths.append(len(tokenizer.encode(text)))
    lengths.sort()
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0, "max": 0, "p50": 0, "p95": 0}

    def pct(p: float) -> int:
        idx = int((len(lengths) - 1) * p)
        return lengths[idx]

    return {
        "count": len(lengths),
        "mean": sum(lengths) / len(lengths),
        "min": lengths[0],
        "max": lengths[-1],
        "p50": pct(0.50),
        "p95": pct(0.95),
    }


def _estimate_prefix_log_loss(
    tokenizer: TokenizerAdapter,
    *,
    train_examples: list[DatasetExample],
    eval_examples: list[DatasetExample],
    seed: int,
    samples_per_example: int,
) -> float:
    sklearn = _require_sklearn()
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    delimiter = "\u0001"

    def to_prefix_samples(examples: list[DatasetExample]) -> tuple[list[str], list[int]]:
        rng = _rng(seed)
        xs: list[str] = []
        ys: list[int] = []
        for ex in examples:
            if ex.label is None:
                continue
            tokens = tokenizer.tokenize(ex.text)
            if not tokens:
                continue
            for _ in range(max(samples_per_example, 1)):
                cut = rng.randint(1, len(tokens))
                xs.append(delimiter.join(tokens[:cut]))
                ys.append(int(ex.label) if isinstance(ex.label, int) else int(str(ex.label)))
        return xs, ys

    train_x, train_y = to_prefix_samples(train_examples)
    eval_x, eval_y = to_prefix_samples(eval_examples)
    if not train_x or not eval_x:
        return 0.0

    vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        analyzer="word",
        tokenizer=lambda s: s.split(delimiter),
        token_pattern=None,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.transform(train_x)
    x_eval = vectorizer.transform(eval_x)

    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(x_train, train_y)
    probs = clf.predict_proba(x_eval)
    return float(log_loss(eval_y, probs, labels=clf.classes_))


def _probe_full_sequence_metric(
    tokenizer: TokenizerAdapter,
    *,
    train_examples: list[DatasetExample],
    eval_examples: list[DatasetExample],
    seed: int,
) -> dict[str, float]:
    _require_sklearn()
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    delimiter = "\u0001"

    def to_samples(examples: list[DatasetExample]) -> tuple[list[str], list[int]]:
        xs: list[str] = []
        ys: list[int] = []
        for ex in examples:
            if ex.label is None:
                continue
            tokens = tokenizer.tokenize(ex.text)
            if not tokens:
                continue
            xs.append(delimiter.join(tokens))
            ys.append(int(ex.label) if isinstance(ex.label, int) else int(str(ex.label)))
        return xs, ys

    train_x, train_y = to_samples(train_examples)
    eval_x, eval_y = to_samples(eval_examples)
    if not train_x or not eval_x:
        return {}

    vectorizer = HashingVectorizer(
        n_features=2**18,
        alternate_sign=False,
        analyzer="word",
        tokenizer=lambda s: s.split(delimiter),
        token_pattern=None,
        ngram_range=(1, 2),
    )
    x_train = vectorizer.transform(train_x)
    x_eval = vectorizer.transform(eval_x)

    clf = LogisticRegression(max_iter=1000, n_jobs=1)
    clf.fit(x_train, train_y)

    preds = clf.predict(x_eval)
    out: dict[str, float] = {"accuracy": float(accuracy_score(eval_y, preds))}
    out["f1"] = float(f1_score(eval_y, preds, zero_division=0))
    if len(clf.classes_) == 2:
        probs = clf.predict_proba(x_eval)[:, 1]
        out["auroc"] = float(roc_auc_score(eval_y, probs))
    return out


def _tokenization_throughput(tokenizer: TokenizerAdapter, texts: list[str]) -> float:
    import time

    total_tokens = 0
    start = time.perf_counter()
    for text in texts:
        total_tokens += len(tokenizer.encode(text))
    elapsed = time.perf_counter() - start
    return float(total_tokens / elapsed) if elapsed else 0.0


def _robustness_metrics(
    tokenizer: TokenizerAdapter,
    *,
    examples: list[DatasetExample],
    seed: int,
) -> dict[str, float]:
    rng = _rng(seed)
    transforms = [_collapse_whitespace, _lowercase]

    distances: list[float] = []
    jitters: list[float] = []
    for ex in examples:
        if not ex.text:
            continue
        orig = tokenizer.encode(ex.text)
        if not orig:
            continue
        text2 = transforms[rng.randrange(len(transforms))](ex.text)
        pert = tokenizer.encode(text2)
        dist = _levenshtein(orig, pert)
        distances.append(dist / max(len(orig), 1))
        jitters.append(abs(len(orig) - len(pert)) / max(len(orig), 1))

    if not distances:
        return {"token_edit_norm": 0.0, "length_jitter": 0.0}
    return {
        "token_edit_norm": float(sum(distances) / len(distances)),
        "length_jitter": float(sum(jitters) / len(jitters)),
    }


def _collapse_whitespace(text: str) -> str:
    import re

    return re.sub(r"\\s+", " ", text).strip()


def _lowercase(text: str) -> str:
    return text.lower()


def _levenshtein(a: list[int], b: list[int]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, start=1):
        curr = [i]
        for j, bj in enumerate(b, start=1):
            cost = 0 if ai == bj else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _rng(seed: int):
    import random

    rng = random.Random(seed)
    return rng


def _require_sklearn() -> Any:
    try:
        import sklearn  # type: ignore
    except ImportError as exc:
        raise ImportError("Install scikit-learn to run experiment probes.") from exc
    return sklearn

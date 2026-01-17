"""Plotting utilities for paper-style figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def plot_rate_distortion(
    *,
    records: Iterable[dict[str, Any]],
    output_path: Path,
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()

    data = list(records)
    datasets = sorted({row["dataset"] for row in data})
    tokenizers = sorted({row["tokenizer"] for row in data})

    fig, axes = plt.subplots(1, max(len(datasets), 1), figsize=(5 * max(len(datasets), 1), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = [row for row in data if row["dataset"] == dataset]
        for tok in tokenizers:
            rows = [row for row in subset if row["tokenizer"] == tok]
            rows.sort(key=lambda r: int(r["vocab_size"]))
            xs = [float(r["rate_mean"]) for r in rows]
            ys = [float(r["distortion_log_loss"]) for r in rows]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", label=tok)
        ax.set_title(dataset)
        ax.set_xlabel("Token rate (mean length)")
        ax.set_ylabel("Distortion (log loss)")
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    axes[0].legend(loc="best", fontsize="small")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_robustness(
    *,
    records: Iterable[dict[str, Any]],
    output_path: Path,
    title: str | None = None,
) -> None:
    plt = _require_matplotlib()

    data = list(records)
    datasets = sorted({row["dataset"] for row in data})
    tokenizers = sorted({row["tokenizer"] for row in data})

    fig, axes = plt.subplots(1, max(len(datasets), 1), figsize=(5 * max(len(datasets), 1), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        subset = [row for row in data if row["dataset"] == dataset]
        subset.sort(key=lambda r: (r["tokenizer"], int(r["vocab_size"])))
        # Choose the largest vocab size per tokenizer for a main-text style snapshot.
        rows: list[dict[str, Any]] = []
        for tok in tokenizers:
            tok_rows = [r for r in subset if r["tokenizer"] == tok]
            if not tok_rows:
                continue
            tok_rows.sort(key=lambda r: int(r["vocab_size"]))
            rows.append(tok_rows[-1])

        xs = list(range(len(rows)))
        heights = [float(r.get("token_edit_norm", 0.0)) for r in rows]
        labels = [str(r["tokenizer"]) for r in rows]
        ax.bar(xs, heights)
        ax.set_xticks(xs, labels, rotation=45, ha="right")
        ax.set_title(dataset)
        ax.set_ylabel("Token edit distance (normalized)")
        ax.grid(True, axis="y", alpha=0.3)

    if title:
        fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise ImportError("Install matplotlib to plot experiment results.") from exc
    return plt

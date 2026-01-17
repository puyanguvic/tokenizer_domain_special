"""Reporting helpers for paper-facing tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_latex_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[tuple[str, str]],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write a simple booktabs LaTeX table.

    columns: list of (key, header) pairs.
    """

    def fmt(value: Any) -> str:
        if value is None:
            return r"\texttt{N/A}"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{" + "l" * len(columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header for _key, header in columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(fmt(row.get(key)) for key, _header in columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    dataset_rows: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Write CSV + LaTeX tables under output_dir/{tables,metrics}."""

    tables_dir = output_dir / "tables"
    write_csv(
        tables_dir / "datasets.csv",
        dataset_rows,
        fieldnames=["domain", "task", "train", "eval", "avg_bytes_train", "avg_bytes_eval"],
    )
    write_latex_table(
        tables_dir / "datasets.tex",
        caption="Datasets and tasks.",
        label="tab:datasets",
        columns=[
            ("domain", "Domain"),
            ("task", "Task"),
            ("train", r"\#Train"),
            ("eval", r"\#Eval"),
            ("avg_bytes_train", "Avg bytes (train)"),
        ],
        rows=dataset_rows,
    )

    write_csv(
        tables_dir / "main_results.csv",
        main_rows,
        fieldnames=[
            "dataset",
            "tokenizer",
            "vocab_size",
            "accuracy",
            "f1",
            "auroc",
            "avg_len",
            "p95_len",
            "tokens_per_sec",
        ],
    )
    write_latex_table(
        tables_dir / "main_results.tex",
        caption="Equal-compute proxy metrics and interface statistics (probe + tokenization).",
        label="tab:main_results",
        columns=[
            ("dataset", "Domain"),
            ("tokenizer", "Tokenizer"),
            ("auroc", "AUROC"),
            ("f1", "F1"),
            ("avg_len", "Avg len"),
            ("p95_len", "P95 len"),
            ("tokens_per_sec", "tokens/s"),
        ],
        rows=main_rows,
    )

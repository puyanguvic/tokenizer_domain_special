from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer

from cit_tokenizers.tokenization_cit import CITTokenizer

from experiments.e1_interface import E1Config, run_e1
from experiments.e2_equal_compute import E2Config, run_e2
from experiments.e3_robustness import E3Config, run_e3


def train_tokenizer(repo_root: Path, dataset: str, algorithm: str, name: str, *, overwrite_corpus: bool) -> Path:
    cmd: List[str] = [
        "python",
        "experiments/run_experiment.py",
        "--dataset",
        dataset,
        "--algorithm",
        algorithm,
        "--name",
        name,
        "--verify",
    ]
    if overwrite_corpus:
        cmd.append("--overwrite")
    subprocess.check_call(cmd, cwd=str(repo_root))
    outdir = repo_root / "tokenizers" / name
    if not outdir.exists():
        raise FileNotFoundError(f"Expected tokenizer output at {outdir}")
    return outdir


def load_tokenizer(algorithm: str, tok_dir: Path):
    if algorithm == "cit":
        return CITTokenizer.from_pretrained(str(tok_dir))
    return AutoTokenizer.from_pretrained(str(tok_dir), trust_remote_code=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run CIT paper experiments and save outputs under results/."
    )
    ap.add_argument("--datasets", nargs="+", default=["waf"], help="Datasets to run (e.g., waf phish_html hdfs).")
    ap.add_argument(
        "--algorithms",
        nargs="+",
        default=["cit", "bpeh", "unigramh", "wordpieceh"],
        help="Tokenizer algorithms.",
    )
    ap.add_argument("--max-e1", type=int, default=20000, help="Max samples for E1 probe.")
    ap.add_argument("--max-e3", type=int, default=2000, help="Max samples for E3 robustness.")
    ap.add_argument("--e2-total-tokens", type=int, default=5_000_000, help="Encoder-token budget for E2.")
    ap.add_argument("--skip-e2", action="store_true")
    ap.add_argument("--skip-e1", action="store_true")
    ap.add_argument("--skip-e3", action="store_true")
    ap.add_argument("--overwrite-corpus", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    for ds in args.datasets:
        for alg in args.algorithms:
            name = f"{ds}_{alg}_tokenizer"
            tok_dir = train_tokenizer(repo_root, ds, alg, name, overwrite_corpus=bool(args.overwrite_corpus))
            tok = load_tokenizer(alg, tok_dir)

            if not args.skip_e1:
                run_e1(cfg=E1Config(dataset=ds, split="val", max_samples=int(args.max_e1)), tokenizer=tok, tokenizer_name=name)

            if not args.skip_e3:
                run_e3(cfg=E3Config(dataset=ds, split="val", max_samples=int(args.max_e3)), tokenizer=tok, tokenizer_name=name)

            if not args.skip_e2:
                run_e2(cfg=E2Config(dataset=ds, total_tokens=int(args.e2_total_tokens)), tokenizer=tok, tokenizer_name=name)


if __name__ == "__main__":
    main()

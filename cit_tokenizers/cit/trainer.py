from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..contract import Contract, ContractConfig
from .compiler import CompiledMatcher
from .runtime import CITArtifact


def _is_boundary(ch: str, boundaries: Sequence[str]) -> bool:
    return ch in boundaries


def _default_boundaries() -> List[str]:
    # Matches the appendix default boundary set.
    return list(" \t\n:;,=&?/ #%()[]{}<>\"'|")


@dataclass
class CITTrainerConfig:
    """Build configuration for CIT."""

    vocab_size: int = 8192
    min_freq: int = 10
    len_min: int = 2
    len_max: int = 24
    boundaries: Optional[List[str]] = None
    lambda_rd: float = 0.0
    seed: int = 0
    sample_texts: Optional[int] = 200_000
    # Distortion proxy options
    distortion_mode: str = "none"  # 'none' or 'boundary_penalty'
    boundary_penalty: float = 1.0
    # Contract
    contract: ContractConfig = ContractConfig()

    def __post_init__(self) -> None:
        if self.boundaries is None:
            self.boundaries = _default_boundaries()
        if self.len_min < 1 or self.len_max < self.len_min:
            raise ValueError("Invalid candidate length range")


class CITTrainer:
    """Trainer that builds a CIT artifact.

    Notes
    -----
    * The full paper describes teacher-aligned distortion via probe cross-entropy.
      In this package we keep the public API stable while providing a safe,
      dependency-free default distortion proxy (boundary penalty). You can plug in
      a label/probe-based estimator later without changing the artifact format.
    """

    SPECIAL_TOKENS: Sequence[str] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

    def __init__(self, config: Optional[CITTrainerConfig] = None):
        self.cfg = config or CITTrainerConfig()
        self._rng = random.Random(self.cfg.seed)
        self._contract = Contract(self.cfg.contract)

    # -------------------------
    # Public API
    # -------------------------
    def train_from_iterator(
        self,
        texts: Iterable[str],
        outdir: str | Path,
        *,
        additional_special_tokens: Optional[Sequence[str]] = None,
    ) -> CITArtifact:
        """Train and write an artifact directory.

        Parameters
        ----------
        texts:
            Iterable of raw strings (train split). If you need JSON field-aware
            serialization, apply it *before* calling this trainer.
        outdir:
            Output directory that will contain a CIT artifact (config + matcher + vocab).
        additional_special_tokens:
            Optional extra tokens to reserve at the front of the vocab.
        """

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # 1) Contract pass + sampling
        proc: List[str] = []
        for i, t in enumerate(texts):
            if self.cfg.sample_texts is not None and i >= self.cfg.sample_texts:
                break
            proc.append(self._contract.apply(t))

        # 2) Candidate extraction
        cand_freq = self._extract_candidates(proc)

        # 3) Greedy induction
        vocab = self._induce_vocab(proc, cand_freq, additional_special_tokens=additional_special_tokens)

        # 4) Compile matcher and write artifact
        matcher = CompiledMatcher.compile(vocab=vocab, max_token_len=self.cfg.len_max)
        art = CITArtifact(
            vocab=vocab,
            matcher=matcher,
            contract=self._contract.config,
            special_tokens=list(self.SPECIAL_TOKENS) + list(additional_special_tokens or []),
        )
        self._write_artifact(outdir, art)
        return art

    # -------------------------
    # Internals
    # -------------------------
    def _extract_candidates(self, texts: Sequence[str]) -> Dict[str, int]:
        """Extract contiguous-span candidates that respect boundary set.

        This is deliberately conservative: we only consider spans that do not cross
        boundaries and fall within [len_min, len_max].
        """

        boundaries = set(self.cfg.boundaries or [])
        freq: Dict[str, int] = {}
        for s in texts:
            n = len(s)
            i = 0
            while i < n:
                # skip boundaries
                if s[i] in boundaries:
                    i += 1
                    continue

                # find maximal non-boundary segment [i, j)
                j = i
                while j < n and s[j] not in boundaries:
                    j += 1
                seg = s[i:j]

                # enumerate spans within segment
                L = len(seg)
                for a in range(L):
                    max_b = min(L, a + self.cfg.len_max)
                    for b in range(a + self.cfg.len_min, max_b + 1):
                        tok = seg[a:b]
                        freq[tok] = freq.get(tok, 0) + 1

                i = j

        # filter by min_freq and remove specials / empty
        out = {t: c for t, c in freq.items() if c >= self.cfg.min_freq and t and t not in self.SPECIAL_TOKENS}
        return out

    def _baseline_vocab(self, additional_special_tokens: Optional[Sequence[str]]) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        idx = 0
        for tok in list(self.SPECIAL_TOKENS) + list(additional_special_tokens or []):
            if tok not in vocab:
                vocab[tok] = idx
                idx += 1
        # reserve typed symbols from hygiene so that integrity constraints hold
        for t in self._contract.typed_symbols():
            if t not in vocab:
                vocab[t] = idx
                idx += 1
        return vocab

    def _distortion_proxy(self, token: str) -> float:
        """Default distortion proxy used during induction.

        Without labels/teachers, we approximate the *risk* of a token crossing a
        structural boundary by penalizing tokens that contain boundary characters.

        If you later plug in a probe-based estimator, you can set distortion_mode
        accordingly and override this method.
        """

        if self.cfg.distortion_mode == "none" or self.cfg.lambda_rd <= 0:
            return 0.0
        if self.cfg.distortion_mode == "boundary_penalty":
            b = set(self.cfg.boundaries or [])
            return self.cfg.boundary_penalty * sum(1 for ch in token if ch in b)
        raise ValueError(f"Unknown distortion_mode={self.cfg.distortion_mode}")

    def _induce_vocab(
        self,
        texts: Sequence[str],
        cand_freq: Dict[str, int],
        *,
        additional_special_tokens: Optional[Sequence[str]],
    ) -> Dict[str, int]:
        """Greedy gain–distortion selection.

        Gain is approximated with an analytic upper bound using frequency and token length:
            g(c) ≈ freq(c) * (len(c) - 1)
        which captures the max possible character saving if c replaces its characters.

        This keeps the trainer lightweight and deterministic, while still providing
        a meaningful rate signal.
        """

        vocab = self._baseline_vocab(additional_special_tokens)
        budget = self.cfg.vocab_size
        if budget < len(vocab):
            raise ValueError(f"vocab_size={budget} is smaller than required specials+typed={len(vocab)}")

        # Pre-score candidates
        scored: List[Tuple[float, str]] = []
        lam = float(self.cfg.lambda_rd)
        for tok, f in cand_freq.items():
            if tok in vocab:
                continue
            gain = float(f) * float(max(len(tok) - 1, 0))
            dist = self._distortion_proxy(tok)
            score = gain - lam * dist
            scored.append((score, tok))

        # Deterministic tie-breaking: higher score, then longer, then lexicographic.
        scored.sort(key=lambda x: (x[0], len(x[1]), x[1]), reverse=True)

        next_id = max(vocab.values()) + 1 if vocab else 0
        for _, tok in scored:
            if len(vocab) >= budget:
                break
            vocab[tok] = next_id
            next_id += 1
        return vocab

    def _write_artifact(self, outdir: Path, art: CITArtifact) -> None:
        # Core artifact JSON (used by CITRuntime)
        (outdir / "cit_artifact.json").write_text(art.dumps(), encoding="utf-8")

        # A minimal transformers-compatible folder structure.
        # AutoTokenizer can load this with trust_remote_code=True.
        (outdir / "tokenizer_config.json").write_text(
            json.dumps({"tokenizer_class": "CITTokenizer", "model_max_length": 512}, indent=2),
            encoding="utf-8",
        )
        (outdir / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "unk_token": "[UNK]",
                    "sep_token": "[SEP]",
                    "pad_token": "[PAD]",
                    "cls_token": "[CLS]",
                    "mask_token": "[MASK]",
                    "additional_special_tokens": [t for t in art.special_tokens if t not in self.SPECIAL_TOKENS],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # Provide the tokenizer implementation alongside the artifact.
        pkg_dir = outdir / "cit_tokenizers"
        pkg_dir.mkdir(exist_ok=True)
        (pkg_dir / "__init__.py").write_text("# Local package stub for AutoTokenizer remote code\n", encoding="utf-8")
        (pkg_dir / "tokenization_cit.py").write_text(
            (Path(__file__).resolve().parents[1] / "tokenization_cit.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

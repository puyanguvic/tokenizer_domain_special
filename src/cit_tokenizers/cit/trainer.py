from __future__ import annotations

import json
import re
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import CITBuildConfig, CITTrainerConfig
from ..interface.contract import Contract, ContractConfig
from .compiler import CompiledMatcher, compile_trie
from .runtime import CITArtifact



TAG_RE = re.compile(r"<[A-Z][A-Z0-9_]{1,31}>")

def _auto_special_tokens_http(texts: Sequence[str], *, min_freq: int = 1, max_tags: int = 64) -> List[str]:
    """Auto-discover uppercase angle-bracket tags like <METHOD> and <URL>.

    This keeps interface markers atomic even when '<' and '>' are boundaries.
    We cap the number of discovered tags to avoid accidental blow-ups.
    """
    freq: Dict[str, int] = {}
    for s in texts:
        for m in TAG_RE.finditer(s):
            t = m.group(0)
            freq[t] = freq.get(t, 0) + 1
    tags = [t for t, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq]
    return tags[:max_tags]


def _is_boundary(ch: str, boundaries: Sequence[str]) -> bool:
    return ch in boundaries


def _default_boundaries() -> List[str]:
    # Matches the appendix default boundary set.
    return list(" \t\n:;,=&?/ #%()[]{}<>\"'|")


def _http_boundaries() -> List[str]:
    # Structured HTTP/log-like boundaries.
    return list(" \t\n=,:&?/ #%()[]{}<>\"'|.-_\\+*;!$")


def _boundary_preset(name: str) -> List[str]:
    key = name.strip().lower()
    if key in ("default", "base"):
        return _default_boundaries()
    if key in ("http", "waf"):
        return _http_boundaries()
    raise ValueError(f"Unknown boundary preset '{name}'")


def _finalize_cfg(cfg: CITTrainerConfig) -> CITTrainerConfig:
    """Fill preset-dependent defaults.

    We keep this as a pure function so that the unified config schema remains a
    simple dataclass without side-effecting __post_init__.
    """

    preset = (cfg.preset or "default").strip().lower()
    boundaries = cfg.boundaries if cfg.boundaries is not None else _boundary_preset(preset)
    include_char_vocab = cfg.include_char_vocab if cfg.include_char_vocab is not None else preset in ("http", "waf")
    include_ascii_fallback = (
        cfg.include_ascii_fallback if cfg.include_ascii_fallback is not None else preset in ("http", "waf")
    )
    symbol_ngram_max_len = cfg.symbol_ngram_max_len
    if symbol_ngram_max_len is None:
        symbol_ngram_max_len = 4 if preset in ("http", "waf") else 0
    len_min = cfg.len_min
    if len_min is None:
        len_min = 1 if preset in ("http", "waf") else 2
    if len_min < 1 or cfg.len_max < len_min:
        raise ValueError("Invalid candidate length range")
    return CITTrainerConfig(
        vocab_size=cfg.vocab_size,
        min_freq=cfg.min_freq,
        len_min=len_min,
        len_max=cfg.len_max,
        boundaries=list(boundaries),
        preset=cfg.preset,
        lambda_rd=cfg.lambda_rd,
        seed=cfg.seed,
        sample_texts=cfg.sample_texts,
        distortion_mode=cfg.distortion_mode,
        boundary_penalty=cfg.boundary_penalty,
        include_char_vocab=include_char_vocab,
        include_ascii_fallback=include_ascii_fallback,
        symbol_ngram_min_len=cfg.symbol_ngram_min_len,
        symbol_ngram_max_len=symbol_ngram_max_len,
    )


def _ascii_fallback_chars() -> List[str]:
    """A conservative ASCII fallback alphabet.

    We include all printable ASCII chars plus common whitespace.
    This is small (~100 tokens) but eliminates [UNK] for most structured domains.
    """

    chars = [chr(i) for i in range(32, 127)]  # printable
    chars.extend(["\t", "\n", "\r"])  # common whitespace
    # ensure deterministic order
    seen = set()
    out: List[str] = []
    for c in chars:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


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

    def __init__(
        self,
        trainer_config: Optional[CITTrainerConfig] = None,
        *,
        contract_config: Optional[ContractConfig] = None,
        build_config: Optional[CITBuildConfig] = None,
    ):
        """Create a trainer.

        You can either pass (trainer_config, contract_config) directly, or pass
        a unified :class:`~cit_tokenizers.config.CITBuildConfig`.
        """

        if build_config is not None:
            trainer_config = build_config.trainer
            contract_config = build_config.contract
            self.build_config = build_config
        else:
            self.build_config = CITBuildConfig(
                trainer=trainer_config or CITTrainerConfig(),
                contract=contract_config or ContractConfig(),
            )
        self.cfg = _finalize_cfg(self.build_config.trainer)
        # Preset-aware default: for HTTP/WAF-like domains, use HTTP-aware hygiene
        # unless the user explicitly overrides typed_hygiene_mode.
        if (self.cfg.preset or "default").strip().lower() in ("http", "waf"):
            mode = getattr(self.build_config.contract, "typed_hygiene_mode", "generic") or "generic"
            if str(mode).lower() in ("generic", ""):
                self.build_config = replace(
                    self.build_config,
                    contract=replace(self.build_config.contract, typed_hygiene_mode="http"),
                )

        self._rng = random.Random(self.cfg.seed)
        self._contract = Contract(self.build_config.contract)

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


        # Auto-reserve contract-emitted typed symbols and common interface markers.
        auto_special: List[str] = []
        # typed symbols (<UUID>, <IPV4>, ...) should be atomic
        auto_special.extend(self._contract.typed_symbols())
        # contract separators should be atomic if present
        ccfg = self._contract.config
        for t in (ccfg.sep_token, ccfg.eol_token, ccfg.empty_token):
            if t:
                auto_special.append(t)

        preset = (self.cfg.preset or "default").strip().lower()
        if preset in ("http", "waf"):
            # Auto-detect uppercase tags like <METHOD>, <URL>, <HDR>
            auto_special.extend(_auto_special_tokens_http(proc, min_freq=max(1, self.cfg.min_freq), max_tags=64))

        if additional_special_tokens is None:
            additional_special_tokens = auto_special
        else:
            # prepend auto specials, preserving user-provided order and de-duping
            merged: List[str] = []
            seen = set()
            for t in list(auto_special) + list(additional_special_tokens):
                if t and t not in seen and t not in self.SPECIAL_TOKENS:
                    merged.append(t)
                    seen.add(t)
            additional_special_tokens = merged
        # 2) Candidate extraction
        cand_freq = self._extract_candidates(proc)

        # 3) Greedy induction
        char_vocab = None
        if self.cfg.include_char_vocab or self.cfg.include_ascii_fallback:
            chars = set()
            if self.cfg.include_char_vocab:
                # chars observed in the processed corpus
                chars |= {ch for s in proc for ch in s}
                # always include boundary symbols if char vocab is enabled
                chars |= set(self.cfg.boundaries or [])
            if self.cfg.include_ascii_fallback:
                chars |= set(_ascii_fallback_chars())
                # also ensure boundary symbols are present for the preset
                chars |= set(self.cfg.boundaries or [])
            char_vocab = sorted(chars)
        vocab = self._induce_vocab(
            proc,
            cand_freq,
            additional_special_tokens=additional_special_tokens,
            char_vocab=char_vocab,
        )

        # 4) Compile matcher and write artifact
        matcher = compile_trie(vocab.items())
        art = CITArtifact(
            meta={
                "schema_version": "cit_artifact.v1",
                "builder": "cit_tokenizers",
                "build_config": self.build_config.to_dict(),
            },
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
        symbol_chars = {ch for ch in boundaries if not ch.isspace()}
        # Avoid creating tokens that straddle typed/tag delimiters like '<' or '>'
        # (e.g., ':<' or '/<' or '%<') which can prevent matching atomic specials such as <PORT>.
        symbol_chars.discard('<')
        symbol_chars.discard('>')
        min_sym = int(self.cfg.symbol_ngram_min_len)
        max_sym = int(self.cfg.symbol_ngram_max_len)
        if max_sym < min_sym:
            max_sym = 0
        freq: Dict[str, int] = {}
        for s in texts:
            n = len(s)
            i = 0
            while i < n:
                # skip boundaries
                if s[i] in boundaries:
                    # optional: add symbol n-gram candidates from boundary runs
                    if max_sym > 0 and symbol_chars:
                        j = i
                        while j < n and s[j] in boundaries:
                            j += 1
                        k = i
                        while k < j:
                            if s[k] not in symbol_chars:
                                k += 1
                                continue
                            r = k
                            while r < j and s[r] in symbol_chars:
                                r += 1
                            seg = s[k:r]
                            L = len(seg)
                            max_b = min(L, max_sym)
                            for a in range(L):
                                b_start = a + min_sym
                                if b_start > L:
                                    continue
                                for b in range(b_start, min(L, a + max_b) + 1):
                                    tok = seg[a:b]
                                    freq[tok] = freq.get(tok, 0) + 1
                            k = r
                        i = j
                        continue
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

    def _baseline_vocab(
        self,
        additional_special_tokens: Optional[Sequence[str]],
        *,
        char_vocab: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
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
        if char_vocab:
            for ch in char_vocab:
                if ch not in vocab:
                    vocab[ch] = idx
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
        char_vocab: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
        """Greedy gain–distortion selection.

        Gain is approximated with an analytic upper bound using frequency and token length:
            g(c) ≈ freq(c) * (len(c) - 1)
        which captures the max possible character saving if c replaces its characters.

        This keeps the trainer lightweight and deterministic, while still providing
        a meaningful rate signal.
        """

        vocab = self._baseline_vocab(additional_special_tokens, char_vocab=char_vocab)
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
        """Write a *data-only* artifact directory.

        Security note
        -------------
        Earlier versions wrote Python modules into the artifact directory and
        relied on Transformers' `trust_remote_code=True` autoload mechanism.
        This is convenient but undesirable in many production/security settings.

        CIT artifacts are now *pure data*: loading never executes code from the
        artifact. The runtime implementation is delivered via the installed
        `cit_tokenizers` Python package (i.e., your environment / dependency
        manager), and users load with:

            from cit_tokenizers.tokenization_cit import CITTokenizer
            tok = CITTokenizer.from_pretrained(outdir)
        """

        # Core artifact JSON (used by CITRuntime)
        (outdir / "cit_artifact.json").write_text(art.dumps(), encoding="utf-8")

        # Optional: a minimal tokenizer_config.json for model_max_length defaults.
        # (No `auto_map` and no embedded Python source files.)
        (outdir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "model_max_length": 512,
                    "padding_side": "right",
                    "truncation_side": "right",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Standard HF special token map (data-only).
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

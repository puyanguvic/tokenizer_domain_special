"""Tokenizer artifact builders/loaders for experiments.

This module is intentionally lightweight and treats tokenizer artifacts as directories
containing either:
- ctok artifacts: vocab.json + rules.json (+ manifest.json)
- tokenizers artifacts: tokenizer.json (+ manifest.json)
- byte baseline artifacts: manifest.json only (type=byte_utf8)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ctok.induction.candidates import collect_ngrams, collect_ngrams_with_labels
from ctok.induction.distortion import NullDistortion, build_label_entropy_distortion
from ctok.induction.greedy import greedy_select
from ctok.tokenization.boundary import DEFAULT_BOUNDARY_CHARS, normalize_boundary_chars
from ctok.tokenization.rules import RuleSet
from ctok.tokenization.tokenizer import CtokTokenizer
from ctok.tokenization.vocab import Vocabulary
from ctok.utils.hashing import sha256_file
from ctok.utils.serialization import read_json, write_json


class TokenizerAdapter(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def tokenize(self, text: str) -> list[str]: ...


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    kind: str
    vocab_size: int
    boundary_mode: str = "none"
    boundary_chars: set[str] | None = None
    base_charset: str = "byte"
    min_freq: int = 2
    min_len: int = 2
    max_len: int = 8
    lambda_weight: float = 0.0
    special_tokens: dict[str, str] | None = None


class ByteUtf8Tokenizer:
    """UTF-8 byte tokenizer baseline (each byte is a token id 0..255)."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def tokenize(self, text: str) -> list[str]:
        return [f"{byte:02x}" for byte in self.encode(text)]


class TokenizersTokenizerAdapter:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def tokenize(self, text: str) -> list[str]:
        return list(self._tokenizer.encode(text).tokens)


def build_tokenizer_artifact(
    spec: TokenizerSpec,
    *,
    corpus: list[str],
    labels: list[str] | None,
    output_dir: Path,
    force: bool = False,
) -> None:
    """Build a tokenizer artifact on disk if it doesn't already exist."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists() and not force:
        return

    kind = spec.kind.lower()
    if kind in {"byte", "byte_utf8"}:
        _build_byte_baseline(spec, output_dir=output_dir)
        return
    if kind in {"ctok", "controlled", "boundary_heal"}:
        _build_ctok_like(spec, corpus=corpus, labels=labels, output_dir=output_dir)
        return
    if kind in {"bpe", "unigram"}:
        _build_tokenizers_model(spec, corpus=corpus, output_dir=output_dir)
        return
    raise ValueError(f"Unsupported tokenizer kind: {spec.kind}")


def load_tokenizer_artifact(artifact_dir: Path) -> TokenizerAdapter:
    manifest_path = artifact_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = read_json(manifest_path)

    if (artifact_dir / "vocab.json").exists():
        return CtokTokenizer.from_pretrained(artifact_dir)

    tokenizer_json = artifact_dir / "tokenizer.json"
    if tokenizer_json.exists():
        tokenizers = _require_tokenizers()
        tok = tokenizers.Tokenizer.from_file(str(tokenizer_json))
        return TokenizersTokenizerAdapter(tok)

    if manifest.get("type") == "byte_utf8":
        return ByteUtf8Tokenizer()

    raise FileNotFoundError(f"Unrecognized tokenizer artifact layout: {artifact_dir}")


def _build_byte_baseline(spec: TokenizerSpec, *, output_dir: Path) -> None:
    manifest = {
        "format_version": 1,
        "type": "byte_utf8",
        "name": spec.name,
        "vocab_size": 256,
        "metadata": {
            "note": "UTF-8 byte baseline (no learned vocab).",
        },
        "files": {},
    }
    write_json(output_dir / "manifest.json", manifest)


def _build_ctok_like(
    spec: TokenizerSpec,
    *,
    corpus: list[str],
    labels: list[str] | None,
    output_dir: Path,
) -> None:
    if not corpus:
        raise ValueError("Corpus is empty.")

    boundary_chars = set(spec.boundary_chars or set())
    boundary_mode = spec.boundary_mode
    if spec.kind.lower() == "boundary_heal":
        boundary_mode = "aware"
        boundary_chars = boundary_chars or DEFAULT_BOUNDARY_CHARS

    base_charset = spec.base_charset.lower()
    if base_charset == "byte":
        base_vocab = [chr(i) for i in range(256)]
        # Ensure we can encode any unicode characters present in the corpus without
        # requiring an explicit byte->unicode transform.
        extra_chars: set[str] = set()
        for line in corpus:
            extra_chars.update(line)
        for ch in sorted(extra_chars):
            if ch not in base_vocab:
                base_vocab.append(ch)
    elif base_charset == "corpus":
        charset: set[str] = set()
        for line in corpus:
            charset.update(line)
        base_vocab = sorted(charset)
    else:
        raise ValueError(f"Unsupported base_charset: {spec.base_charset}")

    if boundary_mode == "aware":
        chars = boundary_chars or DEFAULT_BOUNDARY_CHARS
        for ch in sorted(chars):
            if ch not in base_vocab:
                base_vocab.append(ch)

    if len(base_vocab) > spec.vocab_size:
        raise ValueError("Base charset exceeds requested vocab size.")

    if boundary_mode == "aware":
        boundary_chars = boundary_chars or DEFAULT_BOUNDARY_CHARS
    else:
        boundary_chars = set()

    distortion = NullDistortion()
    candidates: dict[str, int]
    if labels is not None:
        candidates, label_counts = collect_ngrams_with_labels(
            corpus,
            labels,
            min_len=spec.min_len,
            max_len=spec.max_len,
            min_freq=spec.min_freq,
            boundary_chars=boundary_chars if boundary_mode == "aware" else None,
        )
        distortion = build_label_entropy_distortion(label_counts)
    else:
        candidates = collect_ngrams(
            corpus,
            min_len=spec.min_len,
            max_len=spec.max_len,
            min_freq=spec.min_freq,
            boundary_chars=boundary_chars if boundary_mode == "aware" else None,
        )

    budget = max(spec.vocab_size - len(base_vocab), 0)
    selected = greedy_select(
        candidates=candidates,
        budget=budget,
        lambda_weight=spec.lambda_weight,
        distortion=distortion,
    )

    special_tokens = dict(spec.special_tokens or {})
    # Ensure runtime can always represent unknown characters.
    special_tokens.setdefault("unk", "<unk>")
    merged_tokens = list(base_vocab) + selected
    for tok in special_tokens.values():
        if tok not in merged_tokens:
            merged_tokens.append(tok)

    vocab = Vocabulary(tokens=merged_tokens, special_tokens=special_tokens)
    rules = RuleSet.from_vocab(vocab)
    boundary_chars_list = sorted(boundary_chars) if boundary_mode == "aware" else None
    tokenizer = CtokTokenizer(
        vocab=vocab,
        rules=rules,
        special_tokens=special_tokens or None,
        boundary_mode=boundary_mode,
        boundary_chars=boundary_chars_list,
    )

    metadata: dict[str, object] = {
        "name": spec.name,
        "type": spec.kind,
        "vocab_size": spec.vocab_size,
        "base_charset": spec.base_charset,
        "min_freq": spec.min_freq,
        "min_len": spec.min_len,
        "max_len": spec.max_len,
        "lambda_weight": spec.lambda_weight,
        "boundary_mode": boundary_mode,
        "boundary_chars": boundary_chars_list,
        "labels_provided": labels is not None,
    }
    tokenizer.save_pretrained(output_dir, metadata=metadata)


def _build_tokenizers_model(
    spec: TokenizerSpec,
    *,
    corpus: list[str],
    output_dir: Path,
) -> None:
    tokenizers = _require_tokenizers()
    from tokenizers import Tokenizer
    from tokenizers.models import BPE, Unigram
    from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
    from tokenizers.trainers import BpeTrainer, UnigramTrainer

    if not corpus:
        raise ValueError("Corpus is empty.")

    boundary_chars = set(spec.boundary_chars or set())
    if spec.boundary_mode == "aware" and not boundary_chars:
        boundary_chars = DEFAULT_BOUNDARY_CHARS

    unk_token = (spec.special_tokens or {}).get("unk", "<unk>")
    initial_alphabet = ByteLevel.alphabet()

    if spec.kind.lower() == "bpe":
        model = BPE(unk_token=unk_token)
        trainer = BpeTrainer(
            vocab_size=spec.vocab_size,
            min_frequency=spec.min_freq,
            special_tokens=sorted(set((spec.special_tokens or {}).values())),
            initial_alphabet=initial_alphabet,
        )
    else:
        model = Unigram()
        trainer = UnigramTrainer(
            vocab_size=spec.vocab_size,
            special_tokens=sorted(set((spec.special_tokens or {}).values())),
            initial_alphabet=initial_alphabet,
            unk_token=unk_token,
        )

    tokenizer = Tokenizer(model)
    pre_tokenizers = []
    if spec.boundary_mode == "aware" and boundary_chars:
        pattern = "[" + _re_escape_chars(boundary_chars) + "]"
        pre_tokenizers.append(Split(pattern, behavior="isolated"))
    pre_tokenizers.append(ByteLevel(add_prefix_space=False))
    tokenizer.pre_tokenizer = Sequence(pre_tokenizers)

    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    manifest = {
        "format_version": 1,
        "type": spec.kind,
        "name": spec.name,
        "vocab_size": spec.vocab_size,
        "metadata": {
            "min_freq": spec.min_freq,
            "boundary_mode": spec.boundary_mode,
            "boundary_chars": sorted(boundary_chars) if boundary_chars else None,
            "special_tokens": dict(spec.special_tokens or {}),
        },
        "files": {
            "tokenizer.json": sha256_file(tokenizer_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)


def _re_escape_chars(chars: set[str]) -> str:
    import re

    return re.escape("".join(sorted(chars)))


def _require_tokenizers() -> Any:
    try:
        import tokenizers  # type: ignore
    except ImportError as exc:
        raise ImportError("Install 'tokenizers' (via transformers) to build baselines.") from exc
    return tokenizers

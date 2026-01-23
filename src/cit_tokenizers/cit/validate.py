from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .runtime import CITArtifact


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


def validate_typed_symbol_integrity(vocab_tokens: Iterable[str], typed_symbols: Sequence[str]) -> list[ValidationIssue]:
    """Validate that vocab does not contain risky substrings of typed symbols.

    We only flag substrings that include the angle-bracket delimiters, which can
    interfere with atomic matching of typed symbols. Single-character tokens and
    plain alphanumerics (e.g., "IPV", "HASH") are ignored to avoid false positives
    when character vocab/fallback is enabled.
    """
    vocab = set(vocab_tokens)
    issues: list[ValidationIssue] = []
    for sym in typed_symbols:
        for tok in vocab:
            if tok == sym:
                continue
            if not tok or len(tok) < 2:
                continue
            if "<" not in tok and ">" not in tok:
                continue
            if tok in sym:
                issues.append(
                    ValidationIssue(
                        code="typed_symbol_substring",
                        message=f"Vocab token '{tok}' is a proper substring of typed symbol '{sym}'.",
                    )
                )
                # don't spam too much
                break
    return issues


_LONG_HEX_TOKEN_RE = re.compile(r"(?i)^[0-9a-f]{16,}$")
_B64_TOKEN_RE = re.compile(r"^(?:[A-Za-z0-9+/]{24,}={0,2})$")


def validate_high_entropy_contamination(
    vocab_tokens: Iterable[str],
    *,
    typed_symbols: Sequence[str],
    max_long_hex_fraction: float = 0.005,
    max_long_hex_count: int = 32,
    max_b64_fraction: float = 0.002,
    max_b64_count: int = 16,
) -> list[ValidationIssue]:
    """Heuristically detect vocab budget being consumed by high-entropy value fragments.

    Guardrail for structured/security corpora where raw hashes/hex/base64 blobs should be
    normalized into typed symbols (e.g. <HEX>, <HASH>, <B64>) before vocabulary induction.
    """
    vocab = [t for t in vocab_tokens if t]
    typed_set = set(typed_symbols)

    def _is_exempt(tok: str) -> bool:
        # Treat typed and special tokens as exempt from entropy checks.
        if tok in typed_set:
            return True
        if tok.startswith("<") and tok.endswith(">"):
            return True
        if tok.startswith("[") and tok.endswith("]"):
            return True
        return False

    long_hex = [t for t in vocab if (not _is_exempt(t)) and _LONG_HEX_TOKEN_RE.match(t)]
    b64 = [t for t in vocab if (not _is_exempt(t)) and _B64_TOKEN_RE.match(t)]

    n = max(1, len(vocab))
    issues: list[ValidationIssue] = []

    if (len(long_hex) > max_long_hex_count) or (len(long_hex) / n > max_long_hex_fraction):
        sample = ", ".join(long_hex[:5])
        issues.append(
            ValidationIssue(
                code="entropy_hex_overflow",
                message=(
                    f"Vocab contains {len(long_hex)} long-hex tokens (>=16 chars), "
                    f"which is {len(long_hex)/n:.2%} of vocab. "
                    f"Sample: {sample}. "
                    "This usually indicates typed hygiene did not fully normalize hex/hash values."
                ),
            )
        )

    if (len(b64) > max_b64_count) or (len(b64) / n > max_b64_fraction):
        sample = ", ".join(b64[:5])
        issues.append(
            ValidationIssue(
                code="entropy_b64_overflow",
                message=(
                    f"Vocab contains {len(b64)} base64-like tokens (>=24 chars), "
                    f"which is {len(b64)/n:.2%} of vocab. "
                    f"Sample: {sample}. "
                    "This usually indicates typed hygiene did not fully normalize base64 blobs."
                ),
            )
        )

    # Stronger signal: typed symbols exist, but many raw tokens remain.
    if ("<HEX>" in typed_set or "<HASH>" in typed_set) and len(long_hex) > 5:
        issues.append(
            ValidationIssue(
                code="typed_hygiene_leak_hex",
                message=(
                    "Typed symbols include <HEX>/<HASH>, but vocab still contains many raw long-hex tokens. "
                    "Consider tightening regex boundaries or ensuring hygiene runs before candidate extraction."
                ),
            )
        )
    if ("<B64>" in typed_set) and len(b64) > 3:
        issues.append(
            ValidationIssue(
                code="typed_hygiene_leak_b64",
                message=(
                    "Typed symbols include <B64>, but vocab still contains many raw base64-like tokens. "
                    "Consider tightening regex boundaries or ensuring hygiene runs before candidate extraction."
                ),
            )
        )

    return issues


def validate_artifact(
    art: CITArtifact,
    typed_symbols: Sequence[str],
    *,
    max_long_hex_fraction: float = 0.005,
    max_long_hex_count: int = 32,
    max_b64_fraction: float = 0.002,
    max_b64_count: int = 16,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(validate_typed_symbol_integrity(art.vocab.keys(), typed_symbols))
    issues.extend(
        validate_high_entropy_contamination(
            art.vocab.keys(),
            typed_symbols=typed_symbols,
            max_long_hex_fraction=max_long_hex_fraction,
            max_long_hex_count=max_long_hex_count,
            max_b64_fraction=max_b64_fraction,
            max_b64_count=max_b64_count,
        )
    )
    return issues

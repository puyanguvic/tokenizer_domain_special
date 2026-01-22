from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

# NOTE: patterns are intentionally conservative; adjust as needed per-domain.
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
# Basic IPv6 (not fully exhaustive, but practical)
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b")
PORT_RE = re.compile(r"(?<!\d):([0-9]{1,5})(?!\d)")
ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b")
EPOCH_TS_RE = re.compile(r"\b\d{10,13}\b")
HASH_HEX_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
HEX_BLOB_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")
B64_RE = re.compile(r"\b(?:[A-Za-z0-9+/]{24,}={0,2})\b")

# Percent-encoding triplets in URLs / payloads.
# We normalize to uppercase and insert a boundary after each triplet when it is
# immediately followed by an alphanumeric character. This prevents mixed tokens
# like '%3Cscript' or '%3Ealert'.
PCT_TRIPLET_RE = re.compile(r"%([0-9a-fA-F]{2})")
PCT_TRIPLET_NEEDS_BOUNDARY_RE = re.compile(r"%([0-9a-fA-F]{2})(?=[A-Za-z0-9])")

def _normalize_percent_triplets(x: str) -> str:
    # Uppercase all %hh triplets first.
    x = PCT_TRIPLET_RE.sub(lambda m: f"%{m.group(1).upper()}", x)
    # Insert a boundary after %HH if the next char is alphanumeric, so we don't
    # end up with combined tokens like '%3Cscript'.
    x = PCT_TRIPLET_NEEDS_BOUNDARY_RE.sub(lambda m: f"%{m.group(1).upper()} ", x)
    return x

def _repl_b64(m: re.Match) -> str:
    """Heuristic base64 typing to reduce false positives on ASCII words (e.g., SQL keywords)."""
    s = m.group(0)

    # If the match is near URL-encoded material, avoid typing as <B64>.
    # URL-encoded attack payloads often create long "word-like" spans that would
    # otherwise be mistaken for base64.
    src = m.string
    a = max(0, m.start() - 3)
    b = min(len(src), m.end() + 3)
    if "%" in src[a:b]:
        return s

    # Require some character diversity typical for base64 blobs:
    # - digits or '+'/'/' or '=' or lowercase letters
    if re.search(r"[0-9+/=a-z]", s) is None:
        return s
    # Avoid typing ALL-CAPS keyword-y spans (common in payloads like UNION/SELECT/NULL),
    # even if they contain '+' or '/' which may appear in URL-encoded attacks.
    stripped = re.sub(r"[+/=_-]", "", s)
    if re.fullmatch(r"[A-Z]+", stripped) is not None:
        return s
    return "<B64>"


# HTTP-aware patterns (reduce false positives like rv:92.0 in User-Agent)
URL_AUTH_PORT_RE = re.compile(r"(?i)((?:(?:https?|wss?)://)?[A-Za-z0-9.-]+):([0-9]{1,5})(?=[/\s])")
HOST_HEADER_PORT_RE = re.compile(r"(?im)^(Host:\s*[^:\s]+):([0-9]{1,5})\s*$")
# Loose version number (e.g., 92.0, 1.1, 10.0.19045) – handled only in HTTP mode to avoid over-typing.
VERSION_RE = re.compile(r"(?<!\d)(?:\d+\.){1,3}\d+(?!\d)")

def apply_typed_hygiene_http(
    text: str,
    enable_numeric_buckets: bool = True,
    long_num_min_digits: int = 6,
    enable_version_token: bool = True,
) -> str:
    """HTTP-aware variant of :func:`apply_typed_hygiene`.

    Differences vs generic:
      * PORT is only typed in URL authority or Host header, avoiding false positives
        such as 'rv:92.0' in User-Agent.
      * Optionally types dotted version numbers as <VER>.
    """
    # Normalize percent-encoding early to avoid mixed tokens like '%3Cscript'.
    x = _normalize_percent_triplets(text)
    x = UUID_RE.sub("<UUID>", x)
    x = IPV6_RE.sub("<IPV6>", x)
    x = IPV4_RE.sub("<IPV4>", x)
    x = ISO_TS_RE.sub("<TS>", x)
    x = EPOCH_TS_RE.sub("<TS>", x)
    x = HASH_HEX_RE.sub("<HASH>", x)
    x = HEX_BLOB_RE.sub("<HEX>", x)
    x = B64_RE.sub(_repl_b64, x)

    if enable_version_token:
        # Do NOT type versions inside already-typed symbols
        x = VERSION_RE.sub("<VER>", x)

    # ports: only in authority / Host header
    x = URL_AUTH_PORT_RE.sub(lambda m: f"{m.group(1)}:<PORT>", x)
    # Host header line (keeps colon)
    x = HOST_HEADER_PORT_RE.sub(lambda m: f"{m.group(1)}:<PORT>", x)

    if enable_numeric_buckets:
        def repl(m):
            s = m.group(0)
            if s.startswith("<") and s.endswith(">"):
                return s
            return _bucket_long_int(s, long_num_min_digits)
        x = LONG_NUM_RE.sub(repl, x)
    return x


LONG_NUM_RE = re.compile(r"\b\d+\b")

def _bucket_long_int(s: str, min_digits: int) -> str:
    # bucket only if all digits and long enough
    if len(s) < min_digits:
        return s
    # avoid bucketing timestamps already caught: caller orders patterns
    k = min(12, max(0, len(s) - 1))  # approx log10 by digits-1
    return f"<NUM_POW10_{k}>"

def apply_typed_hygiene(
    text: str,
    enable_numeric_buckets: bool = True,
    long_num_min_digits: int = 6,
) -> str:
    """Apply deterministic typed replacements.

    Priority order mirrors the paper:
    UUID -> IP -> TS -> HASH -> HEX -> B64 -> PORT -> LONGNUM(bucket).
    """
    x = text
    x = UUID_RE.sub("<UUID>", x)
    x = IPV6_RE.sub("<IPV6>", x)
    x = IPV4_RE.sub("<IPV4>", x)
    # timestamps
    x = ISO_TS_RE.sub("<TS>", x)
    # epoch timestamps: keep but avoid over-replacing short numbers by requiring 10-13 digits
    x = EPOCH_TS_RE.sub("<TS>", x)
    # hashes/blobs
    x = HASH_HEX_RE.sub("<HASH>", x)
    x = HEX_BLOB_RE.sub("<HEX>", x)
    x = B64_RE.sub(_repl_b64, x)
    # ports: keep the colon to preserve typical structure
    x = PORT_RE.sub(lambda m: f":<PORT>", x)

    if enable_numeric_buckets:
        def repl(m):
            s = m.group(0)
            # don't bucket if it's already typed
            if s.startswith("<") and s.endswith(">"):
                return s
            return _bucket_long_int(s, long_num_min_digits)
        x = LONG_NUM_RE.sub(repl, x)
    return x
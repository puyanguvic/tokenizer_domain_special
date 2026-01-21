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
    x = B64_RE.sub("<B64>", x)
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

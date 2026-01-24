from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Iterable, Union
import json

from .json_serialize import serialize_json_record
from .hygiene import apply_typed_hygiene, apply_typed_hygiene_http
from .http_struct import structure_http

@dataclass(frozen=True)
class ContractConfig:
    # High-level toggles
    enable_json_serialization: bool = True
    enable_typed_hygiene: bool = True

    # JSON serialization
    json_field_order: Optional[list[str]] = None
    json_emit_missing: bool = True

    # Typed hygiene
    enable_numeric_buckets: bool = True
    typed_hygiene_mode: str = "generic"  # 'generic' or 'http'
    enable_version_token: bool = True
    long_num_min_digits: int = 6

    # Structured HTTP preprocessing
    structured_input_mode: str = "none"  # 'none' or 'http'/'waf'
    structured_max_len: int = 4096

    # Case normalization
    #
    # Notes:
    # - For many security/structured domains, case can be semantically meaningful
    #   (e.g., URL paths on some servers, base64/hex blobs, user-provided IDs).
    # - Some fields are effectively case-insensitive (e.g., HTTP header names).
    #
    # We therefore keep the default as "none" and expose an optional
    # ASCII-only lowercasing mode.
    lowercase_mode: str = "none"  # 'none' or 'ascii'

    # Boundary markers / separators
    key_prefix: str = "<K:"
    key_suffix: str = ">"
    sep_token: str = "<SEP>"
    eol_token: str = "<EOL>"
    empty_token: str = "<EMPTY>"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "ContractConfig":
        return ContractConfig(**json.loads(s))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ContractConfig":
        return ContractConfig(**data)

class Contract:
    """Deterministic interface contract applied at build and runtime.

    Contract(X) returns a string X' that:
      (i) optionally serializes JSON records into role-explicit form;
      (ii) optionally applies typed hygiene for high-cardinality patterns.
    """

    def __init__(self, cfg: ContractConfig):
        self.cfg = cfg

    @property
    def config(self) -> ContractConfig:
        return self.cfg

    def apply(self, x: Union[str, Dict[str, Any]]) -> str:
        # Optional structured parsing (HTTP/WAF-like inputs).
        if (self.cfg.structured_input_mode or "none").lower() in ("http", "waf"):
            x = structure_http(
                x,
                sep_token=self.cfg.sep_token,
                eol_token=self.cfg.eol_token,
                key_prefix=self.cfg.key_prefix,
                key_suffix=self.cfg.key_suffix,
                empty_token=self.cfg.empty_token,
                long_num_min_digits=self.cfg.long_num_min_digits,
                max_len=int(self.cfg.structured_max_len),
            )

        # JSON dict -> string serialization (role markers + separators)
        if self.cfg.enable_json_serialization and isinstance(x, dict):
            x = serialize_json_record(
                x,
                field_order=self.cfg.json_field_order,
                emit_missing=self.cfg.json_emit_missing,
                key_prefix=self.cfg.key_prefix,
                key_suffix=self.cfg.key_suffix,
                sep_token=self.cfg.sep_token,
                eol_token=self.cfg.eol_token,
                empty_token=self.cfg.empty_token,
            )
        elif not isinstance(x, str):
            # Fallback: best-effort JSON dumps for structured objects
            x = json.dumps(x, ensure_ascii=False, separators=(",", ":"))

        # Typed hygiene on strings
        if self.cfg.enable_typed_hygiene:
            if (self.cfg.typed_hygiene_mode or "generic").lower() in ("http", "waf"):
                x = apply_typed_hygiene_http(
                    x,
                    enable_numeric_buckets=self.cfg.enable_numeric_buckets,
                    long_num_min_digits=self.cfg.long_num_min_digits,
                    enable_version_token=self.cfg.enable_version_token,
                )
            else:
                x = apply_typed_hygiene(
                    x,
                    enable_numeric_buckets=self.cfg.enable_numeric_buckets,
                    long_num_min_digits=self.cfg.long_num_min_digits,
                )

        # Optional ASCII-only case normalization.
        if self.cfg.lowercase_mode == "ascii":
            # Avoid locale-dependent behavior; only map A-Z -> a-z.
            x = "".join((chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch) for ch in x)
        elif self.cfg.lowercase_mode not in ("none", ""):
            raise ValueError(f"Unknown lowercase_mode={self.cfg.lowercase_mode!r}")
        return x

    def apply_many(self, xs: Iterable[Union[str, Dict[str, Any]]]) -> Iterable[str]:
        for x in xs:
            yield self.apply(x)

    def typed_symbols(self) -> list[str]:
        if not self.cfg.enable_typed_hygiene:
            return []
        symbols = ["<UUID>", "<IPV6>", "<IPV4>", "<TS>", "<HASH>", "<HEX>", "<B64>", "<PORT>", "<VER>"]
        if self.cfg.enable_numeric_buckets:
            start = max(0, self.cfg.long_num_min_digits - 1)
            symbols.extend([f"<NUM_POW10_{k}>" for k in range(start, 13)])
        return symbols

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Iterable, Union
import json

from .json_serialize import serialize_json_record
from .hygiene import apply_typed_hygiene

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
    long_num_min_digits: int = 6

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

class Contract:
    """Deterministic interface contract applied at build and runtime.

    Contract(X) returns a string X' that:
      (i) optionally serializes JSON records into role-explicit form;
      (ii) optionally applies typed hygiene for high-cardinality patterns.
    """

    def __init__(self, cfg: ContractConfig):
        self.cfg = cfg

    def apply(self, x: Union[str, Dict[str, Any]]) -> str:
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
            x = apply_typed_hygiene(
                x,
                enable_numeric_buckets=self.cfg.enable_numeric_buckets,
                long_num_min_digits=self.cfg.long_num_min_digits,
            )
        return x

    def apply_many(self, xs: Iterable[Union[str, Dict[str, Any]]]) -> Iterable[str]:
        for x in xs:
            yield self.apply(x)

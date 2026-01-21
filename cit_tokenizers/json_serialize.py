from __future__ import annotations
from typing import Any, Dict, Optional
import json

def _encode_value(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        # Keep numbers as-is; typed hygiene may bucket later
        return str(v)
    if isinstance(v, str):
        return v
    # nested objects/arrays -> compact json
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

def serialize_json_record(
    obj: Dict[str, Any],
    field_order: Optional[list[str]] = None,
    emit_missing: bool = True,
    key_prefix: str = "<K:",
    key_suffix: str = ">",
    sep_token: str = "<SEP>",
    eol_token: str = "<EOL>",
    empty_token: str = "<EMPTY>",
) -> str:
    """Deterministic role-explicit serialization.

    Output format per field:
      <K:field> <SEP> value <EOL>
    """
    if field_order is None:
        # deterministic: lexicographic keys
        keys = sorted(obj.keys())
    else:
        keys = list(field_order)

    parts = []
    for k in keys:
        if k in obj:
            v = _encode_value(obj[k])
            if v == "":
                v = empty_token
        else:
            if not emit_missing:
                continue
            v = empty_token
        parts.append(f"{key_prefix}{k}{key_suffix} {sep_token} {v} {eol_token}")
    return " ".join(parts)

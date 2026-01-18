from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Set


@dataclass(frozen=True)
class HygienePattern:
    name: str
    pattern: str
    replacement: str
    flags: int = 0

    def compile(self) -> Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass
class HygieneConfig:
    enabled: bool = True
    version: str = "hygiene-v1"
    typed_tokens: List[str] = field(default_factory=list)
    patterns: List[HygienePattern] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "version": self.version,
            "typed_tokens": list(self.typed_tokens),
            "patterns": [
                {
                    "name": p.name,
                    "pattern": p.pattern,
                    "replacement": p.replacement,
                    "flags": p.flags,
                }
                for p in self.patterns
            ],
        }

    @staticmethod
    def from_dict(data: Dict[str, object]) -> "HygieneConfig":
        patterns = []
        for p in data.get("patterns", []):
            patterns.append(
                HygienePattern(
                    name=str(p.get("name", "")),
                    pattern=str(p.get("pattern", "")),
                    replacement=str(p.get("replacement", "")),
                    flags=int(p.get("flags", 0)),
                )
            )
        return HygieneConfig(
            enabled=bool(data.get("enabled", True)),
            version=str(data.get("version", "hygiene-v1")),
            typed_tokens=list(data.get("typed_tokens", [])),
            patterns=patterns,
        )


def default_hygiene_config() -> HygieneConfig:
    typed_tokens = [
        "<IPV4>",
        "<PORT>",
        "<IPV6>",
        "<UUID>",
        "<HEX>",
        "<B64>",
        "<TS>",
        "<BLKID>",
        "<PART>",
        "<JOBID>",
        "<ATTEMPT>",
        "<TASK>",
        "<INTZ>",
        "<INT3>",
        "<INT4>",
        "<INT5P>",
    ]
    patterns = [
        HygienePattern("ipv4_port", r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b", "<IPV4>:<PORT>"),
        HygienePattern("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<IPV4>"),
        HygienePattern("ipv6", r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{0,4}\b", "<IPV6>"),
        HygienePattern(
            "uuid",
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
            "<UUID>",
        ),
        HygienePattern("hex", r"\b[0-9a-fA-F]{16,}\b", "<HEX>"),
        HygienePattern("b64", r"\b[A-Za-z0-9+/]{24,}={0,2}\b", "<B64>"),
        HygienePattern(
            "iso_ts",
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b",
            "<TS>",
        ),
        HygienePattern("epoch_ts", r"\b\d{10}(?:\d{3})?\b", "<TS>"),
        HygienePattern("blkid", r"\bblk_-?\d+\b", "<BLKID>"),
        HygienePattern("part", r"/part-\d+\b", "/<PART>"),
        HygienePattern("attempt", r"\battempt_\d+\b", "<ATTEMPT>"),
        HygienePattern("task", r"\b_task_\d+\b", "<TASK>"),
        HygienePattern("jobid", r"\b(?:job|container)_[A-Za-z0-9_]+(?:-\d+)?\b", "<JOBID>"),
        HygienePattern("intz", r"\b0\d{2,}\b", "<INTZ>"),
        HygienePattern("int3", r"\b\d{3}\b", "<INT3>"),
        HygienePattern("int4", r"\b\d{4}\b", "<INT4>"),
        HygienePattern("int5p", r"\b\d{5,}\b", "<INT5P>"),
    ]
    return HygieneConfig(enabled=True, typed_tokens=typed_tokens, patterns=patterns)


def apply_hygiene(text: str, cfg: HygieneConfig) -> str:
    if not cfg.enabled:
        return text
    out = text
    for p in cfg.patterns:
        out = p.compile().sub(p.replacement, out)
    return out


def is_value_fragment(token: str) -> bool:
    if not token:
        return False
    if token.startswith("<") and token.endswith(">"):
        return False
    digits = sum(1 for c in token if c.isdigit())
    digit_ratio = digits / max(len(token), 1)
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){1,3}", token):
        return True
    if re.fullmatch(r"[0-9a-fA-F]{8,}", token):
        return True
    if re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", token):
        return True
    if digit_ratio >= 0.5 and any(ch in token for ch in ".:/_-"):
        return True
    return False


def is_typed_token_fragment(token: str, typed_tokens: Iterable[str]) -> bool:
    if not token:
        return False
    typed = list(typed_tokens)
    if token in typed:
        return False
    if "<" in token or ">" in token:
        return True
    for t in typed:
        if token in t and token != t:
            return True
    return False


def junk_score(token: str) -> float:
    if not token:
        return 0.0
    digits = sum(1 for c in token if c.isdigit())
    digit_ratio = digits / max(len(token), 1)
    unique_ratio = len(set(token)) / max(len(token), 1)
    frag = 1.0 if is_value_fragment(token) else 0.0
    return 0.7 * digit_ratio + 0.3 * unique_ratio + frag


def vocab_hygiene_metrics(tokens: Iterable[str], typed_tokens: Iterable[str]) -> Dict[str, float]:
    tok_list = list(tokens)
    total = max(len(tok_list), 1)
    typed = set(typed_tokens)
    value_cnt = sum(1 for t in tok_list if is_value_fragment(t))
    typed_cnt = sum(1 for t in tok_list if t in typed)
    return {
        "vocab_value_frac": value_cnt / total,
        "vocab_typed_frac": typed_cnt / total,
    }


def ascii_base_chars() -> Set[str]:
    chars: Set[str] = set()
    for i in range(0x20, 0x7F):
        chars.add(chr(i))
    for ch in ["\t", "\n", "\r"]:
        chars.add(ch)
    return chars

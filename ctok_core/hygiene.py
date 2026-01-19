from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern, Set


ALLOWED_NUMBERS: Set[str] = {
    # HTTP status codes
    "100",
    "101",
    "102",
    "103",
    "200",
    "201",
    "202",
    "203",
    "204",
    "205",
    "206",
    "207",
    "208",
    "226",
    "300",
    "301",
    "302",
    "303",
    "304",
    "305",
    "306",
    "307",
    "308",
    "400",
    "401",
    "402",
    "403",
    "404",
    "405",
    "406",
    "407",
    "408",
    "409",
    "410",
    "411",
    "412",
    "413",
    "414",
    "415",
    "416",
    "417",
    "418",
    "421",
    "422",
    "423",
    "424",
    "425",
    "426",
    "428",
    "429",
    "431",
    "451",
    "500",
    "501",
    "502",
    "503",
    "504",
    "505",
    "506",
    "507",
    "508",
    "510",
    "511",
    # Common ports
    "20",
    "21",
    "22",
    "23",
    "25",
    "53",
    "67",
    "68",
    "69",
    "80",
    "110",
    "111",
    "123",
    "135",
    "137",
    "138",
    "139",
    "143",
    "161",
    "162",
    "389",
    "443",
    "445",
    "465",
    "587",
    "636",
    "993",
    "995",
    "1433",
    "1521",
    "2049",
    "2375",
    "2376",
    "3306",
    "3389",
    "5432",
    "5671",
    "5672",
    "5900",
    "6379",
    "8080",
    "8081",
    "8443",
    "8883",
    "9000",
    "9200",
    "9300",
    "11211",
    "1883",
    "27017",
}


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

    # Lazily-compiled regex cache. Not serialized.
    _compiled: Optional[List[tuple[Pattern[str], str]]] = field(default=None, init=False, repr=False)

    def compiled_patterns(self) -> List[tuple[Pattern[str], str]]:
        """Return cached compiled patterns as (regex, replacement)."""
        if self._compiled is None:
            self._compiled = [(p.compile(), p.replacement) for p in self.patterns]
        return self._compiled

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
        "__IPV4__",
        "__PORT__",
        "__IPV6__",
        "__UUID__",
        "__HEX__",
        "__B64__",
        "__TS__",
        "__BLKID__",
        "__PART__",
        "__JOBID__",
        "__ATTEMPT__",
        "__TASK__",
        "__NUM3__",
        "__NUM4__",
        "__NUM5P__",
    ]
    patterns = [
        HygienePattern("ipv4_port", r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{1,5}\b", "__IPV4__:__PORT__"),
        HygienePattern("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "__IPV4__"),
        HygienePattern("ipv6", r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{0,4}\b", "__IPV6__"),
        HygienePattern(
            "uuid",
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b",
            "__UUID__",
        ),
        HygienePattern("hex", r"\b[0-9a-fA-F]{16,}\b", "__HEX__"),
        HygienePattern("b64_urlsafe", r"\b[A-Za-z0-9_-]{24,}={0,2}\b", "__B64__"),
        HygienePattern("b64_noslash", r"\b[A-Za-z0-9+]{24,}={0,2}\b", "__B64__"),
        HygienePattern(
            "iso_ts",
            r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b",
            "__TS__",
        ),
        HygienePattern("epoch_ts", r"\b\d{10}(?:\d{3})?\b", "__TS__"),
        HygienePattern("blkid", r"\bblk_-?\d+\b", "__BLKID__"),
        HygienePattern("part", r"/part-\d+\b", "/__PART__"),
        HygienePattern("attempt", r"\battempt_\d+\b", "__ATTEMPT__"),
        HygienePattern("task", r"\b_task_\d+\b", "__TASK__"),
        HygienePattern("jobid", r"\b(?:job|container)_[A-Za-z0-9_]+(?:-\d+)?\b", "__JOBID__"),
    ]
    return HygieneConfig(enabled=True, typed_tokens=typed_tokens, patterns=patterns)


def apply_hygiene(text: str, cfg: HygieneConfig) -> str:
    if not cfg.enabled:
        return text
    out = text
    for rx, repl in cfg.compiled_patterns():
        out = rx.sub(repl, out)
    out = normalize_numbers(out)
    return out


def normalize_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        s = match.group(0)
        if s in ALLOWED_NUMBERS:
            return s
        if len(s) <= 2:
            return s
        if len(s) == 3:
            return "__NUM3__"
        if len(s) == 4:
            return "__NUM4__"
        return "__NUM5P__"

    return re.sub(r"(?<!\d)\d+(?!\d)", repl, text)


def is_value_fragment(token: str) -> bool:
    if not token:
        return False
    if token.startswith("<") and token.endswith(">"):
        return False
    if token.startswith("__") and token.endswith("__"):
        return False
    if token.isdigit() and token in ALLOWED_NUMBERS:
        return False
    if re.fullmatch(r"\d{3,}", token):
        return True
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

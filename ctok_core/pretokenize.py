from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass(frozen=True)
class PreTokenizePattern:
    name: str
    pattern: str
    replacement: str
    flags: int = 0

    def compile(self) -> Pattern[str]:
        return re.compile(self.pattern, self.flags)


@dataclass
class PreTokenizerConfig:
    enabled: bool = True
    version: str = "pretokenize-v1"
    patterns: List[PreTokenizePattern] = field(default_factory=list)
    _compiled: Optional[List[Tuple[Pattern[str], str]]] = field(default=None, init=False, repr=False)

    def compiled_patterns(self) -> List[Tuple[Pattern[str], str]]:
        """Compile regex patterns once and reuse across calls."""
        if self._compiled is None:
            self._compiled = [(p.compile(), p.replacement) for p in self.patterns]
        return self._compiled


    def to_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "version": self.version,
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
    def from_dict(data: Dict[str, object]) -> "PreTokenizerConfig":
        patterns = []
        for p in data.get("patterns", []):
            patterns.append(
                PreTokenizePattern(
                    name=str(p.get("name", "")),
                    pattern=str(p.get("pattern", "")),
                    replacement=str(p.get("replacement", "")),
                    flags=int(p.get("flags", 0)),
                )
            )
        return PreTokenizerConfig(
            enabled=bool(data.get("enabled", True)),
            version=str(data.get("version", "pretokenize-v1")),
            patterns=patterns,
        )


def default_pretokenizer_config() -> PreTokenizerConfig:
    patterns = [
        PreTokenizePattern("html_comment_open", r"<!--", r" <!-- "),
        PreTokenizePattern("html_comment_close", r"-->", r" --> "),
        PreTokenizePattern("cdata_open", r"<!\[CDATA\[", r" <![CDATA[ "),
        PreTokenizePattern("cdata_close", r"\]\]>", r" ]]> "),
        PreTokenizePattern("pi_open", r"<\?", r" <? "),
        PreTokenizePattern("pi_close", r"\?>", r" ?> "),
        PreTokenizePattern("tag_close", r"</", r" </ "),
        PreTokenizePattern("tag_self_close", r"/>", r" /> "),
        PreTokenizePattern("percent_hex", r"(%[0-9A-Fa-f]{2})", r" \\1 "),
        PreTokenizePattern("path_traversal_fwd", r"\\.\\./", r" ../ "),
        PreTokenizePattern("path_traversal_back", r"\.\.\\", r" ..\\ "),
        PreTokenizePattern("double_slash", r"//", r" // "),
        PreTokenizePattern("double_backslash", r"\\\\", r" \\\\ "),
        PreTokenizePattern("logic_and", r"&&", r" && "),
        PreTokenizePattern("logic_or", r"\\|\\|", r" || "),
        PreTokenizePattern("cmp_eq", r"==", r" == "),
        PreTokenizePattern("cmp_ne", r"!=", r" != "),
        PreTokenizePattern("cmp_le", r"<=", r" <= "),
        PreTokenizePattern("cmp_ge", r">=", r" >= "),
        PreTokenizePattern("tag_open", r"<", r" < "),
        PreTokenizePattern("tag_end", r">", r" > "),
        PreTokenizePattern("sep_qmark", r"\\?", r" ? "),
        PreTokenizePattern("sep_amp", r"&", r" & "),
        PreTokenizePattern("sep_eq", r"=", r" = "),
        PreTokenizePattern("sep_colon", r":", r" : "),
        PreTokenizePattern("sep_semi", r";", r" ; "),
        PreTokenizePattern("sep_comma", r",", r" , "),
        PreTokenizePattern("sep_slash", r"/", r" / "),
        PreTokenizePattern("sep_backslash", r"(?<!\\)\\(?!\\)", " \\ "),
        PreTokenizePattern("sep_lparen", r"\(", r" ( "),
        PreTokenizePattern("sep_rparen", r"\)", r" ) "),
        PreTokenizePattern("sep_lbrack", r"\[", r" [ "),
        PreTokenizePattern("sep_rbrack", r"\]", r" ] "),
        PreTokenizePattern("sep_lbrace", r"\{", r" { "),
        PreTokenizePattern("sep_rbrace", r"\}", r" } "),
        PreTokenizePattern("sep_pipe", r"\\|", r" | "),
    ]
    return PreTokenizerConfig(enabled=True, patterns=patterns)


def apply_pretokenize(text: str, cfg: PreTokenizerConfig) -> str:
    if not cfg.enabled:
        return text
    out = text
    for pat, repl in cfg.compiled_patterns():
        out = pat.sub(repl, out)
    return out

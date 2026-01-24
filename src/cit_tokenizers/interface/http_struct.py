from __future__ import annotations

import json
import re
from typing import Any, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from .hygiene import (
    EPOCH_TS_RE,
    IPV4_RE,
    IPV6_RE,
    ISO_TS_RE,
    UUID_RE,
)

TAG_ONLY_RE = re.compile(r"^<[^>]+>$")
HEADER_RE = re.compile(r"^\s*([^:]+):\s*(.*)$")
URLENC_RE = re.compile(r"%[0-9a-fA-F]{2}")
HEX_RE = re.compile(r"^[0-9a-fA-F]+$")
_B64_CHARS_RE = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")
_HEX_RUN_RE = re.compile(r"[0-9a-fA-F]{16,}")
_B64_RUN_RE = re.compile(r"[A-Za-z0-9+/_-]{24,}={0,2}")

TOK_METHOD = "<METHOD>"
TOK_URL = "<URL>"
TOK_PROT = "<PROT>"
TOK_HDR = "<HDR>"
TOK_BODY = "<BODY>"
TOK_JSON = "<JSON>"
TOK_FORM = "<FORM>"
TOK_RAW = "<RAW>"
TOK_TRUNC = "<TRUNC>"
TOK_BYTES = "<BYTES>"
TOK_URLENC = "<URLENC>"
TOK_STR = "<STR>"
TOK_NUM = "<NUM>"
TOK_QK = "<QK>"
TOK_QVAL = "<QVAL>"
TOK_COOKIE_K = "<COOKIE_K>"
TOK_COOKIE_V = "<COOKIE_V>"
TOK_AUTH_BEARER = "<AUTH_BEARER>"
TOK_AUTH_BASIC = "<AUTH_BASIC>"


def _ascii_lower(s: str) -> str:
    return "".join((chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch) for ch in s)


def _has_non_printable(s: str) -> bool:
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in ("\t", "\n", "\r"):
            return True
        if o >= 127:
            return True
    return False


def _bucket_long_int(s: str, min_digits: int) -> str:
    if len(s) < min_digits:
        return TOK_NUM
    k = min(12, max(0, len(s) - 1))
    return f"<NUM_POW10_{k}>"


def _is_hash_token(tok: str) -> Optional[str]:
    if HEX_RE.fullmatch(tok) is None:
        return None
    n = len(tok)
    if n in (32, 40, 64):
        return f"<HASH_{n}>"
    if n >= 32:
        return "<HASH>"
    return None


def _is_hex_token(tok: str) -> Optional[str]:
    if HEX_RE.fullmatch(tok) is None:
        return None
    if len(tok) >= 16:
        if len(tok) in (16, 24, 32, 64):
            return f"<HEX_{len(tok)}>"
        return "<HEX>"
    return None


def _is_b64_token(tok: str) -> bool:
    if len(tok) < 24:
        return False
    if _B64_CHARS_RE.fullmatch(tok) is None:
        return False
    if "=" in tok[:-2]:
        return False
    if re.search(r"[0-9+/=a-z]", tok) is None:
        return False
    stripped = re.sub(r"[+/=_-]", "", tok)
    if stripped and re.fullmatch(r"[A-Z]+", stripped) is not None:
        return False
    return True


def value_placeholder(s: str, *, long_num_min_digits: int = 6) -> str:
    s = s.strip()
    if not s:
        return "<EMPTY>"
    if _has_non_printable(s):
        return TOK_BYTES
    if URLENC_RE.search(s):
        return TOK_URLENC
    if UUID_RE.fullmatch(s):
        return "<UUID>"
    if IPV4_RE.fullmatch(s):
        return "<IPV4>"
    if IPV6_RE.fullmatch(s):
        return "<IPV6>"
    if ISO_TS_RE.fullmatch(s) or EPOCH_TS_RE.fullmatch(s):
        return "<TS>"
    hash_tag = _is_hash_token(s)
    if hash_tag is not None:
        return hash_tag
    hex_tag = _is_hex_token(s)
    if hex_tag is not None:
        return hex_tag
    if _is_b64_token(s):
        return "<B64>"
    if s.isdigit():
        return _bucket_long_int(s, long_num_min_digits)
    return TOK_STR


def _split_kv_pairs(raw: str, sep: str = "&") -> Iterable[Tuple[str, str]]:
    for part in raw.split(sep):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        yield k, v


def _sanitize_key(key: str, *, max_len: int) -> List[str]:
    key = key.strip()
    if not key:
        return ["<EMPTY>"]
    if _has_non_printable(key):
        return [TOK_BYTES]
    if URLENC_RE.search(key):
        return [TOK_URLENC]
    if len(key) > max_len:
        return [key[:max_len], TOK_TRUNC]
    return [key]


def _sanitize_cookie_key(key: str, *, max_len: int) -> List[str]:
    key = key.strip()
    if not key:
        return ["<EMPTY>"]
    if _has_non_printable(key):
        return [TOK_BYTES]
    if URLENC_RE.search(key):
        key = URLENC_RE.sub(TOK_URLENC, key)
    if UUID_RE.search(key):
        key = UUID_RE.sub("<UUID>", key)

    def _repl_hex(m: re.Match) -> str:
        n = len(m.group(0))
        if n in (16, 24, 32, 64):
            return f"<HEX_{n}>"
        return "<HEX>"

    if _HEX_RUN_RE.search(key):
        key = _HEX_RUN_RE.sub(_repl_hex, key)
    if _B64_RUN_RE.search(key):
        key = _B64_RUN_RE.sub("<B64>", key)
    if len(key) > max_len:
        return [key[:max_len], TOK_TRUNC]
    return [key]


def _sanitize_value_tokens(
    value: str,
    *,
    keep_raw: bool,
    long_num_min_digits: int,
    max_len: int,
) -> List[str]:
    placeholder = value_placeholder(value, long_num_min_digits=long_num_min_digits)
    if not keep_raw:
        return [placeholder]
    if placeholder not in (TOK_STR, TOK_NUM):
        return [placeholder]
    if _has_non_printable(value):
        return [TOK_BYTES]
    if len(value) > max_len:
        return [value[:max_len], TOK_TRUNC]
    return [value]


def _tokens_from_query(query: str, *, long_num_min_digits: int, max_key_len: int, max_val_len: int) -> List[str]:
    tokens: List[str] = []
    for k, v in _split_kv_pairs(query, "&"):
        tokens.append(TOK_QK)
        tokens.extend(_sanitize_key(k, max_len=max_key_len))
        tokens.append(TOK_QVAL)
        tokens.extend(
            _sanitize_value_tokens(v, keep_raw=False, long_num_min_digits=long_num_min_digits, max_len=max_val_len)
        )
    return tokens


def _tokens_from_cookie(value: str, *, long_num_min_digits: int, max_key_len: int, max_val_len: int) -> List[str]:
    tokens: List[str] = []
    for k, v in _split_kv_pairs(value, ";"):
        tokens.append(TOK_COOKIE_K)
        tokens.extend(_sanitize_cookie_key(k, max_len=max_key_len))
        tokens.append(TOK_COOKIE_V)
        tokens.extend(
            _sanitize_value_tokens(v, keep_raw=False, long_num_min_digits=long_num_min_digits, max_len=max_val_len)
        )
    return tokens


def _tokens_from_json(
    obj: Mapping[str, Any],
    *,
    key_prefix: str,
    key_suffix: str,
    sep_token: str,
    eol_token: str,
    empty_token: str,
    long_num_min_digits: int,
    max_val_len: int,
) -> List[str]:
    tokens: List[str] = []
    for key in sorted(obj.keys()):
        tokens.append(f"{key_prefix}{key}{key_suffix}")
        tokens.append(sep_token)
        val = obj.get(key)
        if val is None:
            tokens.append(empty_token)
        elif isinstance(val, bool):
            tokens.append("true" if val else "false")
        elif isinstance(val, int):
            tokens.append(_bucket_long_int(str(val), long_num_min_digits))
        elif isinstance(val, float):
            tokens.append(TOK_NUM)
        elif isinstance(val, str):
            tokens.extend(
                _sanitize_value_tokens(val, keep_raw=False, long_num_min_digits=long_num_min_digits, max_len=max_val_len)
            )
        else:
            tokens.append(TOK_JSON)
        if eol_token:
            tokens.append(eol_token)
    return tokens


def _parse_tagged_sections(lines: List[str]) -> Optional[Tuple[str, str, str, List[str], str]]:
    method = ""
    url = ""
    proto = ""
    headers: List[str] = []
    body_lines: List[str] = []
    mode = ""
    saw_tag = False
    for line in lines:
        if line.startswith("<METHOD>"):
            method = line[len("<METHOD>") :].strip()
            mode = ""
            saw_tag = True
            continue
        if line.startswith("<URL>"):
            url = line[len("<URL>") :].strip()
            mode = ""
            saw_tag = True
            continue
        if line.startswith("<PROT>"):
            proto = line[len("<PROT>") :].strip()
            mode = ""
            saw_tag = True
            continue
        if line.startswith("<HDR>"):
            mode = "hdr"
            saw_tag = True
            continue
        if line.startswith("<BODY>"):
            mode = "body"
            saw_tag = True
            continue
        if mode == "hdr":
            headers.append(line)
        elif mode == "body":
            body_lines.append(line)
    if not saw_tag:
        return None
    body = "\n".join(body_lines).strip()
    return method, url, proto, headers, body


def _parse_raw_http(text: str) -> Tuple[str, str, str, List[str], str]:
    blocks = text.split("\n\n", 1)
    head = blocks[0]
    body = blocks[1] if len(blocks) > 1 else ""
    lines = [ln for ln in head.splitlines() if ln.strip()]
    method = ""
    url = ""
    proto = ""
    headers: List[str] = []
    if lines:
        parts = lines[0].split()
        if len(parts) >= 2:
            method = parts[0]
            url = parts[1]
            if len(parts) >= 3:
                proto = parts[2]
        headers = lines[1:]
    return method, url, proto, headers, body.strip()


def _tokens_from_url(
    url: str,
    *,
    long_num_min_digits: int,
    max_seg_len: int,
    max_key_len: int,
    max_val_len: int,
) -> List[str]:
    tokens: List[str] = [TOK_URL]
    url = url.strip()
    if not url:
        tokens.append("<EMPTY>")
        return tokens

    parts = urlsplit(url)
    path = parts.path or url
    if path.startswith("/"):
        tokens.append("/")
        path = path[1:]
    if path:
        segments = path.split("/")
        for i, seg in enumerate(segments):
            if not seg:
                continue
            tokens.extend(
                _sanitize_value_tokens(
                    seg,
                    keep_raw=True,
                    long_num_min_digits=long_num_min_digits,
                    max_len=max_seg_len,
                )
            )
            if i < len(segments) - 1:
                tokens.append("/")
    if parts.query:
        tokens.extend(
            _tokens_from_query(
                parts.query,
                long_num_min_digits=long_num_min_digits,
                max_key_len=max_key_len,
                max_val_len=max_val_len,
            )
        )
    return tokens


def structure_http(
    text_or_obj: Any,
    *,
    sep_token: str = "<SEP>",
    eol_token: str = "<EOL>",
    key_prefix: str = "<K:",
    key_suffix: str = ">",
    empty_token: str = "<EMPTY>",
    long_num_min_digits: int = 6,
    max_len: int = 4096,
) -> str:
    if text_or_obj is None:
        return ""
    if isinstance(text_or_obj, Mapping):
        obj = text_or_obj
        keys = {_ascii_lower(str(k)) for k in obj.keys()}
        if not keys.intersection({"method", "url", "protocol", "proto", "headers", "body"}):
            return json.dumps(obj, ensure_ascii=True)
        method = str(obj.get("method", "")).strip()
        url = str(obj.get("url", "")).strip()
        proto = str(obj.get("protocol", obj.get("proto", ""))).strip()
        headers_obj = obj.get("headers", "")
        body = obj.get("body", "")
        headers_lines: List[str] = []
        if isinstance(headers_obj, Mapping):
            for k, v in headers_obj.items():
                headers_lines.append(f"{k}: {v}")
        else:
            headers_lines = str(headers_obj).splitlines() if headers_obj else []
        return _structure_http_parts(
            method,
            url,
            proto,
            headers_lines,
            body,
            sep_token=sep_token,
            eol_token=eol_token,
            key_prefix=key_prefix,
            key_suffix=key_suffix,
            empty_token=empty_token,
            long_num_min_digits=long_num_min_digits,
            max_len=max_len,
        )

    text = str(text_or_obj)
    if TAG_ONLY_RE.fullmatch(text.strip()):
        return text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > max_len:
        text = text[:max_len]
        truncated = True
    else:
        truncated = False
    lines = text.splitlines()
    tagged = _parse_tagged_sections(lines)
    if tagged is not None:
        method, url, proto, headers, body = tagged
    else:
        method, url, proto, headers, body = _parse_raw_http(text)
    out = _structure_http_parts(
        method,
        url,
        proto,
        headers,
        body,
        sep_token=sep_token,
        eol_token=eol_token,
        key_prefix=key_prefix,
        key_suffix=key_suffix,
        empty_token=empty_token,
        long_num_min_digits=long_num_min_digits,
        max_len=max_len,
    )
    if truncated:
        out = f"{out} {TOK_TRUNC}".strip()
    return out


def _structure_http_parts(
    method: str,
    url: str,
    proto: str,
    headers: List[str],
    body: Any,
    *,
    sep_token: str,
    eol_token: str,
    key_prefix: str,
    key_suffix: str,
    empty_token: str,
    long_num_min_digits: int,
    max_len: int,
) -> str:
    tokens: List[str] = []
    method = method.strip()
    proto = proto.strip()

    if method or url or proto:
        tokens.append(TOK_METHOD)
        tokens.append(method or "<EMPTY>")
        tokens.extend(
            _tokens_from_url(
                url,
                long_num_min_digits=long_num_min_digits,
                max_seg_len=64,
                max_key_len=64,
                max_val_len=128,
            )
        )
        tokens.append(TOK_PROT)
        tokens.append(proto or "<EMPTY>")
        if eol_token:
            tokens.append(eol_token)

    for line in headers:
        m = HEADER_RE.match(line)
        if not m:
            continue
        name = _ascii_lower(m.group(1).strip())
        value = m.group(2).strip()
        tokens.append(TOK_HDR)
        tokens.append(name or "<EMPTY>")
        tokens.append(sep_token)
        if name == "cookie":
            tokens.extend(
                _tokens_from_cookie(
                    value,
                    long_num_min_digits=long_num_min_digits,
                    max_key_len=64,
                    max_val_len=128,
                )
            )
        elif name == "authorization":
            lower = _ascii_lower(value)
            if lower.startswith("bearer "):
                tokens.append(TOK_AUTH_BEARER)
                tokens.append(value_placeholder(value[7:].strip(), long_num_min_digits=long_num_min_digits))
            elif lower.startswith("basic "):
                tokens.append(TOK_AUTH_BASIC)
                tokens.append(value_placeholder(value[6:].strip(), long_num_min_digits=long_num_min_digits))
            else:
                tokens.extend(
                    _sanitize_value_tokens(
                        value,
                        keep_raw=False,
                        long_num_min_digits=long_num_min_digits,
                        max_len=128,
                    )
                )
        else:
            tokens.extend(
                _sanitize_value_tokens(
                    value,
                    keep_raw=True,
                    long_num_min_digits=long_num_min_digits,
                    max_len=256,
                )
            )
        if eol_token:
            tokens.append(eol_token)

    if body is not None and str(body).strip():
        tokens.append(TOK_BODY)
        body_text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=True)
        body_text = body_text.strip()
        if body_text.startswith("{") or body_text.startswith("["):
            try:
                obj = json.loads(body_text)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                tokens.append(TOK_JSON)
                tokens.extend(
                    _tokens_from_json(
                        obj,
                        key_prefix=key_prefix,
                        key_suffix=key_suffix,
                        sep_token=sep_token,
                        eol_token=eol_token,
                        empty_token=empty_token,
                        long_num_min_digits=long_num_min_digits,
                        max_val_len=128,
                    )
                )
            elif isinstance(obj, list):
                tokens.append(TOK_JSON)
                tokens.append("<LIST>")
            else:
                tokens.append(TOK_RAW)
                if len(body_text) > max_len:
                    tokens.append(body_text[:max_len])
                    tokens.append(TOK_TRUNC)
                else:
                    tokens.append(body_text)
        elif "&" in body_text and "=" in body_text:
            tokens.append(TOK_FORM)
            tokens.extend(
                _tokens_from_query(
                    body_text,
                    long_num_min_digits=long_num_min_digits,
                    max_key_len=64,
                    max_val_len=128,
                )
            )
        else:
            tokens.append(TOK_RAW)
            if len(body_text) > max_len:
                tokens.append(body_text[:max_len])
                tokens.append(TOK_TRUNC)
            else:
                tokens.append(body_text)

    return " ".join(tok for tok in tokens if tok)

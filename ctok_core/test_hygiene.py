import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_modules():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ctok_core"))
    import hygiene  # noqa: E402
    import tokenization_ctok  # noqa: E402

    return hygiene, tokenization_ctok


def test_hygiene_deterministic():
    hygiene, _ = _load_modules()
    cfg = hygiene.default_hygiene_config()
    text = "connect 10.251.71.9:443 and blk_12345"
    out1 = hygiene.apply_hygiene(text, cfg)
    out2 = hygiene.apply_hygiene(text, cfg)
    assert out1 == out2


def test_hygiene_removes_ip_fragments():
    hygiene, _ = _load_modules()
    cfg = hygiene.default_hygiene_config()
    text = "from 10.251.71.9:443 to 51.38.19.110"
    out = hygiene.apply_hygiene(text, cfg)
    assert "<IPV4>" in out
    assert "<PORT>" in out
    assert "51.38.19" not in out


def test_runtime_equals_build_pipeline():
    hygiene, tokenization_ctok = _load_modules()
    cfg = hygiene.default_hygiene_config()
    with TemporaryDirectory() as td:
        td_path = Path(td)
        vocab = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
            "<IPV4>": 5,
            "<PORT>": 6,
            "from": 7,
            " ": 8,
            "to": 9,
        }
        (td_path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        meta = {"hygiene": cfg.to_dict(), "match_special_tokens": False}
        (td_path / "ctok_meta.json").write_text(json.dumps(meta), encoding="utf-8")

        tok = tokenization_ctok.CTokTokenizer(
            vocab_file=str(td_path / "vocab.json"),
            meta_file=str(td_path / "ctok_meta.json"),
        )
        text = "from 10.251.71.9:443 to 1.2.3.4"
        tokens = tok.tokenize(text)
        assert "<IPV4>" in tokens
        assert "<PORT>" in tokens


def test_typed_token_fragment_filter():
    hygiene, _ = _load_modules()
    typed = hygiene.default_hygiene_config().typed_tokens
    assert hygiene.is_typed_token_fragment("<IPV4", typed)
    assert hygiene.is_typed_token_fragment("IPV4>", typed)
    assert hygiene.is_typed_token_fragment("IPV4", typed)
    assert not hygiene.is_typed_token_fragment("<IPV4>", typed)

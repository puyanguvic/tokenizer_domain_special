from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", help="Path to the built tokenizer artifact directory")
    ap.add_argument("--text", default="Héllò hôw are ü? <tag a=1&b=2> 404 8080 123456")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.artifact)
    print("Loaded tokenizer type:", type(tok))
    print("backend_tokenizer type:", type(tok.backend_tokenizer))
    print("\nNormalizer output:")
    print(tok.backend_tokenizer.normalizer.normalize_str(args.text))
    print("\nPre-tokenizer output:")
    print(tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(args.text))

    enc = tok(args.text, add_special_tokens=True)
    print("\nEncoded:")
    print(enc)
    print("tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))


if __name__ == "__main__":
    main()

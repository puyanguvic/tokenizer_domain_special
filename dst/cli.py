import argparse
from .tokenizer import DSTTokenizer


def main():
    parser = argparse.ArgumentParser(description="Domain-Specific Tokenization CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--input", required=True, help="Path to training corpus")
    train_p.add_argument("--output", default="dst_tokenizer.json")

    encode_p = sub.add_parser("encode")
    encode_p.add_argument("--input", required=True)
    encode_p.add_argument("--tokenizer", default="dst_tokenizer.json")

    args = parser.parse_args()

    if args.cmd == "train":
        with open(args.input, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
        tokenizer = DSTTokenizer.train(corpus)
        tokenizer.save_json(args.output)
        print(f"✅ Tokenizer trained and saved to {args.output}")

    elif args.cmd == "encode":
        with open(args.tokenizer, "r", encoding="utf-8") as f:
            from json import load
            vocab = list(load(f)["vocab"].keys())
        dfst = DSTTokenizer.train(vocab)
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
        print(dfst.encode(text))


if __name__ == "__main__":
    main()

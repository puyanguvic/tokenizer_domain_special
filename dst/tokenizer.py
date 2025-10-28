import json
from .vocab import GrammarGuidedVocab
from .dfst import DFST


class DSTTokenizer:
    """Main interface for Domain-Specific Tokenization."""

    def __init__(self, vocab, dfst):
        self.vocab = vocab
        self.dfst = dfst

    @classmethod
    def train(cls, corpus, **kwargs):
        gg = GrammarGuidedVocab(**kwargs)
        vocab = gg.build_vocab(corpus)
        dfst = DFST()
        for token in vocab:
            dfst.add_token(token)
        return cls(vocab, dfst)

    def encode(self, text: str):
        return self.dfst.encode(text)

    def decode(self, tokens):
        return self.dfst.decode(tokens)

    def verify(self, corpus):
        return self.dfst.verify(corpus)

    def save_json(self, path="dst_tokenizer.json"):
        """Export tokenizer to Hugging Face-compatible JSON format."""
        obj = {
            "version": "1.0",
            "vocab": {t: i for i, t in enumerate(self.vocab)},
            "normalizer": {"type": "identity"},
            "pre_tokenizer": {"type": "Whitespace"},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

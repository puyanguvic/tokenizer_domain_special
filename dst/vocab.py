import re
from collections import defaultdict


class GrammarGuidedVocab:
    """Grammar-guided vocabulary induction."""

    def __init__(self, min_freq=5, max_vocab=32000, patterns=None):
        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.patterns = patterns or [
            r"https?://[^\s]+",
            r"\b\d{1,3}(\.\d{1,3}){3}\b",
            r"[A-Za-z_][A-Za-z0-9_]*",
            r"[A-Za-z0-9_\-]+=[A-Za-z0-9_\-]+",
            r"[A-Za-z0-9_\-]+\.[a-z]{2,4}",
        ]

    def extract_candidates(self, corpus):
        """Extract substrings that match domain grammar patterns."""
        freq = defaultdict(int)
        for line in corpus:
            for pat in self.patterns:
                for match in re.findall(pat, line):
                    freq[match] += 1
        return {w: c for w, c in freq.items() if c >= self.min_freq}

    def build_vocab(self, corpus):
        """Merge grammar-guided candidates into vocabulary."""
        candidates = self.extract_candidates(corpus)
        sorted_vocab = sorted(candidates.items(), key=lambda kv: -kv[1])
        vocab = [w for w, _ in sorted_vocab[: self.max_vocab]]
        ascii_chars = [chr(i) for i in range(32, 127)]
        return ascii_chars + vocab

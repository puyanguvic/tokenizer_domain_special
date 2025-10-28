from dataclasses import dataclass, field


@dataclass
class DFSTNode:
    next: dict = field(default_factory=dict)
    output: str | None = None


@dataclass
class DFST:
    states: list[DFSTNode] = field(default_factory=lambda: [DFSTNode()])

    def add_token(self, token: str):
        """Insert a token path into the trie."""
        state = 0
        for ch in token:
            if ch not in self.states[state].next:
                self.states[state].next[ch] = len(self.states)
                self.states.append(DFSTNode())
            state = self.states[state].next[ch]
        self.states[state].output = token

    def encode(self, text: str) -> list[str]:
        """Deterministic maximal-munch encoding."""
        i, out = 0, []
        while i < len(text):
            s, last_token, last_pos = 0, None, i
            j = i
            while j < len(text) and text[j] in self.states[s].next:
                s = self.states[s].next[text[j]]
                if self.states[s].output:
                    last_token, last_pos = self.states[s].output, j + 1
                j += 1
            if last_token:
                out.append(last_token)
                i = last_pos
            else:
                out.append(text[i])
                i += 1
        return out

    def decode(self, tokens: list[str]) -> str:
        """Exact inverse decoding."""
        return "".join(tokens)

    def verify(self, corpus) -> bool:
        """Check κ(τ(x)) = x for all x in corpus."""
        for line in corpus:
            if self.decode(self.encode(line)) != line:
                return False
        return True

import json
import os
from typing import List, Dict, Optional


class _TrieNode:
    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.token_id: Optional[int] = None


class Tokenizer:
    """Trainable, lossless BPE/SuperBPE-inspired tokenizer.

    Behavior:
    - `__init__` loads a `tokenizer.json` file (vocab + merges) when available.
    - `encode` greedily matches the longest token at each position using a trie.
    - `decode` maps token ids back to strings and concatenates them to reconstruct
      the original input exactly.

    The implementation treats the input as ASCII strings (the competition
    provides preprocessed ASCII-only strings). The tokenizer is lossless: every
    token internally stores the exact character sequence it represents and
    decoding is plain concatenation of those sequences.
    """

    def __init__(self, model_path: str = None):
        # load tokenizer.json if present
        root = os.path.dirname(__file__)
        path = model_path or os.path.join(root, "tokenizer.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # vocab: token -> id
            self.vocab: Dict[str, int] = dict(data.get("vocab", {}))
            # ensure all printable ASCII symbols are present in the final vocab
            max_id = max((int(v) for v in self.vocab.values()), default=0)
            for code in range(32, 127):
                ch = chr(code)
                if ch not in self.vocab:
                    max_id += 1
                    self.vocab[ch] = max_id
            self.id_to_token: Dict[int, str] = {int(v): k for k, v in self.vocab.items()}
        else:
            # fallback: identity character mapping (lossless)
            self.vocab = {chr(i): i for i in range(32, 127)}
            # shift ids to start at 1 to leave 0 unused
            self.vocab = {k: (v - 31) for k, v in self.vocab.items()}
            self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Build a trie for fast greedy longest-match tokenization
        self._trie_root = _TrieNode()
        for token, tid in self.vocab.items():
            node = self._trie_root
            for ch in token:
                node = node.children.setdefault(ch, _TrieNode())
            node.token_id = int(tid)

    def encode(self, texts: List[str]) -> List[List[int]]:
        """Encode a list of ASCII strings into token id sequences.

        Greedy longest-match algorithm using the learned vocabulary trie.
        """
        encoded_batch: List[List[int]] = []
        for text in texts:
            i = 0
            n = len(text)
            ids: List[int] = []
            while i < n:
                node = self._trie_root
                match_id = None
                match_len = 0
                j = i
                # Walk as far as possible
                while j < n and text[j] in node.children:
                    node = node.children[text[j]]
                    j += 1
                    if node.token_id is not None:
                        match_id = node.token_id
                        match_len = j - i
                if match_id is None:
                    # no multi-char token matched; fall back to single character
                    ch = text[i]
                    tid = self.vocab.get(ch)
                    if tid is None:
                        # unseen character: map to integer by ordinal fallback
                        ids.append(ord(ch))
                    else:
                        ids.append(int(tid))
                    i += 1
                else:
                    ids.append(int(match_id))
                    i += match_len
            encoded_batch.append(ids)
        return encoded_batch

    def decode(self, encoded_texts: List[List[int]]) -> List[str]:
        """Decode batches of token id lists back to strings exactly."""
        out: List[str] = []
        for ids in encoded_texts:
            parts: List[str] = []
            for tid in ids:
                token = self.id_to_token.get(int(tid))
                if token is None:
                    # fallback: ord -> char when possible
                    try:
                        parts.append(chr(int(tid)))
                    except Exception:
                        parts.append("")
                else:
                    parts.append(token)
            out.append("".join(parts))
        return out


__all__ = ["Tokenizer"]

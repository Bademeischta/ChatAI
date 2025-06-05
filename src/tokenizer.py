from __future__ import annotations

from typing import List

import numpy as np


class Tokenizer:
    """Whitespace tokenizer with vocabulary management."""

    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>", 3: "<unk>"}

    def build_vocab_from_texts(self, texts: List[str]) -> None:
        """Add tokens from the provided texts to the vocabulary."""
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        """Split text on whitespace."""
        return text.replace("\t", " ").strip().split()

    def encode(self, text: str, max_len: int) -> np.ndarray:
        """Convert text to a sequence of token ids."""
        tokens = ["<bos>"] + self.tokenize(text) + ["<eos>"]
        ids = [self.token_to_id.get(t, self.token_to_id["<unk>"]) for t in tokens]
        if len(ids) < max_len:
            ids += [self.token_to_id["<pad>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: List[int]) -> str:
        """Convert ids back to a text string."""
        tokens = [self.id_to_token.get(i, "") for i in ids if i != self.token_to_id["<pad>"]]
        return " ".join(tokens)

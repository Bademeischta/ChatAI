from __future__ import annotations

from typing import List

import numpy as np


class Tokenizer:
    """Whitespace tokenizer with vocabulary management."""

    def __init__(self) -> None:
        self.token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self.id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

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

    def encode(self, text: str, max_len: int, as_tensor: bool = False):
        """Convert text to a sequence of token ids.

        Args:
            text: Input string.
            max_len: Desired sequence length.
            as_tensor: If ``True`` return a torch tensor instead of ``numpy``.
        """
        tokens = ["<bos>"] + self.tokenize(text) + ["<eos>"]
        ids = [self.token_to_id.get(t, self.token_to_id["<pad>"]) for t in tokens]
        if len(ids) < max_len:
            ids += [self.token_to_id["<pad>"]] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        arr = np.array(ids, dtype=np.int64)
        if as_tensor:
            import torch

            return torch.tensor(arr, dtype=torch.long)
        return arr

    def decode(self, ids: List[int]) -> str:
        """Convert ids back to a text string."""
        if not isinstance(ids, list):
            ids = list(ids)
        tokens = [self.id_to_token.get(int(i), "") for i in ids if i != self.token_to_id["<pad>"]]
        return " ".join(tokens)

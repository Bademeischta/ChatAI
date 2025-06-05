from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(x, *args, **kwargs):
        return x


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class Config:
    """Container for hyperparameters and paths."""

    vocab_size: int = 30000
    embedding_dim: int = 256
    num_heads: int = 8
    ffn_dim: int = 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dropout_rate: float = 0.1
    max_seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 1e-4
    label_smoothing: float = 0.1
    device: str = "cpu"
    beam_width: int = 3
    top_k: int = 50
    top_p: float = 0.9
    train_file: str = "data/train.jsonl"
    valid_file: str = "data/valid.jsonl"
    test_file: str = "data/test.jsonl"
    checkpoint_dir: str = "checkpoints/"
    vocab_file: str = "data/vocab.json"

    @staticmethod
    def from_json(path: str) -> "Config":
        """Load configuration values from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = Config()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


class Logger:
    """Simple logger writing to console and file."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def log(self, message: str) -> None:
        msg = message.strip()
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def create_padding_mask(seq: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """Create mask with -inf where padding tokens are."""
    mask = (seq == pad_id).astype(np.float32)
    return mask[:, None, None, :] * -1e9


def create_look_ahead_mask(size: int) -> np.ndarray:
    """Create look-ahead mask for decoder input."""
    mask = np.triu(np.ones((1, 1, size, size)), k=1).astype(np.float32)
    return mask * -1e9

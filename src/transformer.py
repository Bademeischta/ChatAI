from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .utils import create_look_ahead_mask, create_padding_mask


class LayerNorm:
    """Simple layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


def xavier_init(size: Tuple[int, int]) -> np.ndarray:
    """Xavier uniform initialization."""
    in_dim, out_dim = size
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax implementation."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class Dropout:
    """Drop values randomly during training."""

    def __init__(self, rate: float = 0.0) -> None:
        self.rate = rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rate <= 0.0:
            return x
        mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
        return x * mask / (1 - self.rate)


class MultiHeadAttention:
    """Multi-Head Attention module."""

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.0) -> None:
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = xavier_init((embed_dim, embed_dim))
        self.W_k = xavier_init((embed_dim, embed_dim))
        self.W_v = xavier_init((embed_dim, embed_dim))
        self.W_o = xavier_init((embed_dim, embed_dim))
        self.dropout = Dropout(dropout_rate)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        b, s, d = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        b, h, s, d = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(b, s, h * d)
        return x

    def __call__(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        Q = self._split_heads(q @ self.W_q)
        K = self._split_heads(k @ self.W_k)
        V = self._split_heads(v @ self.W_v)
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        weights = softmax(scores)
        weights = self.dropout(weights)
        heads = weights @ V
        out = self._combine_heads(heads) @ self.W_o
        return out, weights


class FeedForwardNetwork:
    """Two-layer feed-forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout_rate: float = 0.0) -> None:
        self.W1 = xavier_init((embed_dim, ffn_dim))
        self.b1 = np.zeros(ffn_dim)
        self.W2 = xavier_init((ffn_dim, embed_dim))
        self.b2 = np.zeros(embed_dim)
        self.dropout = Dropout(dropout_rate)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = relu(x @ self.W1 + self.b1)
        h = self.dropout(h)
        return h @ self.W2 + self.b2


class EncoderLayer:
    """Single Transformer encoder layer."""

    def __init__(self, config) -> None:
        self.self_attn = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout_rate)
        self.dropout1 = Dropout(config.dropout_rate)
        self.norm1 = LayerNorm(config.embedding_dim)
        self.ffn = FeedForwardNetwork(config.embedding_dim, config.ffn_dim, config.dropout_rate)
        self.dropout2 = Dropout(config.dropout_rate)
        self.norm2 = LayerNorm(config.embedding_dim)

    def __call__(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer:
    """Single Transformer decoder layer."""

    def __init__(self, config) -> None:
        self.self_attn = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout_rate)
        self.cross_attn = MultiHeadAttention(config.embedding_dim, config.num_heads, config.dropout_rate)
        self.norm1 = LayerNorm(config.embedding_dim)
        self.norm2 = LayerNorm(config.embedding_dim)
        self.norm3 = LayerNorm(config.embedding_dim)
        self.dropout = Dropout(config.dropout_rate)
        self.ffn = FeedForwardNetwork(config.embedding_dim, config.ffn_dim, config.dropout_rate)

    def __call__(self, x: np.ndarray, enc_out: np.ndarray, look_ahead_mask: Optional[np.ndarray] = None, padding_mask: Optional[np.ndarray] = None) -> np.ndarray:
        attn1, _ = self.self_attn(x, x, x, look_ahead_mask)
        x = self.norm1(x + self.dropout(attn1))
        attn2, _ = self.cross_attn(x, enc_out, enc_out, padding_mask)
        x = self.norm2(x + self.dropout(attn2))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Encoder:
    """Transformer encoder consisting of multiple layers."""

    def __init__(self, config) -> None:
        self.embedding = xavier_init((config.vocab_size, config.embedding_dim))
        self.pos_enc = positional_encoding(config.max_seq_len, config.embedding_dim)
        self.layers = [EncoderLayer(config) for _ in range(config.num_encoder_layers)]

    def __call__(self, src_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = self.embedding[src_ids] + self.pos_enc[: src_ids.shape[1]]
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder:
    """Transformer decoder consisting of multiple layers."""

    def __init__(self, config) -> None:
        self.embedding = xavier_init((config.vocab_size, config.embedding_dim))
        self.pos_enc = positional_encoding(config.max_seq_len, config.embedding_dim)
        self.layers = [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        self.fc_out = xavier_init((config.embedding_dim, config.vocab_size))

    def __call__(self, tgt_ids: np.ndarray, enc_out: np.ndarray, look_ahead_mask: Optional[np.ndarray] = None, padding_mask: Optional[np.ndarray] = None) -> np.ndarray:
        x = self.embedding[tgt_ids] + self.pos_enc[: tgt_ids.shape[1]]
        for layer in self.layers:
            x = layer(x, enc_out, look_ahead_mask, padding_mask)
        return x @ self.fc_out


def positional_encoding(seq_len: int, dim: int) -> np.ndarray:
    """Create sinusoidal positional encoding."""
    PE = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            angle = pos / np.power(10000, (2 * i) / dim)
            PE[pos, i] = math.sin(angle)
            if i + 1 < dim:
                PE[pos, i + 1] = math.cos(angle)
    return PE

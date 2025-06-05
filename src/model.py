from __future__ import annotations

from typing import Optional

import numpy as np

from .transformer import Decoder, Encoder, positional_encoding
from .utils import Config, create_look_ahead_mask, create_padding_mask


class Seq2SeqTransformer:
    """Minimal Seq2Seq Transformer model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.params = [
            self.encoder.embedding,
            self.decoder.embedding,
            self.decoder.fc_out,
        ]
        for layer in self.encoder.layers + self.decoder.layers:
            self.params.extend([
                layer.self_attn.W_q,
                layer.self_attn.W_k,
                layer.self_attn.W_v,
                layer.self_attn.W_o,
                getattr(layer, "cross_attn", layer.self_attn).W_q,
            ])

    def encode(self, src_ids: np.ndarray, src_mask: Optional[np.ndarray]) -> np.ndarray:
        return self.encoder(src_ids, src_mask)

    def decode(self, tgt_ids: np.ndarray, enc_out: np.ndarray, look_ahead_mask: Optional[np.ndarray], padding_mask: Optional[np.ndarray]) -> np.ndarray:
        return self.decoder(tgt_ids, enc_out, look_ahead_mask, padding_mask)

    def forward(self, src_ids: np.ndarray, tgt_ids: np.ndarray, src_mask: Optional[np.ndarray], tgt_mask: Optional[np.ndarray]) -> np.ndarray:
        enc_out = self.encode(src_ids, src_mask)
        logits = self.decode(tgt_ids, enc_out, tgt_mask, src_mask)
        return logits

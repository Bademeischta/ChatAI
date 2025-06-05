"""PyTorch-based Seq2Seq Transformer model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .utils import Config


class Seq2SeqTransformer(nn.Module):
    """Minimal wrapper around ``nn.Transformer`` for sequence generation."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.transformer = nn.Transformer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout_rate,
            batch_first=True,
        )
        self.output_proj = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute logits for a batch."""

        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.output_proj(out)


def greedy_decode(
    model: Seq2SeqTransformer, src: torch.Tensor, tokenizer, max_len: int
) -> str:
    model.eval()
    device = src.device
    src_padding = src == tokenizer.token_to_id["<pad>"]

    with torch.no_grad():
        # Encode source only once
        src_emb = model.embedding(src)
        memory = model.transformer.encoder(
            src_emb, src_key_padding_mask=src_padding
        )

        ys = torch.full(
            (1, 1), tokenizer.token_to_id["<bos>"], dtype=torch.long, device=device
        )
        for _ in range(max_len - 1):
            tgt_emb = model.embedding(ys)
            tgt_mask = model.transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.transformer.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_padding,
            )
            out = model.output_proj(out)
            prob = out[:, -1, :].softmax(dim=-1)
            next_word = int(prob.argmax(dim=-1))
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == tokenizer.token_to_id["<eos>"]:
                break
    return tokenizer.decode(ys[0].cpu().tolist())


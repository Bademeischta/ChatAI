from __future__ import annotations

from typing import Optional

import numpy as np

from .model import Seq2SeqTransformer
from .tokenizer import Tokenizer
from .utils import Config, create_look_ahead_mask, create_padding_mask


def greedy_decode(model: Seq2SeqTransformer, src_tokens: np.ndarray, tokenizer: Tokenizer, config: Config) -> str:
    tgt_ids = np.array([[tokenizer.token_to_id["<bos>"]] + [tokenizer.token_to_id["<pad>"]] * (config.max_seq_len - 1)])
    for i in range(1, config.max_seq_len):
        logits = model.forward(src_tokens, tgt_ids, None, None)
        next_id = int(np.argmax(logits[0, i - 1]))
        tgt_ids[0, i] = next_id
        if next_id == tokenizer.token_to_id["<eos>"]:
            break
    return tokenizer.decode(tgt_ids[0])


def generate_response(input_text: str, model: Seq2SeqTransformer, tokenizer: Tokenizer, config: Config) -> str:
    src_tokens = tokenizer.encode(input_text, config.max_seq_len)[None, :]
    return greedy_decode(model, src_tokens, tokenizer, config)

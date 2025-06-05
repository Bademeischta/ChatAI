from __future__ import annotations

import numpy as np

from .model import Seq2SeqTransformer, greedy_decode
from .tokenizer import Tokenizer
from .utils import Config


def generate_response(input_text: str, model: Seq2SeqTransformer, tokenizer: Tokenizer, config: Config) -> str:
    """Generate a response using greedy decoding."""
    src_tokens = tokenizer.encode(input_text, config.max_seq_len, as_tensor=True).unsqueeze(0)
    return greedy_decode(model, src_tokens, tokenizer, config.max_seq_len)

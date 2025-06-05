import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.model import Seq2SeqTransformer
from src.utils import Config


def test_cross_attention_params_present():
    cfg = Config()
    model = Seq2SeqTransformer(cfg)
    for layer in model.decoder.layers:
        assert any(p is layer.cross_attn.W_k for p in model.params)
        assert any(p is layer.cross_attn.W_v for p in model.params)
        assert any(p is layer.cross_attn.W_o for p in model.params)

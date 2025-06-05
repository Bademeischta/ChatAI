import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.transformer import MultiHeadAttention


def test_attention_shapes():
    x = np.random.randn(2, 4, 8)
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    out, attn = mha(x, x, x, mask=None)
    assert out.shape == (2, 4, 8)
    assert attn.shape == (2, 2, 4, 4)

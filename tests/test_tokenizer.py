import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.tokenizer import Tokenizer


def test_tokenizer_basic():
    tok = Tokenizer()
    tok.build_vocab_from_texts(["hallo", "welt"])
    assert tok.encode("hallo", 4).tolist()[1] == tok.token_to_id["hallo"]
    assert tok.decode([tok.token_to_id["welt"], 0]) == "welt"


def test_unknown_token_maps_to_unk():
    tok = Tokenizer()
    tok.build_vocab_from_texts(["hallo"])
    ids = tok.encode("foo", 4)
    assert ids[1] == tok.token_to_id["<unk>"]

import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from src.tokenizer import Tokenizer


def test_tokenizer_basic():
    tok = Tokenizer()
    tok.build_vocab_from_texts(["hallo", "welt"])
    assert tok.encode("hallo", 4).tolist()[1] == tok.token_to_id["hallo"]
    assert tok.decode([tok.token_to_id["welt"], 0]) == "welt"


def test_tokenizer_save_load(tmp_path):
    tok = Tokenizer()
    tok.build_vocab_from_texts(["foo", "bar"])
    path = tmp_path / "vocab.json"
    tok.save_vocab(path)
    tok2 = Tokenizer()
    tok2.load_vocab(path)
    assert tok.token_to_id == tok2.token_to_id

import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import numpy as np
from src.train import create_batches, train_epoch, save_checkpoint
from src.tokenizer import Tokenizer
from src.model import Seq2SeqTransformer
from src.optimizer import AdamOptimizer
from src.utils import Config, Logger


def test_training_integration(tmp_path):
    cfg = Config()
    cfg.checkpoint_dir = str(tmp_path)
    data = [{"input": "Hallo", "response": "Welt"}]
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_texts(["Hallo", "Welt"])
    batches = create_batches(data, tokenizer, cfg)
    model = Seq2SeqTransformer(cfg)
    opt = AdamOptimizer(model.params, lr=cfg.learning_rate)
    logger = Logger("/tmp/log.txt")
    train_epoch(model, batches, opt, cfg, logger, 0)
    save_checkpoint(model, opt, 0, cfg)
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch_0.npz"))

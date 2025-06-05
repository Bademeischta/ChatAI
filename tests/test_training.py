import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os
import torch
from torch.utils.data import DataLoader
from src.train import ChatDataset, train_epoch, save_checkpoint
from src.tokenizer import Tokenizer
from src.model import Seq2SeqTransformer
from src.utils import Config, Logger


def test_training_integration(tmp_path):
    cfg = Config()
    cfg.checkpoint_dir = str(tmp_path)
    data = [{"input": "Hallo", "response": "Welt"}]
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_texts(["Hallo", "Welt"])
    dataset = ChatDataset(data, tokenizer, cfg)
    loader = DataLoader(dataset, batch_size=1)
    model = Seq2SeqTransformer(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    crit = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
    logger = Logger("/tmp/log.txt")
    train_epoch(model, loader, opt, crit, logger, 0)
    save_checkpoint(model, opt, 0, cfg)
    assert os.path.exists(os.path.join(cfg.checkpoint_dir, "checkpoint_epoch_0.pt"))

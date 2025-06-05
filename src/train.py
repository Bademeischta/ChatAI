"""Training and CLI entry point using PyTorch."""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import Seq2SeqTransformer, greedy_decode
from .tokenizer import Tokenizer
from .utils import Config, Logger, load_jsonl, tqdm


class ChatDataset(Dataset):
    """Dataset that returns encoded input and target sequences."""

    def __init__(self, data: List[dict], tokenizer: Tokenizer, config: Config) -> None:
        self.samples = []
        for d in data:
            src = tokenizer.encode(d["input"], config.max_seq_len, as_tensor=True)
            tgt = tokenizer.encode(d["response"], config.max_seq_len, as_tensor=True)
            self.samples.append((src, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def train_epoch(
    model: Seq2SeqTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    logger: Logger,
    epoch_id: int,
) -> None:
    model.train()
    for src, tgt in tqdm(dataloader, desc=f"Epoch {epoch_id}"):
        optimizer.zero_grad()
        logits = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        logger.log(f"Epoch {epoch_id}\tLoss {loss.item():.4f}")


def save_checkpoint(model: Seq2SeqTransformer, optimizer: torch.optim.Optimizer, epoch: int, config: Config) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatAI Trainer")
    parser.add_argument("--mode", choices=["train", "chat"], required=True)
    parser.add_argument("--config_file", default="config.json")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    config = Config.from_json(args.config_file)
    logger = Logger("training.log")
    tokenizer = Tokenizer()

    model = Seq2SeqTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    if args.mode == "train":
        train_data = load_jsonl(config.train_file)
        texts = [d["input"] for d in train_data] + [d["response"] for d in train_data]
        tokenizer.build_vocab_from_texts(texts)
        dataset = ChatDataset(train_data, tokenizer, config)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        train_epoch(model, dataloader, optimizer, criterion, logger, epoch_id=1)
        save_checkpoint(model, optimizer, 1, config)
    else:  # chat mode
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for chat mode")
        model.eval()
        while True:
            user_input = input("User: ")
            if user_input.strip().lower() == "exit":
                break
            src = tokenizer.encode(user_input, config.max_seq_len, as_tensor=True).unsqueeze(0)
            response = greedy_decode(model, src, tokenizer, config.max_seq_len)
            print(f"Bot: {response}")


if __name__ == "__main__":  # pragma: no cover
    main()


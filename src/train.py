from __future__ import annotations

import argparse
import os
from typing import Iterator, List

import numpy as np

from .model import Seq2SeqTransformer
from .optimizer import AdamOptimizer
from .tokenizer import Tokenizer
from .utils import Config, Logger, create_look_ahead_mask, create_padding_mask, load_jsonl, tqdm


def compute_loss(logits: np.ndarray, target_ids: np.ndarray, pad_id: int = 0, smoothing: float = 0.0) -> float:
    probs = softmax(logits)
    log_probs = np.log(probs + 1e-9)
    vocab = log_probs.shape[-1]
    smooth = smoothing / (vocab - 1) if smoothing > 0 else 0.0

    b_range = np.arange(log_probs.shape[0])[:, None]
    t_range = np.arange(log_probs.shape[1])[None, :]
    log_probs_target = log_probs[b_range, t_range, target_ids]
    sum_log_probs = np.sum(log_probs, axis=-1)

    loss = -((1.0 - smoothing) * log_probs_target + smooth * sum_log_probs)
    mask = (target_ids != pad_id)
    masked_loss = loss * mask
    return float(masked_loss.sum() / max(mask.sum(), 1))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def compute_grads(loss: float, params: List[np.ndarray]) -> List[np.ndarray]:
    # Placeholder for autograd; returns zero gradients
    return [np.zeros_like(p) for p in params]


def train_epoch(model: Seq2SeqTransformer, data: List[dict], optimizer: AdamOptimizer, config: Config, logger: Logger, epoch_id: int) -> None:
    for batch in tqdm(data, desc=f"Epoch {epoch_id}"):
        logits = model.forward(batch["input_ids"], batch["target_ids"], None, None)
        loss = compute_loss(logits, batch["target_ids"], smoothing=config.label_smoothing)
        grads = compute_grads(loss, model.params)
        optimizer.step(grads)
        logger.log(f"Epoch {epoch_id}\tLoss {loss:.4f}")


def save_checkpoint(model: Seq2SeqTransformer, optimizer: AdamOptimizer, epoch: int, config: Config) -> None:
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, f"checkpoint_epoch_{epoch}.npz")
    np.savez(path, **{f"param_{i}": p for i, p in enumerate(model.params)}, t=optimizer.t)

def create_batches(data: List[dict], tokenizer: Tokenizer, config: Config) -> List[dict]:
    batches = []
    for d in data:
        batches.append({
            "input_ids": tokenizer.encode(d["input"], config.max_seq_len)[None, :],
            "target_ids": tokenizer.encode(d["response"], config.max_seq_len)[None, :],
        })
    return batches


def main() -> None:
    parser = argparse.ArgumentParser(description="ChatAI Trainer")
    parser.add_argument("--mode", choices=["train", "chat"], required=True)
    parser.add_argument("--config_file", default="config.json")
    args = parser.parse_args()

    config = Config.from_json(args.config_file)
    logger = Logger("training.log")
    tokenizer = Tokenizer()

    if args.mode == "train":
        train_data = load_jsonl(config.train_file)
        texts = [d["input"] for d in train_data] + [d["response"] for d in train_data]
        tokenizer.build_vocab_from_texts(texts)
        batches = create_batches(train_data, tokenizer, config)
        model = Seq2SeqTransformer(config)
        optimizer = AdamOptimizer(model.params, lr=config.learning_rate)
        train_epoch(model, batches, optimizer, config, logger, epoch_id=1)
        save_checkpoint(model, optimizer, 1, config)
    elif args.mode == "chat":
        print("Chat mode not fully implemented.")


if __name__ == "__main__":
    main()

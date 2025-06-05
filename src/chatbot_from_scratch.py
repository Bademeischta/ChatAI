"""Minimal Transformer Chatbot implemented with NumPy.

This script contains a tiny implementation of a Seq2Seq Transformer and a simple
training loop. The goal is educational: understand how the basic building blocks
work without relying on a deep learning framework.

CLI options allow training and chatting. Paths for data and checkpoints can be
configured via arguments.
"""

import argparse
import json
import math
import os
import numpy as np
from typing import List

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = lambda x: x

# ----------------------------- Tokenizer ------------------------------------
class Tokenizer:
    """Very simple whitespace tokenizer.

    Note: currently monolingual (expects plain whitespace-separated text). For
    other languages consider Unicode-aware tokenization or BPE.
    """

    def __init__(self):
        self.token_to_id = {'<pad>': 0, '<bos>': 1, '<eos>': 2}
        self.id_to_token = {0: '<pad>', 1: '<bos>', 2: '<eos>'}

    def build_vocab(self, texts: List[str]):
        for line in texts:
            for token in self.tokenize(line):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def extend_vocabulary(self, new_texts: List[str]):
        """Add tokens from new_texts to the vocabulary."""
        self.build_vocab(new_texts)

    def tokenize(self, text: str) -> List[str]:
        return text.replace('\t', ' ').strip().split()

    def encode(self, text: str, max_len: int = 32) -> np.ndarray:
        tokens = ['<bos>'] + self.tokenize(text) + ['<eos>']
        ids = [self.token_to_id.get(t, self.token_to_id['<pad>']) for t in tokens]
        if len(ids) < max_len:
            ids += [self.token_to_id['<pad>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(i, '') for i in ids if i != self.token_to_id['<pad>']]
        return ' '.join(tokens)

# ----------------------------- Helpers --------------------------------------

def xavier_init(size):
    in_dim, out_dim = size
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class Dropout:
    def __init__(self, rate=0.1):
        self.rate = rate

    def __call__(self, x):
        if self.rate <= 0:
            return x
        mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
        return x * mask / (1 - self.rate)

# Sinusoidale Positionskodierung
def positional_encoding(seq_len, dim):
    PE = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            angle = pos / np.power(10000, (2 * i) / dim)
            PE[pos, i] = math.sin(angle)
            if i + 1 < dim:
                PE[pos, i + 1] = math.cos(angle)
    return PE

# ----------------------------- Attention and FFN ----------------------------
class MultiHeadSelfAttention:
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = Dropout(dropout)
        self.W_q = xavier_init((embed_dim, embed_dim))
        self.W_k = xavier_init((embed_dim, embed_dim))
        self.W_v = xavier_init((embed_dim, embed_dim))
        self.W_o = xavier_init((embed_dim, embed_dim))

    def split_heads(self, x):
        b, s, d = x.shape
        x = x.reshape(b, s, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        b, h, s, d = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(b, s, h * d)
        return x

    def __call__(self, x, mask=None):
        Q = self.split_heads(x @ self.W_q)
        K = self.split_heads(x @ self.W_k)
        V = self.split_heads(x @ self.W_v)
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        weights = softmax(scores)
        weights = self.dropout(weights)
        heads = weights @ V
        out = self.combine_heads(heads)
        return out @ self.W_o

class FeedForward:
    def __init__(self, dim, hidden_dim, dropout=0.0):
        self.W1 = xavier_init((dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier_init((hidden_dim, dim))
        self.b2 = np.zeros(dim)
        self.dropout = Dropout(dropout)

    def __call__(self, x):
        h = relu(x @ self.W1 + self.b1)
        h = self.dropout(h)
        return h @ self.W2 + self.b2

class TransformerBlock:
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.0):
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim, dropout)
        self.ln2 = LayerNorm(dim)

    def __call__(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

# ----------------------------- Other utils ----------------------------------

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

# ----------------------------- Encoder/Decoder ------------------------------
class TransformerEncoder:
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim, dropout=0.0):
        self.embedding = xavier_init((vocab_size, embed_dim))
        self.pe = positional_encoding(seq_len, embed_dim)
        self.layers = [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]

    def __call__(self, x_ids):
        x = self.embedding[x_ids] + self.pe[:len(x_ids[0])]
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerDecoder:
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim, dropout=0.0):
        self.embedding = xavier_init((vocab_size, embed_dim))
        self.pe = positional_encoding(seq_len, embed_dim)
        self.layers = [TransformerBlock(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)]
        self.fc_out = xavier_init((embed_dim, vocab_size))
        self.vocab_size = vocab_size
        self.params = [self.embedding, self.fc_out]

    def __call__(self, x_ids, enc_out):
        x = self.embedding[x_ids] + self.pe[:len(x_ids[0])]
        mask = np.triu(np.full((x.shape[1], x.shape[1]), -np.inf), k=1)
        for layer in self.layers:
            x = layer(x, mask)
        logits = x @ self.fc_out
        return logits

class Seq2SeqTransformer:
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim=64, num_heads=4, hidden_dim=128, dropout=0.0):
        self.encoder = TransformerEncoder(num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim, dropout)
        self.decoder = TransformerDecoder(num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim, dropout)
        self.params = [self.encoder.embedding, self.decoder.embedding, self.decoder.fc_out]
        for block in self.encoder.layers + self.decoder.layers:
            self.params.extend([block.attn.W_q, block.attn.W_k, block.attn.W_v, block.attn.W_o,
                                block.ffn.W1, block.ffn.b1, block.ffn.W2, block.ffn.b2])

    def __call__(self, src_ids, tgt_ids):
        enc_out = self.encoder(src_ids)
        logits = self.decoder(tgt_ids, enc_out)
        return logits

# ----------------------------- Loss & Optimizer -----------------------------

def cross_entropy(logits, target_ids, pad_id=0, smoothing=0.0):
    probs = softmax(logits)
    if smoothing > 0:
        vocab = probs.shape[-1]
        smooth = smoothing / (vocab - 1)
    batch, seq_len, _ = probs.shape
    losses = []
    for b in range(batch):
        for t in range(seq_len):
            if target_ids[b, t] != pad_id:
                if smoothing > 0:
                    true_prob = 1.0 - smoothing
                    loss = -(true_prob * math.log(probs[b, t, target_ids[b, t]] + 1e-9) +
                             smooth * np.sum(np.log(probs[b, t] + 1e-9)))
                else:
                    loss = -math.log(probs[b, t, target_ids[b, t]] + 1e-9)
                losses.append(loss)
    return sum(losses) / len(losses)

class AdamOptimizer:
    def __init__(self, params, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ----------------------------- Training utilities --------------------------

def compute_grads(loss, params):
    # Placeholder: autograd not implemented
    return [np.zeros_like(p) for p in params]

def save_checkpoint(model, optimizer, path):
    np.savez(path, **{f'param_{i}': p for i, p in enumerate(model.params)},
             **{f'm_{i}': m for i, m in enumerate(optimizer.m)},
             **{f'v_{i}': v for i, v in enumerate(optimizer.v)}, t=optimizer.t)

def train(model, data, epochs=1, batch_size=2, lr=1e-4, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = AdamOptimizer(model.params, lr=lr)
    for epoch in range(1, epochs + 1):
        pbar = tqdm(data)
        epoch_losses = []
        for batch in pbar:
            logits = model(batch['input_ids'], batch['target_ids'])
            loss = cross_entropy(logits, batch['target_ids'], smoothing=0.1)
            grads = compute_grads(loss, model.params)
            optimizer.step(grads)
            epoch_losses.append(loss)
            pbar.set_description(f'Epoch {epoch} Loss {loss:.4f}')
        with open('training.log', 'a') as f:
            f.write(f'Epoch {epoch}\t{np.mean(epoch_losses)}\n')
        save_checkpoint(model, optimizer, os.path.join(checkpoint_dir, f'epoch_{epoch}.npz'))

# ----------------------------- Decoding ------------------------------------

def greedy_decode(model, tokenizer, src, max_len=32):
    src_ids = tokenizer.encode(src, max_len).reshape(1, -1)
    tgt_ids = np.array([[tokenizer.token_to_id['<bos>']] + [tokenizer.token_to_id['<pad>']] * (max_len - 1)])
    for i in range(1, max_len):
        logits = model(src_ids, tgt_ids)
        next_id = int(np.argmax(logits[0, i-1]))
        tgt_ids[0, i] = next_id
        if next_id == tokenizer.token_to_id['<eos>']:
            break
    return tokenizer.decode(tgt_ids[0])

# ----------------------------- Data loading --------------------------------

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

# ----------------------------- Command line --------------------------------

def main():
    parser = argparse.ArgumentParser(description='Minimal NumPy Transformer chatbot')
    parser.add_argument('--mode', choices=['train', 'chat'], required=True)
    parser.add_argument('--train_file', default='data/train.jsonl')
    parser.add_argument('--valid_file', default='data/valid.jsonl')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=32)
    args = parser.parse_args()

    tokenizer = Tokenizer()

    if args.mode == 'train':
        train_data = load_jsonl(args.train_file)
        texts = [d['input'] for d in train_data] + [d['response'] for d in train_data]
        tokenizer.build_vocab(texts)
        # Dummy conversion for demonstration
        data_batches = []
        for d in train_data:
            data_batches.append({'input_ids': tokenizer.encode(d['input'], args.max_len),
                                 'target_ids': tokenizer.encode(d['response'], args.max_len)})
        model = Seq2SeqTransformer(num_layers=1, vocab_size=len(tokenizer.token_to_id), seq_len=args.max_len)
        train(model, data_batches, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, checkpoint_dir=args.checkpoint_dir)
    elif args.mode == 'chat':
        src = input('User: ')
        # minimal loading: create new model with vocab from saved files would be needed
        print('Model loading not implemented in this demo.')

if __name__ == '__main__':
    main()

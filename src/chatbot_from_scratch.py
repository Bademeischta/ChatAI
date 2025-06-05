import numpy as np
import math

# Tokenizer for building vocabulary and encoding/decoding
class Tokenizer:
    def __init__(self):
        self.token_to_id = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2
        }
        self.id_to_token = {0: '<pad>', 1: '<bos>', 2: '<eos>'}

    def build_vocab(self, texts):
        for line in texts:
            for token in self.tokenize(line):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def tokenize(self, text):
        text = text.replace('\t', ' ').strip()
        return text.split()

    def encode(self, text, max_len=32):
        tokens = ["<bos>"] + self.tokenize(text) + ["<eos>"]
        ids = [self.token_to_id.get(t, self.token_to_id['<pad>']) for t in tokens]
        if len(ids) < max_len:
            ids += [self.token_to_id['<pad>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return np.array(ids, dtype=np.int32)

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, '') for i in ids if i != self.token_to_id['<pad>']]
        return ' '.join(tokens)

# Simple Xavier initialization

def xavier_init(size):
    in_dim, out_dim = size
    limit = math.sqrt(6.0 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size)

# Layer Normalization
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

# Multi-Head Self-Attention
class MultiHeadSelfAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Weights
        self.W_q = xavier_init((embed_dim, embed_dim))
        self.W_k = xavier_init((embed_dim, embed_dim))
        self.W_v = xavier_init((embed_dim, embed_dim))
        self.W_o = xavier_init((embed_dim, embed_dim))

    def split_heads(self, x):
        batch, seq_len, dim = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch, heads, seq_len, dim = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(batch, seq_len, heads * dim)
        return x

    def __call__(self, x, mask=None):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        scores = Q @ K.transpose(0,1,3,2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores += mask
        weights = softmax(scores)
        heads = weights @ V
        out = self.combine_heads(heads)
        return out @ self.W_o

# Position Encoding using sinusoids
def positional_encoding(seq_len, dim):
    PE = np.zeros((seq_len, dim))
    for pos in range(seq_len):
        for i in range(0, dim, 2):
            angle = pos / np.power(10000, (2 * i)/dim)
            PE[pos, i] = math.sin(angle)
            if i + 1 < dim:
                PE[pos, i+1] = math.cos(angle)
    return PE

# Feed Forward Network
class FeedForward:
    def __init__(self, dim, hidden_dim):
        self.W1 = xavier_init((dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = xavier_init((hidden_dim, dim))
        self.b2 = np.zeros(dim)

    def __call__(self, x):
        h = relu(x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

# Transformer Block
class TransformerBlock:
    def __init__(self, dim, num_heads, hidden_dim):
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ln1 = LayerNorm(dim)
        self.ffn = FeedForward(dim, hidden_dim)
        self.ln2 = LayerNorm(dim)

    def __call__(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

# Softmax and Relu helpers
def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)

# Encoder
class TransformerEncoder:
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim):
        self.embedding = xavier_init((vocab_size, embed_dim))
        self.pe = positional_encoding(seq_len, embed_dim)
        self.layers = [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]

    def __call__(self, x_ids):
        x = self.embedding[x_ids] + self.pe[:len(x_ids[0])]
        for layer in self.layers:
            x = layer(x)
        return x

# Decoder
class TransformerDecoder:
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim):
        self.embedding = xavier_init((vocab_size, embed_dim))
        self.pe = positional_encoding(seq_len, embed_dim)
        self.layers = [TransformerBlock(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)]
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
    def __init__(self, num_layers, vocab_size, seq_len, embed_dim=64, num_heads=4, hidden_dim=128):
        self.encoder = TransformerEncoder(num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim)
        self.decoder = TransformerDecoder(num_layers, vocab_size, seq_len, embed_dim, num_heads, hidden_dim)
        # gather parameters for optimizer
        self.params = [self.encoder.embedding, self.decoder.embedding, self.decoder.fc_out]
        for block in self.encoder.layers + self.decoder.layers:
            self.params.extend([block.attn.W_q, block.attn.W_k, block.attn.W_v, block.attn.W_o,
                                block.ffn.W1, block.ffn.b1, block.ffn.W2, block.ffn.b2])

    def __call__(self, src_ids, tgt_ids):
        enc_out = self.encoder(src_ids)
        logits = self.decoder(tgt_ids, enc_out)
        return logits

# Cross Entropy Loss with masking

def cross_entropy(logits, target_ids, pad_id=0):
    probs = softmax(logits)
    batch, seq_len, vocab = probs.shape
    losses = []
    for b in range(batch):
        for t in range(seq_len):
            if target_ids[b, t] != pad_id:
                losses.append(-math.log(probs[b, t, target_ids[b, t]] + 1e-9))
    return sum(losses) / len(losses)

# Adam Optimizer
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

# Example skeleton for training loop

def train(model, data, epochs=1, batch_size=2):
    optimizer = AdamOptimizer(model.params)
    for epoch in range(epochs):
        for batch in data:
            logits = model(batch['input_ids'], batch['target_ids'])
            loss = cross_entropy(logits, batch['target_ids'])
            grads = compute_grads(loss, model.params)
            optimizer.step(grads)
        print(f"Epoch {epoch+1} finished")

# Placeholder function for gradient computation

def compute_grads(loss, params):
    return [np.zeros_like(p) for p in params]


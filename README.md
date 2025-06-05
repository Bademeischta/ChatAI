# ChatAI from Scratch

This repository contains a minimal example of building a Transformer-based chatbot completely from scratch using only NumPy. The code is located in `src/chatbot_from_scratch.py` and demonstrates how to implement tokenization, embeddings, attention layers, and an optimizer without relying on deep learning frameworks.

The implementation is simplified and serves as a starting point. It follows these steps:

1. Build a vocabulary with the `Tokenizer` class.
2. Define basic Transformer encoder and decoder blocks with multi-head attention and feed-forward networks.
3. Combine the encoder and decoder into a small `Seq2SeqTransformer` model.
4. Provide a skeleton for a training loop and an Adam optimizer implemented in NumPy.

The repository requirements are minimal; only `numpy` is needed. Install via:

```bash
pip install numpy
```

This project is intended for educational purposes to illustrate how a neural chatbot can be implemented without external machine learning libraries.

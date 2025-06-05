from __future__ import annotations

from typing import Iterable, List

import numpy as np


class AdamOptimizer:
    """Minimal Adam optimizer implementation."""

    def __init__(self, parameters: Iterable[np.ndarray], lr: float = 1e-4, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p) for p in self.parameters]
        self.v = [np.zeros_like(p) for p in self.parameters]
        self.t = 0

    def step(self, grads: List[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(self.parameters, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for m in self.m:
            m.fill(0)
        for v in self.v:
            v.fill(0)

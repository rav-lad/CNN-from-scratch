"""
src/layers/dropout.py
Dropout layer (inverted dropout).
- Active only during training.
- During inference, returns input unchanged.
"""

from __future__ import annotations
import numpy as np
from .base import Layer


class Dropout(Layer):
    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        assert 0.0 <= p < 1.0, "p must be in [0,1)"
        self.p = float(p)
        self.rng = np.random.default_rng() if rng is None else rng
        self._mask: np.ndarray | None = None
        self._scale: float = 1.0 / (1.0 - self.p) if self.p > 0 else 1.0  # inverted

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        # Bernoulli mask with keep prob (1-p)
        mask = self.rng.random(size=x.shape) >= self.p
        self._mask = mask
        return (x * mask) * self._scale

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            # eval mode -> identity
            return grad_out
        return (grad_out * self._mask) * self._scale

"""
src/layers/activations.py
Common activation layers with explicit backward:
- ReLU
- LeakyReLU
- Tanh
- Softmax (forward only typically; backward rarely used directly)

All cache only what is necessary for backward.
"""

from __future__ import annotations
import numpy as np
from .base import Layer


class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            raise RuntimeError("ReLU.backward called before forward.")
        return grad_out * self._mask


class LeakyReLU(Layer):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        self._x = x
        return np.where(x > 0, x, self.negative_slope * x)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("LeakyReLU.backward called before forward.")
        dx = np.ones_like(self._x)
        dx[self._x < 0] = self.negative_slope
        return grad_out * dx


class Tanh(Layer):
    def __init__(self) -> None:
        super().__init__()
        self._y: np.ndarray | None = None  # cache tanh(x)

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        y = np.tanh(x)
        self._y = y
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._y is None:
            raise RuntimeError("Tanh.backward called before forward.")
        # d/dx tanh(x) = 1 - tanh(x)^2
        return grad_out * (1.0 - self._y * self._y)


class Softmax(Layer):
    """
    Softmax over last dimension. Mostly useful for inference pipelines.
    For training use the numerically stable softmax cross entropy loss.
    """

    def __init__(self, axis: int = -1) -> None:
        super().__init__()
        self.axis = int(axis)
        self._out: np.ndarray | None = None  # cache probabilities

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        z = x - np.max(x, axis=self.axis, keepdims=True)
        exp_z = np.exp(z)
        out = exp_z / np.sum(exp_z, axis=self.axis, keepdims=True)
        self._out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Generic softmax backward is O(C^2). Rarely used since CE backward is simpler.
        Implements: dY = J_softmax * grad_out
        """
        if self._out is None:
            raise RuntimeError("Softmax.backward called before forward.")
        y = self._out
        # For each sample compute Jv = (diag(y) - y y^T) v along the last axis
        # Vectorized implementation
        # grad = grad_out - sum(grad_out * y) over classes, then multiply by y
        dot = np.sum(grad_out * y, axis=self.axis, keepdims=True)
        return y * (grad_out - dot)

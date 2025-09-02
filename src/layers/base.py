"""
src/layers/base.py
Abstract base for all layers.

Each layer must implement:
- forward(x, training=True) -> np.ndarray
- backward(grad_out) -> np.ndarray
- params() -> dict[str, np.ndarray]
- grads()  -> dict[str, np.ndarray]
- train() / eval() to switch behavior (e.g., Dropout, BatchNorm)

Conventions:
- Inputs are np.ndarray
- Forward caches any intermediates required for backward
- Backward returns grad wrt input with same shape as input
"""

from __future__ import annotations
import numpy as np
from typing import Dict


ParamDict = Dict[str, np.ndarray]


class Layer:
    def __init__(self) -> None:
        self.training: bool = True  # default in training mode

    # -------- lifecycle --------
    def train(self) -> None:
        """Switch to training mode (affects Dropout/BatchNorm)."""
        self.training = True

    def eval(self) -> None:
        """Switch to eval/inference mode."""
        self.training = False

    # -------- API to implement --------
    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        """
        Compute layer output for input x.
        If training is provided, overrides current mode.
        """
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate gradient from next layer to this layer's input.
        Must fill internal grad buffers for params, accessible via grads().
        """
        raise NotImplementedError

    # -------- parameters interface --------
    def params(self) -> ParamDict:
        """Return learnable parameters as a dict. Empty if none."""
        return {}

    def grads(self) -> ParamDict:
        """Return gradients wrt params with same keys/shapes as params()."""
        return {}

    # -------- utility checks --------
    @staticmethod
    def _assert_same_shape(a: np.ndarray, b: np.ndarray, msg: str = "") -> None:
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}. {msg}")

    @staticmethod
    def _require_4d(x: np.ndarray, name: str = "x") -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor for {name}, got {x.ndim}D with shape {x.shape}.")

    @staticmethod
    def _require_2d(x: np.ndarray, name: str = "x") -> None:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor for {name}, got {x.ndim}D with shape {x.shape}.")

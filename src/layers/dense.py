"""
src/layers/dense.py
Fully-connected layer (aka Linear / Affine).

Input:  x of shape (N, in_features)  or (N, ..., in_features) flattened internally
Weights: W of shape (out_features, in_features)
Bias:    b of shape (out_features,)
Output: y of shape (N, out_features)

Notes:
- If x has more than 2 dims, we flatten to (N, -1) and remember the original
  shape to properly reshape grad_input in backward.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

from .base import Layer, ParamDict
from ..core.initializers import xavier_uniform, he_normal, bias_zeros


class Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_init: str = "xavier_uniform",  # or "he_normal" if paired with ReLU
        rng: np.random.Generator | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        super().__init__()
        assert in_features > 0 and out_features > 0
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)
        self.dtype = dtype

        if weight_init == "xavier_uniform":
            self.W = xavier_uniform((out_features, in_features), rng=rng, dtype=dtype)
        elif weight_init == "he_normal":
            self.W = he_normal((out_features, in_features), rng=rng, dtype=dtype)
        else:
            raise ValueError(f"Unknown weight_init {weight_init}")

        self.b = bias_zeros((out_features,), dtype=dtype) if self.use_bias else None

        # grad buffers
        self._dW = np.zeros_like(self.W, dtype=dtype)
        self._db = np.zeros_like(self.b, dtype=dtype) if self.use_bias else None

        # cache
        self._x_2d: np.ndarray | None = None  # cached flattened input
        self._x_shape: Tuple[int, ...] | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        self._x_shape = x.shape
        if x.ndim > 2:
            N = x.shape[0]
            self._x_2d = x.reshape(N, -1).astype(self.dtype, copy=False)
        else:
            self._x_2d = x.astype(self.dtype, copy=False)
        y = self._x_2d @ self.W.T
        if self.use_bias and self.b is not None:
            y += self.b
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_2d is None or self._x_shape is None:
            raise RuntimeError("Dense.backward called before forward.")
        grad_out = grad_out.astype(self.dtype, copy=False)

        # dW = grad_out^T @ x
        self._dW = grad_out.T @ self._x_2d
        if self.use_bias and self._db is not None:
            self._db = np.sum(grad_out, axis=0)

        # dX = grad_out @ W
        grad_x_2d = grad_out @ self.W
        if len(self._x_shape) > 2:
            grad_x = grad_x_2d.reshape(self._x_shape)
        else:
            grad_x = grad_x_2d
        return grad_x

    def params(self) -> ParamDict:
        out: ParamDict = {"W": self.W}
        if self.use_bias and self.b is not None:
            out["b"] = self.b
        return out

    def grads(self) -> ParamDict:
        out: ParamDict = {"W": self._dW}
        if self.use_bias and self._db is not None:
            out["b"] = self._db
        return out

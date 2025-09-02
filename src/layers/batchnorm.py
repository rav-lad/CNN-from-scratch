"""
src/layers/batchnorm.py
Batch Normalization for 2D feature maps (N, C, H, W).
- Per-channel mean/var computed over N*H*W in training
- Running stats used in eval
- Learnable gamma (scale) and beta (shift)
"""

from __future__ import annotations
import numpy as np
from .base import Layer, ParamDict


class BatchNorm2D(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.9) -> None:
        super().__init__()
        assert num_features > 0
        self.C = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)

        # Learnable params
        self.gamma = np.ones((self.C,), dtype=np.float32)
        self.beta = np.zeros((self.C,), dtype=np.float32)

        # Running stats (for eval)
        self.running_mean = np.zeros((self.C,), dtype=np.float32)
        self.running_var = np.ones((self.C,), dtype=np.float32)

        # Grad buffers
        self._dgamma = np.zeros_like(self.gamma)
        self._dbeta = np.zeros_like(self.beta)

        # Cache
        self._x_centered: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None
        self._x_hat: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        if x.ndim != 4 or x.shape[1] != self.C:
            raise ValueError(f"BatchNorm2D expects (N,C,H,W) with C={self.C}, got {x.shape}")

        if self.training:
            # Compute per-channel mean/var over N*H*W
            N, C, H, W = x.shape
            x_resh = x.transpose(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)
            mean = x_resh.mean(axis=1)                       # (C,)
            var = x_resh.var(axis=1, ddof=0)                 # (C,)

            # Normalize
            x_centered = (x - mean[None, :, None, None])
            inv_std = 1.0 / np.sqrt(var + self.eps)
            x_hat = x_centered * inv_std[None, :, None, None]

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            # Cache for backward
            self._x_centered = x_centered
            self._inv_std = inv_std
            self._x_hat = x_hat

            y = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
            return y.astype(x.dtype)
        else:
            # Eval: use running stats
            x_centered = x - self.running_mean[None, :, None, None]
            x_hat = x_centered / np.sqrt(self.running_var[None, :, None, None] + self.eps)
            y = self.gamma[None, :, None, None] * x_hat + self.beta[None, :, None, None]
            return y.astype(x.dtype)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_centered is None or self._inv_std is None or self._x_hat is None:
            raise RuntimeError("BatchNorm2D.backward called before forward in training mode.")

        # shapes
        N, C, H, W = grad_out.shape
        axes = (0, 2, 3)  # reduction over N,H,W

        # grads w.r.t. scale/shift
        self._dgamma = np.sum(grad_out * self._x_hat, axis=axes)
        self._dbeta  = np.sum(grad_out,               axis=axes)

        # compact, numerically stable formula for dx
        gamma = self.gamma[None, :, None, None].astype(grad_out.dtype, copy=False)
        inv_std = self._inv_std[None, :, None, None].astype(grad_out.dtype, copy=False)
        x_hat = self._x_hat.astype(grad_out.dtype, copy=False)
        dy = grad_out.astype(grad_out.dtype, copy=False)

        # per-channel means across N,H,W
        mean_dy = np.mean(dy, axis=axes, keepdims=True)
        mean_dy_xhat = np.mean(dy * x_hat, axis=axes, keepdims=True)

        dx = (gamma * inv_std) * (dy - mean_dy - x_hat * mean_dy_xhat)
        return dx


    def params(self) -> ParamDict:
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self) -> ParamDict:
        return {"gamma": self._dgamma, "beta": self._dbeta}

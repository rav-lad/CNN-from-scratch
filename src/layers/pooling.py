"""
src/layers/pooling.py
Pooling layers: MaxPool2D and AvgPool2D.

Input / Output conventions:
- Input  x: (N, C, H, W)
- Output y: (N, C, H_out, W_out)
  where:
    H_out = (H - KH)//stride + 1   if no padding
    W_out = (W - KW)//stride + 1

Notes:
- MaxPool caches argmax indices for backward.
- AvgPool distributes gradient equally.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

from .base import Layer


class MaxPool2D(Layer):
    def __init__(self, kernel_size: Tuple[int, int] | int, stride: int | None = None) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]

        # Cache for backward
        self._x_shape: Tuple[int, int, int, int] | None = None
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        S = self.stride

        H_out = (H - KH) // S + 1
        W_out = (W - KW) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        mask = np.zeros_like(x, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                h_end, w_end = h_start + KH, w_start + KW
                window = x[:, :, h_start:h_end, w_start:w_end]
                max_vals = np.max(window, axis=(2, 3))
                out[:, :, i, j] = max_vals
                # Create mask for backward
                max_mask = (window == max_vals[:, :, None, None])
                mask[:, :, h_start:h_end, w_start:w_end] |= max_mask

        self._x_shape = x.shape
        self._mask = mask
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._mask is None:
            raise RuntimeError("MaxPool2D.backward called before forward.")

        N, C, H, W = self._x_shape
        KH, KW = self.kernel_size
        S = self.stride
        H_out, W_out = grad_out.shape[2], grad_out.shape[3]

        grad_x = np.zeros((N, C, H, W), dtype=grad_out.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                h_end, w_end = h_start + KH, w_start + KW
                mask_slice = self._mask[:, :, h_start:h_end, w_start:w_end]
                grad_slice = grad_out[:, :, i, j][:, :, None, None]
                grad_x[:, :, h_start:h_end, w_start:w_end] += grad_slice * mask_slice

        return grad_x


class AvgPool2D(Layer):
    def __init__(self, kernel_size: Tuple[int, int] | int, stride: int | None = None) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size[0]

        self._x_shape: Tuple[int, int, int, int] | None = None

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = training
        N, C, H, W = x.shape
        KH, KW = self.kernel_size
        S = self.stride

        H_out = (H - KH) // S + 1
        W_out = (W - KW) // S + 1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                h_end, w_end = h_start + KH, w_start + KW
                window = x[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.mean(window, axis=(2, 3))

        self._x_shape = x.shape
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_shape is None:
            raise RuntimeError("AvgPool2D.backward called before forward.")

        N, C, H, W = self._x_shape
        KH, KW = self.kernel_size
        S = self.stride
        H_out, W_out = grad_out.shape[2], grad_out.shape[3]

        grad_x = np.zeros((N, C, H, W), dtype=grad_out.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start, w_start = i * S, j * S
                h_end, w_end = h_start + KH, w_start + KW
                grad_slice = grad_out[:, :, i, j][:, :, None, None] / (KH * KW)
                grad_x[:, :, h_start:h_end, w_start:w_end] += grad_slice

        return grad_x

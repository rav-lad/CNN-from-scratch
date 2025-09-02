"""
src/layers/conv2d.py
Manual 2D convolution layer using im2col/col2im.

Input / Output conventions:
- Input  x: (N, C_in, H, W)
- Weights W: (C_out, C_in, KH, KW)
- Bias    b: (C_out,)
- Output y: (N, C_out, H_out, W_out)
  where:
    H_out = (H + 2*pad - KH)//stride + 1
    W_out = (W + 2*pad - KW)//stride + 1

No autograd. Backward implemented by hand.

Notes:
- We cache im2col result for efficient backward.
- Shapes are checked for clarity and early failure.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Literal

from .base import Layer, ParamDict
from ..core.utils import im2col, col2im
from ..core.initializers import he_normal, xavier_uniform, bias_zeros


InitKind = Literal["he_normal", "xavier_uniform"]


class Conv2D(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] | int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        weight_init: InitKind = "he_normal",
        rng: np.random.Generator | None = None,
        dtype: np.dtype = np.float64,
    ) -> None:
        super().__init__()
        assert in_channels > 0 and out_channels > 0, "channels must be positive"
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        KH, KW = kernel_size
        assert KH > 0 and KW > 0, "kernel dims must be positive"
        assert stride >= 1, "stride must be >= 1"
        assert padding >= 0, "padding must be >= 0"

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(KH), int(KW))
        self.stride = int(stride)
        self.padding = int(padding)
        self.use_bias = bool(bias)
        self.dtype = dtype

        # Parameters
        W_shape = (self.out_channels, self.in_channels, KH, KW)
        if weight_init == "he_normal":
            self.W = he_normal(W_shape, rng=rng, dtype=dtype)
        elif weight_init == "xavier_uniform":
            self.W = xavier_uniform(W_shape, rng=rng, dtype=dtype)
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        self.b = bias_zeros((self.out_channels,), dtype=dtype) if self.use_bias else None

        # Grad buffers
        self._dW = np.zeros_like(self.W, dtype=dtype)
        self._db = np.zeros_like(self.b, dtype=dtype) if self.use_bias else None

        # Cache for backward
        self._x_shape: Tuple[int, int, int, int] | None = None
        self._x_cols: np.ndarray | None = None
        self._out_hw: Tuple[int, int] | None = None

    def _calc_out_hw(self, H: int, W: int) -> Tuple[int, int]:
        KH, KW = self.kernel_size
        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"Invalid output size: got H_out={H_out}, W_out={W_out}. "
                f"Check kernel={self.kernel_size}, stride={self.stride}, padding={self.padding}."
            )
        return H_out, W_out

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            self.training = bool(training)
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N,C,H,W). Got shape {x.shape}.")
        x = x.astype(self.dtype, copy=False)
        N, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2D in_channels={self.in_channels} but got input with C={C}.")

        KH, KW = self.kernel_size
        H_out, W_out = self._calc_out_hw(H, W)
        x_cols = im2col(x, (KH, KW), stride=self.stride, pad=self.padding).astype(self.dtype, copy=False)

        W_row = self.W.reshape(self.out_channels, -1)
        out = x_cols @ W_row.T
        if self.use_bias:
            out += self.b

        self._x_shape = (N, C, H, W)
        self._x_cols = x_cols
        self._out_hw = (H_out, W_out)

        out = out.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self._x_shape is None or self._x_cols is None or self._out_hw is None:
            raise RuntimeError("Conv2D.backward called before forward or cache cleared.")

        grad_out = grad_out.astype(self.dtype, copy=False)

        N, C_in, H, W = self._x_shape
        H_out, W_out = self._out_hw
        KH, KW = self.kernel_size

        if grad_out.shape != (N, self.out_channels, H_out, W_out):
            raise ValueError(
                f"grad_out shape {grad_out.shape} does not match expected "
                f"(N={N}, C_out={self.out_channels}, H_out={H_out}, W_out={W_out})."
            )

        grad_cols_out = grad_out.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, self.out_channels)

        dW_row = grad_cols_out.T @ self._x_cols
        self._dW = dW_row.reshape(self.out_channels, C_in, KH, KW)

        if self.use_bias and self.b is not None:
            self._db = np.sum(grad_cols_out, axis=0)

        W_row = self.W.reshape(self.out_channels, -1)
        dX_cols = grad_cols_out @ W_row

        dX = col2im(dX_cols, self._x_shape, (KH, KW), stride=self.stride, pad=self.padding).astype(self.dtype, copy=False)
        return dX

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

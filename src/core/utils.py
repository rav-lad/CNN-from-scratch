"""
src/core/utils.py
Low-level utilities: im2col/col2im for convolutions, one-hot encoding,
mini-batch generator, reproducibility helpers.
"""

from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    """
    Fix NumPy's random seed for reproducibility.
    """
    np.random.seed(seed)


# ----------------------------
# One-hot encoding
# ----------------------------
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode integer labels.

    Args:
        y: shape (N,), dtype int
        num_classes: total classes

    Returns:
        arr: shape (N, num_classes)
    """
    y = y.astype(int).ravel()
    N = y.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float32)
    out[np.arange(N), y] = 1.0
    return out


# ----------------------------
# Mini-batching
# ----------------------------
def make_batches(n_samples: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """
    Yield slices (start, end) to iterate mini-batches.

    Args:
        n_samples: total number of samples
        batch_size: batch size

    Yields:
        (start, end) indices
    """
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield start, end


# ----------------------------
# im2col / col2im
# ----------------------------
def im2col(
    x: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    """
    Transform input image batch into 2D array of columns for fast conv.

    Args:
        x: shape (N, C, H, W)
        kernel_size: (KH, KW)
        stride: stride
        pad: zero padding

    Returns:
        cols: shape (N * out_h * out_w, C * KH * KW)
    """
    N, C, H, W = x.shape
    KH, KW = kernel_size

    # Pad input
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    out_h = (H + 2 * pad - KH) // stride + 1
    out_w = (W + 2 * pad - KW) // stride + 1

    cols = np.zeros((N, C, KH, KW, out_h, out_w), dtype=x.dtype)

    for i in range(KH):
        i_max = i + stride * out_h
        for j in range(KW):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x_padded[:, :, i:i_max:stride, j:j_max:stride]

    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return cols


def col2im(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    kernel_size: Tuple[int, int],
    stride: int,
    pad: int,
) -> np.ndarray:
    """
    Reverse operation of im2col. Reconstruct image batch.

    Args:
        cols: shape (N * out_h * out_w, C * KH * KW)
        x_shape: original (N, C, H, W)
        kernel_size: (KH, KW)
        stride: stride
        pad: padding

    Returns:
        x_reconstructed: shape (N, C, H, W)
    """
    N, C, H, W = x_shape
    KH, KW = kernel_size
    out_h = (H + 2 * pad - KH) // stride + 1
    out_w = (W + 2 * pad - KW) // stride + 1

    cols_reshaped = cols.reshape(N, out_h, out_w, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    x_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=cols.dtype)

    for i in range(KH):
        i_max = i + stride * out_h
        for j in range(KW):
            j_max = j + stride * out_w
            x_padded[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, :, i, j, :, :]

    if pad == 0:
        return x_padded
    return x_padded[:, :, pad:-pad, pad:-pad]

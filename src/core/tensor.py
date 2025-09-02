"""
src/core/tensor.py
Lightweight helpers around NumPy arrays:
- dtype management
- safety checks (finite)
- channel order conversions
- image normalization utilities
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, Tuple

DEFAULT_DTYPE = np.float32


def as_farray(x, dtype: np.dtype = DEFAULT_DTYPE, copy: bool = False) -> np.ndarray:
    """
    Convert input to contiguous NumPy array with desired dtype.

    Args:
        x: array-like
        dtype: target dtype
        copy: force a copy

    Returns:
        np.ndarray with dtype and C-order memory layout
    """
    arr = np.array(x, dtype=dtype, copy=copy, order="C")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def ensure_contiguous(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous view or copy."""
    return x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)


def assert_finite(x: np.ndarray, name: str = "tensor") -> None:
    """Raise ValueError if x has NaN or Inf."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non finite values")


# ------------- channel order conversions -------------
def to_nchw(x: np.ndarray) -> np.ndarray:
    """
    Convert NHWC to NCHW if needed.
    Accepts NCHW and returns it unchanged.
    """
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor, got shape {x.shape}")
    # Heuristic: if last dim is small (<= 4) and second dim is not small, assume NHWC
    if x.shape[-1] <= 4 and x.shape[1] > 4:
        return np.transpose(x, (0, 3, 1, 2))
    return x


def to_nhwc(x: np.ndarray) -> np.ndarray:
    """
    Convert NCHW to NHWC if needed.
    Accepts NHWC and returns it unchanged.
    """
    if x.ndim != 4:
        raise ValueError(f"expected 4D tensor, got shape {x.shape}")
    if x.shape[1] <= 4 and x.shape[-1] > 4:
        return np.transpose(x, (0, 2, 3, 1))
    return x


# ------------- image normalization (NCHW) -------------
def normalize_nchw(x: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    """
    Channel wise normalization for images in NCHW.

    Args:
        x: shape (N, C, H, W)
        mean: iterable of length C
        std: iterable of length C
    """
    if x.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {x.shape}")
    C = x.shape[1]
    mean_arr = np.asarray(list(mean), dtype=x.dtype).reshape(1, C, 1, 1)
    std_arr = np.asarray(list(std), dtype=x.dtype).reshape(1, C, 1, 1)
    return (x - mean_arr) / (std_arr + 1e-12)


def denormalize_nchw(x: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    """Inverse of normalize_nchw."""
    if x.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {x.shape}")
    C = x.shape[1]
    mean_arr = np.asarray(list(mean), dtype=x.dtype).reshape(1, C, 1, 1)
    std_arr = np.asarray(list(std), dtype=x.dtype).reshape(1, C, 1, 1)
    return x * (std_arr + 1e-12) + mean_arr


# ------------- small summaries -------------
def summary(x: np.ndarray) -> str:
    """Return a short summary string with shape, dtype, min, max, mean."""
    x = np.asarray(x)
    mn = float(np.min(x)) if x.size else float("nan")
    mx = float(np.max(x)) if x.size else float("nan")
    mu = float(np.mean(x)) if x.size else float("nan")
    return f"shape={x.shape} dtype={x.dtype} min={mn:.4g} max={mx:.4g} mean={mu:.4g}"


# ------------- padding helpers -------------
def pad2d(x: np.ndarray, pad: int) -> np.ndarray:
    """
    Zero pad a 4D NCHW tensor on H and W by pad.
    """
    if x.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {x.shape}")
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

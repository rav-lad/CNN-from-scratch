"""
src/core/initializers.py
Weight initialization utilities for Dense and Conv2D layers.

- Xavier/Glorot (normal & uniform)
- He/Kaiming (normal & uniform)
- Zeros / Ones / Constant
- Orthogonal (optional for Dense)

Conventions:
- Dense weights: (out_features, in_features)
- Conv2D weights: (C_out, C_in, KH, KW)
Biases are usually initialized to zeros.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def _fan_in_out(shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Compute fan_in and fan_out for Dense or Conv2D weights.

    Dense: (out_features, in_features)
    Conv2D: (C_out, C_in, KH, KW)
    """
    if len(shape) == 2:  # Dense
        out_f, in_f = shape
        fan_in, fan_out = in_f, out_f
    elif len(shape) == 4:  # Conv2D
        c_out, c_in, kh, kw = shape
        receptive_field = kh * kw
        fan_in = c_in * receptive_field
        fan_out = c_out * receptive_field
    else:
        n = int(np.prod(shape))
        fan_in = fan_out = int(np.sqrt(n))
    return fan_in, fan_out


# ----------------------------
# Basic initializers
# ----------------------------
def zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def ones(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
    return np.ones(shape, dtype=dtype)


def constant(shape: Tuple[int, ...], value: float, dtype: np.dtype = np.float64) -> np.ndarray:
    return np.full(shape, fill_value=value, dtype=dtype)


# ----------------------------
# Xavier / Glorot
# ----------------------------
def xavier_uniform(
    shape: Tuple[int, ...],
    rng: np.random.Generator | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Glorot uniform: U(-a, a) with a = sqrt(6 / (fan_in + fan_out))
    Good default for tanh and linear.
    """
    fan_in, fan_out = _fan_in_out(shape)
    a = np.sqrt(6.0 / (fan_in + fan_out))
    g = np.random.default_rng() if rng is None else rng
    return g.uniform(-a, a, size=shape).astype(dtype)


def xavier_normal(
    shape: Tuple[int, ...],
    rng: np.random.Generator | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Glorot normal: N(0, std^2) with std = sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _fan_in_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    g = np.random.default_rng() if rng is None else rng
    return g.normal(0.0, std, size=shape).astype(dtype)


# ----------------------------
# He / Kaiming (for ReLU-like activations)
# ----------------------------
def he_uniform(
    shape: Tuple[int, ...],
    rng: np.random.Generator | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Kaiming or He uniform: U(-a, a) with a = sqrt(6 / fan_in)
    Recommended for ReLU and LeakyReLU.
    """
    fan_in, _ = _fan_in_out(shape)
    a = np.sqrt(6.0 / fan_in)
    g = np.random.default_rng() if rng is None else rng
    return g.uniform(-a, a, size=shape).astype(dtype)


def he_normal(
    shape: Tuple[int, ...],
    rng: np.random.Generator | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Kaiming or He normal: N(0, std^2) with std = sqrt(2 / fan_in)
    """
    fan_in, _ = _fan_in_out(shape)
    std = np.sqrt(2.0 / fan_in)
    g = np.random.default_rng() if rng is None else rng
    return g.normal(0.0, std, size=shape).astype(dtype)


# ----------------------------
# Orthogonal (useful for Dense)
# ----------------------------
def orthogonal(
    shape: Tuple[int, ...],
    gain: float = 1.0,
    rng: np.random.Generator | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """
    Orthogonal initializer for 2D shapes. Falls back to Glorot for others.

    Args:
        shape: expect (out_features, in_features)
        gain: scaling factor
    """
    if len(shape) != 2:
        return xavier_uniform(shape, rng=rng, dtype=dtype)

    g = np.random.default_rng() if rng is None else rng
    rows, cols = shape
    a = g.normal(0.0, 1.0, size=(rows, cols)).astype(dtype)
    u, _, vt = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == (rows, cols) else vt
    q = q.astype(dtype)
    return (gain * q).astype(dtype)


# ----------------------------
# Bias helper
# ----------------------------
def bias_zeros(shape: Tuple[int, ...], dtype: np.dtype = np.float64) -> np.ndarray:
    return zeros(shape, dtype=dtype)

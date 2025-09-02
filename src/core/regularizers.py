"""
src/core/regularizers.py
Regularization utilities independent from the optimizer step.

Design notes:
- Weight decay can be applied inside optimizers (already supported) or
  you can add an explicit penalty term to the loss with these helpers.
- Exclude biases and BatchNorm parameters by default.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Iterable


ParamDict = Dict[str, np.ndarray]


def _exclude_name(name: str, exclude: Iterable[str]) -> bool:
    lname = name.lower()
    return any(tok in lname for tok in exclude)


def l2_penalty(params: ParamDict, exclude: Iterable[str] = ("bias", "b", "beta", "gamma")) -> float:
    """
    Sum of squared weights for selected params.
    You can add lambda * l2_penalty(...) to the data loss.

    Args:
        params: dict of parameter arrays
        exclude: parameter name substrings to exclude from penalty

    Returns:
        float penalty (no 0.5 factor included)
    """
    total = 0.0
    for k, v in params.items():
        if _exclude_name(k, exclude):
            continue
        total += float(np.sum(v * v))
    return total


def l1_penalty(params: ParamDict, exclude: Iterable[str] = ("bias", "b", "beta", "gamma")) -> float:
    """
    Sum of absolute weights for selected params.
    """
    total = 0.0
    for k, v in params.items():
        if _exclude_name(k, exclude):
            continue
        total += float(np.sum(np.abs(v)))
    return total


def max_norm(params: ParamDict, max_value: float = 3.0, exclude: Iterable[str] = ("bias", "b", "beta", "gamma")) -> None:
    mv = float(max_value)
    for k, v in params.items():
        lname = k.lower()
        if any(tok in lname for tok in exclude):
            continue
        norm = float(np.linalg.norm(v))
        if norm > mv and norm > 0.0:
            scale = mv / norm
            # in-place update to keep same ndarray object
            params[k][...] *= scale

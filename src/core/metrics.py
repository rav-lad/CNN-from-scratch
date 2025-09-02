"""
src/core/metrics.py
Basic metrics for classification.
"""

from __future__ import annotations
import numpy as np


def accuracy(logits_or_probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Top-1 accuracy for multiclass classification.

    Args:
        logits_or_probs: shape (N, C). Raw logits or probabilities.
        y_true: shape (N,), integer class labels.

    Returns:
        accuracy in [0, 1]
    """
    assert logits_or_probs.ndim == 2, "input must be (N, C)"
    assert y_true.ndim == 1, "y_true must be (N,)"
    y_pred = np.argmax(logits_or_probs, axis=1)
    return float(np.mean(y_pred == y_true))


def topk_accuracy(logits_or_probs: np.ndarray, y_true: np.ndarray, k: int = 5) -> float:
    """
    Top-k accuracy for multiclass classification.

    Args:
        logits_or_probs: shape (N, C). Raw logits or probabilities.
        y_true: shape (N,), integer class labels.
        k: number of highest scores considered correct

    Returns:
        top-k accuracy in [0, 1]
    """
    assert logits_or_probs.ndim == 2, "input must be (N, C)"
    assert y_true.ndim == 1, "y_true must be (N,)"
    N, C = logits_or_probs.shape
    k = int(max(1, min(k, C)))
    topk = np.argpartition(logits_or_probs, -k, axis=1)[:, -k:]
    # For each row, check if true label is among the top-k indices
    correct = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(correct))

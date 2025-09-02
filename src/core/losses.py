"""
src/core/losses.py
Manual losses with explicit forward and backward.
- Softmax Cross Entropy (stable)
- Mean Squared Error
No autograd. All gradients are derived and implemented by hand.
"""

from __future__ import annotations
import numpy as np


# ----------------------------
# Softmax Cross Entropy
# ----------------------------
def _softmax_stable(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax along last axis.

    Args:
        logits: shape (N, C)

    Returns:
        probs: shape (N, C), rows sum to 1
    """
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return probs


def softmax_cross_entropy(logits: np.ndarray, y_onehot: np.ndarray) -> float:
    """
    Forward pass of softmax cross entropy averaged over batch.

    Args:
        logits: shape (N, C), raw scores
        y_onehot: shape (N, C), one-hot targets

    Returns:
        loss (float): mean cross entropy over N
    """
    assert logits.ndim == 2, "logits must be (N, C)"
    assert y_onehot.shape == logits.shape, "y_onehot must match logits shape"
    # Stable log-softmax: log p = z - logsumexp
    z = logits - np.max(logits, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    log_probs = z - logsumexp
    # Cross entropy: -sum y * log p
    ce = -np.sum(y_onehot * log_probs, axis=1)
    loss = float(np.mean(ce))
    return loss


def softmax_cross_entropy_backward(logits: np.ndarray, y_onehot: np.ndarray) -> np.ndarray:
    """
    Backward pass: gradient of mean CE wrt logits.

    dL/dlogits = (softmax(logits) - y) / N

    Args:
        logits: shape (N, C)
        y_onehot: shape (N, C)

    Returns:
        grad_logits: shape (N, C)
    """
    N = logits.shape[0]
    probs = _softmax_stable(logits)
    grad = (probs - y_onehot) / N
    return grad


# ----------------------------
# Mean Squared Error
# ----------------------------
def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Mean squared error averaged over all elements.

    Args:
        pred: arbitrary shape
        target: same shape as pred

    Returns:
        loss (float)
    """
    assert pred.shape == target.shape, "pred and target must have same shape"
    diff = pred - target
    return float(np.mean(diff * diff))


def mse_backward(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Gradient of mean squared error wrt pred.

    dL/dpred = 2*(pred - target) / num_elements

    Args:
        pred: arbitrary shape
        target: same shape

    Returns:
        grad_pred: same shape as pred
    """
    num = pred.size
    return (2.0 / num) * (pred - target)

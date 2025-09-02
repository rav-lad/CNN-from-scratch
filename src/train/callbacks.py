"""
src/train/callbacks.py
Common training callbacks.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Dict


def EarlyStopping(monitor: str = "val_loss", patience: int = 5, mode: str = "min"):
    best = np.inf if mode == "min" else -np.inf
    wait = 0
    stopped = False

    def cb(state: Dict):
        nonlocal best, wait, stopped
        value = float(state[monitor])
        improved = value < best if mode == "min" else value > best
        if improved:
            best = value
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stopped = True
                print(f"EarlyStopping: no improvement in {patience} epochs on {monitor}.")
                # Optionally, you could raise an exception to break outer loop.
    cb.stopped = lambda: stopped  # type: ignore[attr-defined]
    return cb


def ModelCheckpoint(filepath: str, monitor: str = "val_acc", mode: str = "max"):
    best = -np.inf if mode == "max" else np.inf

    def save_weights(model, path: str):
        params = model.params()
        np.savez(path, **{k: v for k, v in params.items()})
        print(f"Saved checkpoint to {path}")

    def cb(state: Dict):
        nonlocal best
        value = float(state[monitor])
        improved = value > best if mode == "max" else value < best
        if improved:
            best = value
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            save_weights(state["model"], filepath)
    return cb


def ReduceLROnPlateau(optimizer, monitor: str = "val_loss", factor: float = 0.5, patience: int = 3, mode: str = "min"):
    best = np.inf if mode == "min" else -np.inf
    wait = 0

    def cb(state: Dict):
        nonlocal best, wait
        value = float(state[monitor])
        improved = value < best if mode == "min" else value > best
        if improved:
            best = value
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                optimizer.lr *= factor
                wait = 0
                print(f"ReduceLROnPlateau: lr -> {optimizer.lr:.3e}")
    return cb

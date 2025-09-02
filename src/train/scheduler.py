"""
src/train/scheduler.py
Learning rate schedulers for manual training loops.

Usage pattern:
    scheduler = build_scheduler(optimizer, cfg)
    for epoch in range(1, epochs+1):
        ...
        scheduler.step(epoch)

Supported:
- StepLR(step_size, gamma)
- CosineAnnealingLR(T_max, min_lr=0.0)
- WarmupCosineLR(warmup_epochs, T_max, max_lr=None, base_lr=None, min_lr=0.0)

Notes:
- Optimizer must expose an attribute `lr` that we update in place.
"""

from __future__ import annotations
import math
from typing import Optional


class LRScheduler:
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer

    def step(self, epoch: int) -> None:
        raise NotImplementedError

    def _set_lr(self, lr: float) -> None:
        lr = float(lr)
        if lr <= 0:
            lr = 1e-12
        self.optimizer.lr = lr


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int = 10, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        self.base_lr = float(optimizer.lr)

    def step(self, epoch: int) -> None:
        k = epoch // self.step_size
        new_lr = self.base_lr * (self.gamma ** k)
        self._set_lr(new_lr)


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, min_lr: float = 0.0) -> None:
        super().__init__(optimizer)
        self.T_max = int(T_max)
        self.min_lr = float(min_lr)
        self.max_lr = float(optimizer.lr)

    def step(self, epoch: int) -> None:
        # epoch in [1, T_max]
        t = min(max(epoch, 1), self.T_max)
        cos_inner = math.pi * (t - 1) / self.T_max
        new_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(cos_inner))
        self._set_lr(new_lr)


class WarmupCosineLR(LRScheduler):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to min_lr.

    Args:
        warmup_epochs: number of warmup steps
        T_max: total epochs for cosine (after warmup)
        max_lr: peak lr at the end of warmup (defaults to optimizer.lr)
        base_lr: lr at epoch 1 (defaults to optimizer.lr if max_lr is given)
        min_lr: floor lr for cosine phase
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int = 5,
        T_max: int = 50,
        max_lr: Optional[float] = None,
        base_lr: Optional[float] = None,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__(optimizer)
        self.warmup_epochs = int(warmup_epochs)
        self.T_max = int(T_max)
        self.base_lr = float(base_lr) if base_lr is not None else float(optimizer.lr)
        self.max_lr = float(max_lr) if max_lr is not None else float(optimizer.lr)
        self.min_lr = float(min_lr)

    def step(self, epoch: int) -> None:
        if epoch <= self.warmup_epochs:
            # linear warmup from base_lr to max_lr
            if self.warmup_epochs <= 0:
                self._set_lr(self.max_lr)
                return
            alpha = epoch / float(self.warmup_epochs)
            lr = self.base_lr + alpha * (self.max_lr - self.base_lr)
            self._set_lr(lr)
        else:
            # cosine from max_lr to min_lr over T_max epochs
            t = min(epoch - self.warmup_epochs, self.T_max)
            cos_inner = math.pi * (t) / self.T_max
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(cos_inner))
            self._set_lr(lr)


def build_scheduler(optimizer, cfg: dict | None):
    """
    Factory. cfg example:
        {"name": "step", "step_size": 5, "gamma": 0.5}
        {"name": "cosine", "T_max": 50, "min_lr": 1e-5}
        {"name": "warmup_cosine", "warmup_epochs": 5, "T_max": 50, "min_lr": 1e-5}

    Returns a scheduler instance or None if cfg is empty.
    """
    if not cfg:
        return None
    name = str(cfg.get("name", "")).lower()
    if name == "step":
        return StepLR(optimizer, step_size=int(cfg.get("step_size", 10)), gamma=float(cfg.get("gamma", 0.1)))
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=int(cfg.get("T_max", 50)), min_lr=float(cfg.get("min_lr", 0.0)))
    if name == "warmup_cosine":
        return WarmupCosineLR(
            optimizer,
            warmup_epochs=int(cfg.get("warmup_epochs", 5)),
            T_max=int(cfg.get("T_max", 50)),
            max_lr=float(cfg.get("max_lr", optimizer.lr)),
            base_lr=float(cfg.get("base_lr", optimizer.lr)),
            min_lr=float(cfg.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unknown scheduler name: {name}")

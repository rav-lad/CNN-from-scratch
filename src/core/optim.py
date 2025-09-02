"""
src/core/optim.py
Simple optimizers implemented manually:
- SGD (with momentum, Nesterov optional)
- Adam

Each optimizer exposes:
- step(params, grads): in-place update of params given grads
- zero_like(params): utility to create grad buffers with same shapes
- state_dict() / load_state_dict(): to save/restore optimizer state
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


ParamDict = Dict[str, np.ndarray]


class Optimizer:
    def step(self, params: ParamDict, grads: ParamDict) -> None:
        raise NotImplementedError

    def zero_like(self, params: ParamDict) -> ParamDict:
        return {k: np.zeros_like(v) for k, v in params.items()}

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum and Nesterov.
    L2 weight decay implemented as additive grad term: grad += wd * param
    """

    def __init__(
        self,
        lr: float = 1e-2,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
    ):
        assert lr > 0, "lr must be positive"
        assert momentum >= 0, "momentum must be non-negative"
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)
        self.clip_grad_norm = clip_grad_norm
        self._velocity: ParamDict | None = None

    def _maybe_clip(self, grads: ParamDict) -> None:
        if self.clip_grad_norm is None:
            return
        # global L2 norm
        total_sq = 0.0
        for g in grads.values():
            total_sq += float(np.sum(g * g))
        norm = np.sqrt(total_sq) + 1e-12
        if norm > self.clip_grad_norm:
            scale = self.clip_grad_norm / norm
            for k in grads:
                grads[k] *= scale

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        if self._velocity is None:
            self._velocity = {k: np.zeros_like(v) for k, v in params.items()}

        # Add weight decay to grads (L2)
        if self.weight_decay > 0.0:
            for k in grads:
                grads[k] = grads[k] + self.weight_decay * params[k]

        self._maybe_clip(grads)

        if self.momentum == 0.0:
            # Plain SGD
            for k in params:
                params[k] -= self.lr * grads[k]
            return

        # Momentum update
        for k in params:
            v = self._velocity[k]
            v *= self.momentum
            v += grads[k]
            if self.nesterov:
                # p <- p - lr * (mu*v + grad)
                params[k] -= self.lr * (self.momentum * v + grads[k])
            else:
                # p <- p - lr * v
                params[k] -= self.lr * v

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "nesterov": self.nesterov,
            "weight_decay": self.weight_decay,
            "clip_grad_norm": self.clip_grad_norm,
            "velocity": None if self._velocity is None else {k: v.copy() for k, v in self._velocity.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = float(state["lr"])
        self.momentum = float(state["momentum"])
        self.nesterov = bool(state["nesterov"])
        self.weight_decay = float(state["weight_decay"])
        self.clip_grad_norm = state["clip_grad_norm"]
        vel = state.get("velocity", None)
        if vel is None:
            self._velocity = None
        else:
            self._velocity = {k: v.copy() for k, v in vel.items()}


class Adam(Optimizer):
    """
    Adam optimizer.
    - Bias corrections on m, v.
    - Weight decay as additive grad term (classic L2, not decoupled).
    """

    def __init__(
        self,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        clip_grad_norm: float | None = None,
    ):
        assert lr > 0, "lr must be positive"
        b1, b2 = betas
        assert 0 <= b1 < 1 and 0 <= b2 < 1, "betas must be in [0,1)"
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.clip_grad_norm = clip_grad_norm

        self._m: ParamDict | None = None
        self._v: ParamDict | None = None
        self._t: int = 0

    def _maybe_clip(self, grads: ParamDict) -> None:
        if self.clip_grad_norm is None:
            return
        total_sq = 0.0
        for g in grads.values():
            total_sq += float(np.sum(g * g))
        norm = np.sqrt(total_sq) + 1e-12
        if norm > self.clip_grad_norm:
            scale = self.clip_grad_norm / norm
            for k in grads:
                grads[k] *= scale

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        if self._m is None:
            self._m = {k: np.zeros_like(v) for k, v in params.items()}
        if self._v is None:
            self._v = {k: np.zeros_like(v) for k, v in params.items()}

        # Add weight decay to grads (L2)
        if self.weight_decay > 0.0:
            for k in grads:
                grads[k] = grads[k] + self.weight_decay * params[k]

        self._maybe_clip(grads)

        self._t += 1
        b1, b2 = self.b1, self.b2

        for k in params:
            g = grads[k]
            m = self._m[k] = b1 * self._m[k] + (1 - b1) * g
            v = self._v[k] = b2 * self._v[k] + (1 - b2) * (g * g)

            # Bias correction
            m_hat = m / (1 - b1**self._t)
            v_hat = v / (1 - b2**self._t)

            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lr": self.lr,
            "b1": self.b1,
            "b2": self.b2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "clip_grad_norm": self.clip_grad_norm,
            "t": self._t,
            "m": None if self._m is None else {k: v.copy() for k, v in self._m.items()},
            "v": None if self._v is None else {k: v.copy() for k, v in self._v.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.lr = float(state["lr"])
        self.b1 = float(state["b1"])
        self.b2 = float(state["b2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        self.clip_grad_norm = state["clip_grad_norm"]
        self._t = int(state.get("t", 0))
        self._m = None if state.get("m", None) is None else {k: v.copy() for k, v in state["m"].items()}
        self._v = None if state.get("v", None) is None else {k: v.copy() for k, v in state["v"].items()}

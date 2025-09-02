"""
src/models/sequential.py
Lightweight sequential container to stack Layer instances.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict
from ..layers.base import Layer, ParamDict


class Sequential(Layer):
    def __init__(self, layers: List[Layer]) -> None:
        super().__init__()
        if not layers:
            raise ValueError("Sequential requires at least one layer.")
        self.layers = layers

    def train(self) -> None:
        self.training = True
        for l in self.layers:
            l.train()

    def eval(self) -> None:
        self.training = False
        for l in self.layers:
            l.eval()

    def forward(self, x: np.ndarray, training: bool | None = None) -> np.ndarray:
        if training is not None:
            if training:
                self.train()
            else:
                self.eval()
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        grad = grad_out
        for l in reversed(self.layers):
            grad = l.backward(grad)
        return grad

    def params(self) -> ParamDict:
        out: ParamDict = {}
        for i, l in enumerate(self.layers):
            for k, v in l.params().items():
                out[f"{i}.{l.__class__.__name__}.{k}"] = v
        return out

    def grads(self) -> ParamDict:
        out: ParamDict = {}
        for i, l in enumerate(self.layers):
            for k, v in l.grads().items():
                out[f"{i}.{l.__class__.__name__}.{k}"] = v
        return out

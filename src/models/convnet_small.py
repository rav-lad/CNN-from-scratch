"""
src/models/convnet_small.py
Tiny preset models to get training quickly:
- lenet_mnist()
- vgg_tiny_cifar10()
"""

from __future__ import annotations
from ..layers.conv2d import Conv2D
from ..layers.pooling import MaxPool2D
from ..layers.batchnorm import BatchNorm2D
from ..layers.dropout import Dropout
from ..layers.activations import ReLU
from ..layers.dense import Dense
from .sequential import Sequential


def lenet_mnist(num_classes: int = 10) -> Sequential:
    """
    Input: (N, 1, 28, 28)
    """
    return Sequential([
        Conv2D(1, 6, 5, padding=2),   # -> (N, 6, 28, 28)
        ReLU(),
        MaxPool2D(2),                 # -> (N, 6, 14, 14)

        Conv2D(6, 16, 5),             # -> (N, 16, 10, 10)
        ReLU(),
        MaxPool2D(2),                 # -> (N, 16, 5, 5)

        # flatten happens in Dense
        Dense(16*5*5, 120, weight_init="he_normal"),
        ReLU(),
        Dense(120, 84, weight_init="he_normal"),
        ReLU(),
        Dropout(0.3),
        Dense(84, num_classes, weight_init="xavier_uniform"),
    ])


def vgg_tiny_cifar10(num_classes: int = 10) -> Sequential:
    """
    Input: (N, 3, 32, 32)
    """
    return Sequential([
        Conv2D(3, 32, 3, padding=1),
        BatchNorm2D(32),
        ReLU(),
        Conv2D(32, 32, 3, padding=1),
        BatchNorm2D(32),
        ReLU(),
        MaxPool2D(2),       # 16x16
        Dropout(0.25),

        Conv2D(32, 64, 3, padding=1),
        BatchNorm2D(64),
        ReLU(),
        Conv2D(64, 64, 3, padding=1),
        BatchNorm2D(64),
        ReLU(),
        MaxPool2D(2),       # 8x8
        Dropout(0.25),

        Dense(64*8*8, 256, weight_init="he_normal"),
        ReLU(),
        Dropout(0.5),
        Dense(256, num_classes, weight_init="xavier_uniform"),
    ])

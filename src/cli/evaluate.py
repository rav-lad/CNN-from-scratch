"""
src/cli/evaluate.py
Evaluate a trained model on the test set given a config and weights.
"""

from __future__ import annotations
import argparse
import yaml
import numpy as np
from ..models.convnet_small import lenet_mnist, vgg_tiny_cifar10
from ..models.sequential import Sequential
from ..core.utils import set_seed
from ..core.metrics import accuracy, topk_accuracy
from ..data.mnist import load_mnist
from ..data.cifar10 import load_cifar10


def build_model(name: str, num_classes: int) -> Sequential:
    name = name.lower()
    if name == "lenet_mnist":
        return lenet_mnist(num_classes)
    if name == "vgg_tiny_cifar10":
        return vgg_tiny_cifar10(num_classes)
    raise ValueError(f"Unknown model {name}")


def load_weights(model: Sequential, path: str) -> None:
    data = np.load(path)
    params = model.params()
    # assign by matching keys
    for k in params.keys():
        if k in data:
            params[k][...] = data[k]
        else:
            raise KeyError(f"Weight key {k} not found in checkpoint {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))

    dataset = cfg.get("dataset", "mnist").lower()
    if dataset == "mnist":
        (_, _), (_, _), (X_test, y_test), num_classes = load_mnist()
        model_name = cfg.get("model", "lenet_mnist")
    elif dataset == "cifar10":
        (_, _), (_, _), (X_test, y_test), num_classes = load_cifar10()
        model_name = cfg.get("model", "vgg_tiny_cifar10")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    model = build_model(model_name, num_classes)
    load_weights(model, args.weights)

    model.eval()
    logits = model.forward(X_test, training=False)
    acc = accuracy(logits, y_test)
    top5 = topk_accuracy(logits, y_test, k=5)
    print(f"Test accuracy: {acc:.4f}, Top-5: {top5:.4f}")


if __name__ == "__main__":
    main()

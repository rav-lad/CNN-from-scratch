"""
src/cli/export.py
Export model weights to .npz and a simple JSON with meta info.
"""

from __future__ import annotations
import argparse
import json
import yaml
import numpy as np
from ..models.convnet_small import lenet_mnist, vgg_tiny_cifar10
from ..models.sequential import Sequential
from ..core.utils import set_seed
from ..data.mnist import load_mnist
from ..data.cifar10 import load_cifar10


def build_model(name: str, num_classes: int) -> Sequential:
    name = name.lower()
    if name == "lenet_mnist":
        return lenet_mnist(num_classes)
    if name == "vgg_tiny_cifar10":
        return vgg_tiny_cifar10(num_classes)
    raise ValueError(f"Unknown model {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--arch_out", type=str, default="checkpoints/arch.json")
    parser.add_argument("--weights_out", type=str, default="checkpoints/weights_exported.npz")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))

    dataset = cfg.get("dataset", "mnist").lower()
    if dataset == "mnist":
        (_, _), (_, _), (_, _), num_classes = load_mnist()
        model_name = cfg.get("model", "lenet_mnist")
    elif dataset == "cifar10":
        (_, _), (_, _), (_, _), num_classes = load_cifar10()
        model_name = cfg.get("model", "vgg_tiny_cifar10")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    model = build_model(model_name, num_classes)

    # load weights from an existing checkpoint then re-save them in a clean file
    ckpt = np.load(args.weights)
    params = model.params()
    for k in params.keys():
        if k in ckpt:
            params[k][...] = ckpt[k]
        else:
            raise KeyError(f"Key {k} not found in {args.weights}")

    # Save architecture metadata and weights
    arch = {"model": model_name, "dataset": dataset, "num_classes": num_classes}
    with open(args.arch_out, "w") as f:
        json.dump(arch, f, indent=2)
    np.savez(args.weights_out, **params)
    print(f"Exported arch to {args.arch_out} and weights to {args.weights_out}")


if __name__ == "__main__":
    main()

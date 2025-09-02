"""
src/cli/train.py
Train a preset model on MNIST or CIFAR-10 using YAML config.
"""

from __future__ import annotations
import argparse
import yaml
import numpy as np
from ..models.convnet_small import lenet_mnist, vgg_tiny_cifar10
from ..models.sequential import Sequential
from ..core.optim import SGD, Adam
from ..core.utils import set_seed
from ..train.loop import train
from ..train.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from ..data.mnist import load_mnist
from ..data.cifar10 import load_cifar10


def build_model(name: str, num_classes: int) -> Sequential:
    name = name.lower()
    if name == "lenet_mnist":
        return lenet_mnist(num_classes)
    if name == "vgg_tiny_cifar10":
        return vgg_tiny_cifar10(num_classes)
    raise ValueError(f"Unknown model {name}")


def build_optimizer(cfg: dict):
    opt = cfg.get("optimizer", "sgd").lower()
    lr = float(cfg.get("lr", 1e-2))
    wd = float(cfg.get("weight_decay", 0.0))
    if opt == "sgd":
        return SGD(lr=lr, momentum=float(cfg.get("momentum", 0.0)), weight_decay=wd)
    if opt == "adam":
        return Adam(lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer {opt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    dataset = cfg.get("dataset", "mnist").lower()
    if dataset == "mnist":
        (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = load_mnist()
        model_name = cfg.get("model", "lenet_mnist")
    elif dataset == "cifar10":
        (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = load_cifar10()
        model_name = cfg.get("model", "vgg_tiny_cifar10")
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    model = build_model(model_name, num_classes)
    optimizer = build_optimizer(cfg.get("train", {}))

    cbs = []
    for item in (cfg.get("callbacks") or []):
        if "early_stopping" in item:
            p = item["early_stopping"]
            cbs.append(EarlyStopping(monitor=p.get("monitor", "val_loss"),
                                     patience=int(p.get("patience", 5)),
                                     mode="min" if "loss" in p.get("monitor", "val_loss") else "max"))
        if "checkpoint" in item:
            p = item["checkpoint"]
            cbs.append(ModelCheckpoint(filepath=p.get("filepath", "checkpoints/best.npz"),
                                       monitor=p.get("monitor", "val_acc"),
                                       mode="max"))
        if "reduce_lr_on_plateau" in item:
            p = item["reduce_lr_on_plateau"]
            cbs.append(ReduceLROnPlateau(optimizer,
                                         monitor=p.get("monitor", "val_loss"),
                                         factor=float(p.get("factor", 0.5)),
                                         patience=int(p.get("patience", 3)),
                                         mode="min" if "loss" in p.get("monitor", "val_loss") else "max"))

    hist = train(
        model,
        optimizer,
        (X_train, y_train),
        (X_val, y_val),
        epochs=int(cfg["train"]["epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        num_classes=num_classes,
        log_csv_path="reports/results.csv",
        callbacks=cbs,
    )

    # Optionally evaluate on test set here or via separate CLI
    print("Training done.")


if __name__ == "__main__":
    main()

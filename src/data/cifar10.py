"""
src/data/cifar10.py
Download and load CIFAR-10 as NCHW float32 in [0,1], with train/val/test splits.
"""

from __future__ import annotations
import os
import tarfile
import pickle
import urllib.request
from typing import Tuple
import numpy as np

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    tmp = path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, path)

def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    X = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # NCHW
    y = np.array(d["labels"], dtype=np.int64)
    return X, y

def load_cifar10(data_dir: str = "data", val_ratio: float = 0.1, seed: int = 42) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    int,
]:
    root = os.path.join(data_dir, "cifar10")
    os.makedirs(root, exist_ok=True)
    tgz = os.path.join(root, "cifar-10-python.tar.gz")
    _download(CIFAR10_URL, tgz)

    extract_dir = os.path.join(root, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(path=root)

    # load train batches
    X_list, y_list = [], []
    for i in range(1, 6):
        Xb, yb = _load_batch(os.path.join(extract_dir, f"data_batch_{i}"))
        X_list.append(Xb)
        y_list.append(yb)
    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    # test
    X_test, y_test = _load_batch(os.path.join(extract_dir, "test_batch"))

    # train/val split
    rng = np.random.default_rng(seed)
    N = X_train.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(N * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_val, y_val = X_train[val_idx], y_train[val_idx]
    X_train, y_train = X_train[train_idx], y_train[train_idx]

    num_classes = 10
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes

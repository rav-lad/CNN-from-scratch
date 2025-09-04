
# src/data/mnist.py (en-tête + download helpers)

from __future__ import annotations
import os
import gzip
import hashlib
import time
import urllib.request
from typing import Tuple, List
import numpy as np

# Miroirs (ordre de préférence). MD5 officiels conservés.
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",      # Google (HTTPS)
    "http://yann.lecun.com/exdb/mnist",                         # Site original (HTTP, parfois 404)
]
MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images":  ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels":  ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c"),
}

def _download_with_retries(urls: List[str], path: str, md5: str | None = None, retries: int = 3, sleep: float = 1.0) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    last_err = None
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                tmp = path + ".tmp"
                # User-Agent pour éviter certains blocages
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
                    out.write(resp.read())
                if md5 is not None:
                    with open(tmp, "rb") as f:
                        data = f.read()
                    if hashlib.md5(data).hexdigest() != md5:
                        os.remove(tmp)
                        raise RuntimeError(f"MD5 mismatch for {url}")
                os.rename(tmp, path)
                return
            except Exception as e:
                last_err = e
                if os.path.exists(tmp := path + ".tmp"):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass
                if attempt < retries:
                    time.sleep(sleep)
                else:
                    # essaie prochain miroir
                    break
    raise RuntimeError(f"Failed to download after trying all mirrors. Last error: {last_err}")

def _download_all(root: str) -> None:
    for key, (fname, md5) in MNIST_FILES.items():
        urls = [f"{base}/{fname}" for base in MNIST_MIRRORS]
        _download_with_retries(urls, os.path.join(root, fname), md5)

def _load_idx_images(path_gz: str) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise RuntimeError("Invalid MNIST image file")
    N = int.from_bytes(data[4:8], "big")
    H = int.from_bytes(data[8:12], "big")
    W = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(N, H, W)
    # to NCHW float32 in [0,1]
    arr = arr.astype(np.float32) / 255.0
    arr = arr[:, None, :, :]
    return arr

def _load_idx_labels(path_gz: str) -> np.ndarray:
    with gzip.open(path_gz, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise RuntimeError("Invalid MNIST label file")
    N = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(N,)
    return arr.astype(np.int64)

def load_mnist(data_dir: str = "data", val_ratio: float = 0.1, seed: int = 42) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    int,
]:
    root = os.path.join(data_dir, "mnist")
    os.makedirs(root, exist_ok=True)

    # Téléchargement unique via les miroirs
    _download_all(root)

    # Lecture des fichiers gz
    X_train = _load_idx_images(os.path.join(root, "train-images-idx3-ubyte.gz"))
    y_train = _load_idx_labels(os.path.join(root, "train-labels-idx1-ubyte.gz"))
    X_test  = _load_idx_images(os.path.join(root, "t10k-images-idx3-ubyte.gz"))
    y_test  = _load_idx_labels(os.path.join(root, "t10k-labels-idx1-ubyte.gz"))

    # Split train/val
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

"""
src/train/loop.py
Generic training and evaluation loops with callbacks and logging hooks.
"""

from __future__ import annotations
import csv
import time
from typing import Callable, Dict, Iterable, Tuple
import numpy as np

from ..models.sequential import Sequential
from ..core.losses import softmax_cross_entropy, softmax_cross_entropy_backward
from ..core.metrics import accuracy
from ..core.utils import make_batches, one_hot


BatchIter = Iterable[Tuple[np.ndarray, np.ndarray]]
Callback = Callable[[Dict], None]


def train(
    model: Sequential,
    optimizer,
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray] | None,
    epochs: int = 10,
    batch_size: int = 128,
    num_classes: int = 10,
    log_csv_path: str | None = None,
    callbacks: list[Callback] | None = None,
    scheduler=None,
) -> Dict[str, list[float]]:
    X_train, y_train = train_data
    X_val, y_val = val_data if val_data is not None else (None, None)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    if log_csv_path is not None:
        with open(log_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    model.train()
    N = X_train.shape[0]
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # training
        perm = np.random.permutation(N)
        X_train, y_train = X_train[perm], y_train[perm]

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for start, end in make_batches(N, batch_size):
            xb = X_train[start:end]
            yb = y_train[start:end]
            logits = model.forward(xb, training=True)  # (B, C)
            y_one = one_hot(yb, num_classes)
            loss = softmax_cross_entropy(logits, y_one)
            grad_logits = softmax_cross_entropy_backward(logits, y_one)

            # backward
            grad = grad_logits
            grad = model.backward(grad)

            # params update
            params = model.params()
            grads = model.grads()
            optimizer.step(params, grads)

            # metrics
            total_loss += loss * (end - start)
            preds = np.argmax(logits, axis=1)
            total_correct += int(np.sum(preds == yb))
            total_seen += (end - start)

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen

        # validation
        if X_val is not None:
            model.eval()
            val_logits = []
            val_targets = []
            for start, end in make_batches(X_val.shape[0], batch_size):
                xb = X_val[start:end]
                logits = model.forward(xb, training=False)
                val_logits.append(logits)
                val_targets.append(y_val[start:end])
            val_logits = np.concatenate(val_logits, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            y_one = one_hot(val_targets, num_classes)
            val_loss = softmax_cross_entropy(val_logits, y_one)
            val_acc = accuracy(val_logits, val_targets)
            model.train()
        else:
            val_loss = float("nan")
            val_acc = float("nan")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if log_csv_path is not None:
            with open(log_csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        # callbacks
        if callbacks:
            state = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "model": model,
                "optimizer": optimizer,
            }
            for cb in callbacks:
                cb(state)

        # scheduler step at end of epoch
        if scheduler is not None:
            scheduler.step(epoch)

        dt = time.time() - t0
        lr_str = f" lr={getattr(optimizer, 'lr', None):.3e}" if hasattr(optimizer, "lr") else ""
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{lr_str} ({dt:.1f}s)")

    return history

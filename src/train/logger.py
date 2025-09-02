"""
src/train/logger.py
Simple CSV logger utilities and a callback to log epoch metrics.

Usage:
    from .logger import csv_logger_callback
    cb = csv_logger_callback("reports/results.csv")
    callbacks = [cb, ...]
"""

from __future__ import annotations
import csv
import os
from typing import Dict, Iterable


class CSVLogger:
    """
    Append metrics to a CSV file. Creates the file and headers on first write.
    """

    def __init__(self, path: str, fieldnames: Iterable[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        need_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if need_header:
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def log(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow({k: row.get(k, "") for k in self.fieldnames})


def csv_logger_callback(path: str):
    """
    Callback that logs standard keys from the training state:
    epoch, train_loss, train_acc, val_loss, val_acc

    Example:
        callbacks = [csv_logger_callback("reports/results.csv")]
    """
    logger = CSVLogger(path, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def cb(state: Dict) -> None:
        row = {
            "epoch": state.get("epoch"),
            "train_loss": state.get("train_loss"),
            "train_acc": state.get("train_acc"),
            "val_loss": state.get("val_loss"),
            "val_acc": state.get("val_acc"),
        }
        logger.log(row)

    return cb

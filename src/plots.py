"""Save matplotlib figures: training curves, confusion matrices, per-class metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

CLASS_NAMES = ("real", "fake")


def _figures_dir(root: Path) -> Path:
    d = root / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_training_curves(history: dict[str, list[Any]], artifacts_dir: Path) -> Path:
    """Line plots: training loss, val accuracy, val precision / recall / F1 (macro)."""
    out = _figures_dir(artifacts_dir) / "training_curves.png"
    phases = history.get("phase", [])
    x = np.arange(len(history["train_loss"]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(x, history["train_loss"], color="#2563eb", linewidth=1.8, label="Train loss")
    ax0.set_xlabel("Epoch (phase 1 then phase 2)")
    ax0.set_ylabel("Loss")
    ax0.set_title("Training loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend()

    if phases:
        for i, lbl in enumerate(phases):
            if i > 0 and lbl != phases[i - 1]:
                ax0.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.6)

    ax1 = axes[1]
    ax1.plot(x, history["val_acc"], color="#16a34a", linewidth=1.8, marker="o", markersize=3, label="Val accuracy")
    ax1.plot(x, history["val_precision_macro"], color="#ca8a04", linewidth=1.5, label="Val precision (macro)")
    ax1.plot(x, history["val_recall_macro"], color="#9333ea", linewidth=1.5, label="Val recall (macro)")
    ax1.plot(x, history["val_f1_macro"], color="#dc2626", linewidth=1.5, label="Val F1 (macro)")
    ax1.set_xlabel("Epoch (phase 1 then phase 2)")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Validation metrics")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    if phases:
        for i, lbl in enumerate(phases):
            if i > 0 and lbl != phases[i - 1]:
                ax1.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.6)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    artifacts_dir: Path,
    name: str,
    title: str,
) -> Path:
    out = _figures_dir(artifacts_dir) / f"{name}.png"
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=list(CLASS_NAMES),
        cmap="Blues",
        colorbar=True,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_per_class_metrics_bar(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    artifacts_dir: Path,
    name: str,
    title: str,
) -> Path:
    """Grouped bar chart: precision, recall, F1 per class (test or val)."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    out = _figures_dir(artifacts_dir) / f"{name}.png"
    x = np.arange(len(CLASS_NAMES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w, p, width=w, label="Precision", color="#2563eb")
    ax.bar(x, r, width=w, label="Recall", color="#16a34a")
    ax.bar(x + w, f, width=w, label="F1", color="#ca8a04")
    ax.set_xticks(x)
    ax.set_xticklabels(list(CLASS_NAMES))
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def save_history_json(history: dict[str, list[Any]], artifacts_dir: Path) -> Path:
    path = artifacts_dir / "history.json"
    serializable: dict[str, list[Any]] = {}
    for k, v in history.items():
        row: list[Any] = []
        for x in v:
            if isinstance(x, (np.floating, np.integer)):
                row.append(float(x))
            else:
                row.append(x)
        serializable[k] = row
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path

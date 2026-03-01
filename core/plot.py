import os
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score
)

def plot_training_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    for ax, (tk, vk), title, ylabel in zip(
        axes,
        [("train_loss","val_loss"),("train_acc","val_acc"),("train_f1","val_f1")],
        ["Loss","Accuracy (%)","Macro F1"],
        ["Cross-Entropy Loss","Accuracy (%)","Macro F1"],
    ):
        scale = 100 if "acc" in tk else 1
        ax.plot(epochs, [v*scale for v in history[tk]], label="Train", lw=2)
        ax.plot(epochs, [v*scale for v in history[vk]], label="Val",   lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "training_curves.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Saved: {p}")


def plot_confusion_matrix(labels, preds, words, out_dir):
    cm   = confusion_matrix(labels, preds, labels=list(range(len(words))))
    disp = ConfusionMatrixDisplay(cm, display_labels=words)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Confusion Matrix — Validation Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Saved: {p}")


def plot_per_class_metrics(labels, preds, words, out_dir):
    kw  = dict(average=None, zero_division=0, labels=list(range(len(words))))
    pre = precision_score(labels, preds, **kw)
    rec = recall_score(labels, preds, **kw)
    f1  = f1_score(labels, preds, **kw)
    x, w = np.arange(len(words)), 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, pre, w, label="Precision", color="#4C72B0")
    ax.bar(x,     rec, w, label="Recall",    color="#DD8452")
    ax.bar(x + w, f1,  w, label="F1",        color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(words, rotation=40, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 — Validation Set",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "per_class_metrics.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Saved: {p}")


def plot_split_distribution(full_ds, train_idx, val_idx, out_dir):
    """Bar chart showing how many examples per class ended up in train vs val."""
    n     = full_ds.n_classes
    words = full_ds.words
    tr    = [0] * n;  vl = [0] * n
    for i in train_idx: tr[full_ds.samples[i][1]] += 1
    for i in val_idx:   vl[full_ds.samples[i][1]] += 1

    x, w = np.arange(n), 0.38
    fig, ax = plt.subplots(figsize=(max(10, n * 1.1), 5))
    b_tr = ax.bar(x - w/2, tr, w, label="Train", color="#4C72B0")
    b_vl = ax.bar(x + w/2, vl, w, label="Val",   color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(words, rotation=40, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Stratified Split Distribution per Class",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(list(b_tr) + list(b_vl), tr + vl):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    p = os.path.join(out_dir, "split_distribution.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"[PLOT] Saved: {p}")

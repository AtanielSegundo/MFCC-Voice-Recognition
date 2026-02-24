import os
import sys
import json
import wave
import datetime
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, classification_report
)

sys.path.insert(0, os.path.dirname(__file__))
from MFCC.cepstral import get_mfcc, Signal

# ─── Constants ────────────────────────────────────────────────────────────────

FS          = 16_000
DURATION_S  = 0.75
WINDOW_DT   = 25e-3
HOP_DT      = 10e-3
N_FFT       = 1024
N_FILTERS   = 20
N_MELS      = 16
BANK_MIN_F  = 0.0
BANK_MAX_F  = 4600

TRAIN_SPLIT = 0.70
BATCH_SIZE  = 128
EPOCHS      = 500
LR          = 1e-4
WEIGHT_DECAY= 1e-4
SEED        = 42

# ─── Audio reader ─────────────────────────────────────────────────────────────

def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        n_ch  = wf.getnchannels()
        sw    = wf.getsampwidth()
        sr    = wf.getframerate()
        raw   = wf.readframes(wf.getnframes())
    dtype   = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    samples /= float(2 ** (8 * sw - 1))
    return samples, sr


# ─── MFCC feature extraction ──────────────────────────────────────────────────

def _delta(mfcc: np.ndarray, width: int = 2) -> np.ndarray:
    n_frames = mfcc.shape[0]
    delta    = np.zeros_like(mfcc)
    pad      = np.pad(mfcc, ((width, width), (0, 0)), mode="edge")
    denom    = 2.0 * sum(t ** 2 for t in range(1, width + 1))
    for t in range(n_frames):
        delta[t] = sum(
            w * (pad[t + width + w] - pad[t + width - w])
            for w in range(1, width + 1)
        ) / denom
    return delta


def extract_features(wav_path: str, fs: float, duration: float) -> np.ndarray:
    samples, sr = read_wav(wav_path)
    target_len  = int(fs * duration)
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    elif len(samples) > target_len:
        excess  = len(samples) - target_len
        samples = samples[excess // 2: excess // 2 + target_len]
    sig  = Signal(samples, fs, duration)
    mfcc = get_mfcc(sig, WINDOW_DT, HOP_DT, N_FFT, N_FILTERS, N_MELS,
                    BANK_MIN_F, BANK_MAX_F)
    d1   = _delta(mfcc, width=2)
    d2   = _delta(d1,   width=2)
    return np.stack([mfcc, d1, d2], axis=0).astype(np.float32)


# ─── Stratified split ─────────────────────────────────────────────────────────

def stratified_split(
    samples: list,
    n_classes: int,
    train_frac: float,
    seed: int,
) -> tuple:
    """
    Split dataset indices so every class contributes exactly
    round(train_frac * class_count) samples to train and the rest to val.

    Unlike random_split, which is class-blind and can leave a class almost
    entirely in one split by chance, this guarantees proportional
    representation for every class regardless of dataset size.

    Returns (train_indices, val_indices).
    """
    rng = np.random.default_rng(seed)

    class_indices = {c: [] for c in range(n_classes)}
    for idx, (_, cls) in enumerate(samples):
        class_indices[cls].append(idx)

    train_idx, val_idx = [], []
    for cls in range(n_classes):
        idxs = np.array(class_indices[cls])
        rng.shuffle(idxs)
        n_train = max(1, round(len(idxs) * train_frac))
        n_train = min(n_train, len(idxs) - 1)   # always keep >= 1 for val
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:].tolist())

    return train_idx, val_idx


# ─── Dataset ──────────────────────────────────────────────────────────────────

class NumbersDataset(Dataset):
    def __init__(self, root_dir: str, word_set_path=None, cache: bool = True):
        self.root_dir = root_dir
        self.cache    = cache
        self._cache   = {}

        ws_path = word_set_path or os.path.join(root_dir, "set.json")
        if not os.path.isfile(ws_path):
            raise FileNotFoundError(f"Word-set JSON not found: {ws_path}")
        with open(ws_path, "r", encoding="utf-8") as f:
            self.words = json.load(f)["words"]
        self.n_classes = len(self.words)

        metadata_path = os.path.join(root_dir, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
        self.fs       = metadata.get("sample_rate") or FS
        self.duration = metadata.get("duration")    or DURATION_S

        self.samples = []
        for fname in sorted(os.listdir(root_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            parts = os.path.splitext(fname)[0].split("_")
            if len(parts) < 3:
                continue
            try:
                cls = int(parts[1])
            except ValueError:
                continue
            if cls < 0 or cls >= self.n_classes:
                print(f"[DATASET] Warning: class {cls} out of range in {fname}, skipping.")
                continue
            self.samples.append((os.path.join(root_dir, fname), cls))

        if not self.samples:
            raise RuntimeError(f"No valid .wav files found in {root_dir}")

        print(f"[DATASET] Loaded {len(self.samples)} samples, "
              f"{self.n_classes} classes: {self.words}")
        counts = [0] * self.n_classes
        for _, c in self.samples:
            counts[c] += 1
        for i, (w, n) in enumerate(zip(self.words, counts)):
            print(f"  class {i:2d}  '{w}'  :  {n} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.cache and idx in self._cache:
            features = self._cache[idx]
        else:
            wav_path, _ = self.samples[idx]
            features    = extract_features(wav_path, self.fs, self.duration)
            if self.cache:
                self._cache[idx] = features
        _, cls = self.samples[idx]
        return torch.from_numpy(features), cls

    def class_weights(self):
        counts = torch.zeros(self.n_classes)
        for _, c in self.samples:
            counts[c] += 1
        weights = 1.0 / counts.clamp(min=1)
        return weights / weights.sum() * self.n_classes


# ─── Model ────────────────────────────────────────────────────────────────────

class MFCCClassifier(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True), nn.Dropout(0.50),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.conv(x)))


# ─── Training loop ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels   = labels.to(device)
            logits   = model(features)
            loss     = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


# ─── Plotting helpers ─────────────────────────────────────────────────────────

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


def save_report(history, labels, preds, words, out_dir,
                train_size, val_size, elapsed_s,
                train_idx, val_idx, full_ds):
    path   = os.path.join(out_dir, "training_report.txt")
    report = classification_report(labels, preds, target_names=words, zero_division=0)
    best_e = int(np.argmin(history["val_loss"])) + 1

    tr = [0] * full_ds.n_classes;  vl = [0] * full_ds.n_classes
    for i in train_idx: tr[full_ds.samples[i][1]] += 1
    for i in val_idx:   vl[full_ds.samples[i][1]] += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write(f"  Training Report - {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"  Dataset    : train={train_size}  val={val_size}\n")
        f.write(f"  Epochs     : {EPOCHS}\n")
        f.write(f"  Batch size : {BATCH_SIZE}\n")
        f.write(f"  LR         : {LR}\n")
        f.write(f"  Elapsed    : {elapsed_s:.1f} s\n\n")
        f.write(f"  Best val-loss epoch : {best_e}\n")
        f.write(f"  Final train loss    : {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final val   loss    : {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final val   acc     : {history['val_acc'][-1]*100:.2f} %\n")
        f.write(f"  Final val   macro-F1: {history['val_f1'][-1]:.4f}\n\n")
        f.write("-" * 65 + "\n")
        f.write("Stratified split counts per class\n")
        f.write("-" * 65 + "\n")
        f.write(f"  {'Class':<6}  {'Word':<12}  {'Train':>6}  {'Val':>6}  {'Ratio':>7}\n")
        for i, w in enumerate(words):
            total = tr[i] + vl[i]
            ratio = tr[i] / total if total else 0
            f.write(f"  {i:<6}  {w:<12}  {tr[i]:>6}  {vl[i]:>6}  {ratio:>6.1%}\n")
        f.write("\n")
        f.write("-" * 65 + "\n")
        f.write("Classification Report (validation set)\n")
        f.write("-" * 65 + "\n")
        f.write(report + "\n")
    print(f"[REPORT] Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dataset_path: str, word_set_path=None):
    import time
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Device: {device}")

    ds_name = os.path.basename(os.path.normpath(dataset_path))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(dataset_path)),
                           ds_name + "_training")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[TRAIN] Output folder: {out_dir}")

    print("\n[TRAIN] Building dataset ...")
    full_ds = NumbersDataset(dataset_path, word_set_path=word_set_path, cache=True)

    # ── Stratified split ──────────────────────────────────────────────────────
    # Replaces random_split which is class-blind: by pure chance it can put
    # almost all examples of one class into train and leave the val set with
    # only 1-2 samples of that class, making validation metrics unreliable.
    # stratified_split fixes this by splitting each class independently.
    train_idx, val_idx = stratified_split(
        full_ds.samples, full_ds.n_classes, TRAIN_SPLIT, SEED)

    n_train, n_val = len(train_idx), len(val_idx)
    print(f"\n[SPLIT] Stratified {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}"
          f"  ->  {n_train} train / {n_val} val")

    cls_tr = [0] * full_ds.n_classes
    cls_vl = [0] * full_ds.n_classes
    for i in train_idx: cls_tr[full_ds.samples[i][1]] += 1
    for i in val_idx:   cls_vl[full_ds.samples[i][1]] += 1
    for c, (w, t, v) in enumerate(zip(full_ds.words, cls_tr, cls_vl)):
        print(f"  class {c:2d}  '{w}'  :  train={t}  val={v}  "
              f"({t/(t+v)*100:.0f}% / {v/(t+v)*100:.0f}%)")

    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    model    = MFCCClassifier(n_classes=full_ds.n_classes, n_mels=N_MELS).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[MODEL] Parameters: {n_params:,}")
    print(model)

    weights   = full_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    history = {k: [] for k in
               ("train_loss","val_loss","train_acc","val_acc","train_f1","val_f1")}
    best_val_loss   = float("inf")
    best_model_path = os.path.join(out_dir, "model_best.pt")

    print(f"\n[TRAIN] Starting {EPOCHS} epochs ...\n")
    t0 = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True)
        vl_loss, vl_acc, vl_f1, val_preds, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, device, training=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "val_f1":      vl_f1,
                "words":       full_ds.words,
                "n_mels":      N_MELS,
                "fs":          full_ds.fs,
                "duration":    full_ds.duration,
                "window_dt":   WINDOW_DT,
                "hop_dt":      HOP_DT,
                "n_fft":       N_FFT,
                "n_filters":   N_FILTERS,
                "bank_min_f":  BANK_MIN_F,
                "bank_max_f":  BANK_MAX_F,
            }, best_model_path)
            tag = "  <- best"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}  |  "
                f"loss  train={tr_loss:.4f}  val={vl_loss:.4f}  |  "
                f"acc  train={tr_acc*100:.1f}%  val={vl_acc*100:.1f}%  |  "
                f"F1  val={vl_f1:.3f}  |  lr={lr_now:.2e}{tag}"
            )

    elapsed = time.perf_counter() - t0
    print(f"\n[TRAIN] Finished in {elapsed:.1f} s")

    final_model_path = os.path.join(out_dir, "model_final.pt")
    torch.save({
        "epoch":       EPOCHS,
        "model_state": model.state_dict(),
        "words":       full_ds.words,
        "n_mels":      N_MELS,
        "fs":          full_ds.fs,
        "duration":    full_ds.duration,
        "window_dt":   WINDOW_DT,
        "hop_dt":      HOP_DT,
        "n_fft":       N_FFT,
        "n_filters":   N_FILTERS,
        "bank_min_f":  BANK_MIN_F,
        "bank_max_f":  BANK_MAX_F,
    }, final_model_path)
    print(f"[SAVE] Final model : {final_model_path}")
    print(f"[SAVE] Best model  : {best_model_path}")

    print("\n[PLOT] Generating plots ...")
    plot_training_curves(history, out_dir)
    plot_confusion_matrix(val_labels, val_preds, full_ds.words, out_dir)
    plot_per_class_metrics(val_labels, val_preds, full_ds.words, out_dir)
    plot_split_distribution(full_ds, train_idx, val_idx, out_dir)
    save_report(history, val_labels, val_preds, full_ds.words,
                out_dir, n_train, n_val, elapsed, train_idx, val_idx, full_ds)

    print(f"\n[DONE] All outputs in: {out_dir}/")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Numbers Classifier Trainer")
    parser.add_argument("--dataset",    "-ds",  default="./numbers")
    parser.add_argument("--word_set",   "-ws",  default=None)
    parser.add_argument("--epochs",     "-e",   type=int,   default=EPOCHS)
    parser.add_argument("--batch_size", "-bs",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",                 type=float, default=LR)
    parser.add_argument("--window_dt",  "-wdt", type=float, default=WINDOW_DT)
    parser.add_argument("--hop_dt",     "-hdt", type=float, default=HOP_DT)
    parser.add_argument("--n_fft",              type=int,   default=N_FFT)
    parser.add_argument("--n_filters",          type=int,   default=N_FILTERS)
    parser.add_argument("--n_mels",             type=int,   default=N_MELS)
    parser.add_argument("--bank_min_f", "-bmin",type=float, default=BANK_MIN_F)
    parser.add_argument("--bank_max_f", "-bmax",type=float, default=BANK_MAX_F)
    args = parser.parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    LR         = args.lr
    WINDOW_DT  = args.window_dt
    HOP_DT     = args.hop_dt
    N_FFT      = args.n_fft
    N_FILTERS  = args.n_filters
    N_MELS     = args.n_mels
    BANK_MIN_F = args.bank_min_f
    BANK_MAX_F = args.bank_max_f

    main(dataset_path=args.dataset, word_set_path=args.word_set)
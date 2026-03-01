import os
import sys
import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(__file__))

from core.datasets import stratified_split,HashedSpeechDataset
from core.plot import (plot_confusion_matrix  ,plot_per_class_metrics,
                       plot_split_distribution,plot_training_curves)
from core.models import AVAILABLE_MODELS
from core.cepstral import ConfigMFCC


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

MODEL          = next(iter(AVAILABLE_MODELS.keys()))
AVERAGE_FRAMES = False 
USE_DELTAS     = True

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
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    avg_loss = total_loss / len(loader.dataset)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


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
    mfcc_cfg = ConfigMFCC(WINDOW_DT,HOP_DT,N_FFT,N_FILTERS,N_MELS,BANK_MIN_F,BANK_MAX_F)
    full_ds =HashedSpeechDataset(dataset_path, word_set_path=word_set_path, cache=True,
                                 mfcc_cfg=mfcc_cfg)

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

    sample_features, _ = full_ds[0]
    n_frames = sample_features.shape[1]
    model_cls = AVAILABLE_MODELS.get(MODEL,None)
    if not model_cls:
        print(f"[ERROR] {MODEL} Is Not An Valid Model") 
    model    = AVAILABLE_MODELS[MODEL](n_classes=full_ds.n_classes,n_frames=n_frames,
                                       n_mels=N_MELS,average_frames=AVERAGE_FRAMES,
                                       use_deltas=USE_DELTAS
                                       ).to(device)
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
                "model"     :  MODEL,
                "n_frames"  : n_frames,
                "average_frames": AVERAGE_FRAMES,
                "use_deltas": USE_DELTAS
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
        "model":       MODEL,
        "n_frames":    n_frames,
        "average_frames": AVERAGE_FRAMES,
        "use_deltas": USE_DELTAS
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
    parser = argparse.ArgumentParser("Word Classifier Trainer")
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
    parser.add_argument("--model","-m",type=str,default=MODEL,
                        help=f"Available Models: {', '.join(list(AVAILABLE_MODELS.keys()))}"
                        )
    parser.add_argument("--average_frames", "-avg", default=AVERAGE_FRAMES, action="store_true")
    parser.add_argument("--no_deltas", "-deltas", default=USE_DELTAS, action="store_false")
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
    MODEL      = args.model
    AVERAGE_FRAMES = args.average_frames
    USE_DELTAS = args.no_deltas

    main(dataset_path=args.dataset, word_set_path=args.word_set)
import json
import numpy as np
import os

import torch
from torch.utils.data import Dataset

from core.dsp import extract_features_from_wav
from core.cepstral import ConfigMFCC

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
        n_train = min(n_train, len(idxs) - 1)
        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:].tolist())

    return train_idx, val_idx

class HashedSpeechDataset(Dataset):
    def __init__(self, root_dir: str, 
                 word_set_path=None, 
                 cache: bool = True,
                 sample_rate = 16_000,
                 duration_s  = 0.7,
                 mfcc_cfg    = None
    ):
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

        self.fs       = metadata.get("sample_rate") or sample_rate
        self.duration = metadata.get("duration")    or duration_s
        self.mfcc_cfg = mfcc_cfg or ConfigMFCC()

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
            features    = extract_features_from_wav(wav_path, self.fs, self.duration, self.mfcc_cfg)
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
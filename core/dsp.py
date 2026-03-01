from typing import Callable
import numpy as np
import wave

from core.cepstral import get_mfcc, Signal, ConfigMFCC

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

# Speaker-independent isolated word recognition using dynamic features of speech spectrum
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

def extract_features(samples:np.ndarray, fs: float, duration: float,
                     mfcc_cfg: ConfigMFCC,
                     ) -> np.ndarray:
    sig  = Signal(samples, fs, duration)
    mfcc = get_mfcc(sig, mfcc_cfg)
    d1   = _delta(mfcc, width=2)
    d2   = _delta(d1,   width=2)
    return np.stack([mfcc, d1, d2], axis=0).astype(np.float32)

def extract_features_from_wav(wav_path: str, 
                              fs: float, 
                              duration: float,
                              mfcc_cfg: ConfigMFCC):
    samples, sr = read_wav(wav_path)
    target_len  = int(fs * duration)
    if len(samples) < target_len:
        samples = np.pad(samples, (0, target_len - len(samples)))
    elif len(samples) > target_len:
        excess  = len(samples) - target_len
        samples = samples[excess // 2: excess // 2 + target_len]
    return extract_features(samples,fs,duration,mfcc_cfg)
    
_MAX_GAIN = 10 ** (40.0 / 20.0)   # 40 dB ceiling

def pre_emphasis(samples: np.ndarray, coef: float) -> np.ndarray:
    """y[n] = x[n] - coef * x[n-1]  — first-order high-pass."""
    if coef == 0.0:
        return samples
    out    = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - coef * samples[:-1]
    return out


def rms_normalize(samples: np.ndarray, target: float) -> np.ndarray:
    """Scale samples to *target* RMS. No-op when target <= 0 or signal is silent."""
    if target <= 0.0:
        return samples
    r = float(np.sqrt(np.mean(samples ** 2)))
    if r < 1e-9:
        return samples
    scale = min(target / r, _MAX_GAIN)
    return np.clip(samples * scale, -1.0, 1.0)
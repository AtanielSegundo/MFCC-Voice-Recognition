import numpy as np
from numpy.lib.stride_tricks import as_strided
from dataclasses import dataclass

ms      = lambda t : t * 1e-3
f_to_qf = lambda f : 2595 * np.log10(1 + f/700)
qf_to_f = lambda qf : 700 * (10**(qf/2595) - 1)

#(PICONE 1993, RABINER 1993)
BWcr = lambda f : 25 + 75*(1+ 1.4*(f/1000)**2)**(0.69)

class Signal:
	def __init__(self,S:np.ndarray,Fs,duration_s):
		self.fs = Fs
		self.duration_s = duration_s
		self.samples = S
	def __call__(self):
		return self.samples

@dataclass
class ConfigMFCC:
    window_dt  : float = 25e-3
    hop_dt     : float = 10e-3
    n_fft      : int   = 1024
    n_filters  : int   = 20
    n_mels     : int   = 16
    bank_min_f: float  = 0.0
    bank_max_f: float  = 4600

def dct_type2_ortho(x, axis=1):
    """
    DCT-II (type=2) with 'ortho' normalization implemented with numpy.
    x: array with shape (..., N, ...) where the transform is along `axis`.
    Returns array with same shape.
    """
    x = np.asarray(x, dtype=float)
    if axis != 1:
        x = np.moveaxis(x, axis, 1)
        moved = True
    else:
        moved = False

    if x.ndim != 2:
        front_shape = x.shape[0]
        newshape = (x.shape[0], x.shape[1])
    
    n_frames, N = x.shape

    n = np.arange(N)
    k = np.arange(N)[:, None]  # shape (N,1)
    # kernel: cos(pi * (2n + 1) * k / (2N)), shape (N, N) with rows = k, cols = n
    phi = np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))  # shape (N, N)

    alpha = np.sqrt(2.0 / N) * np.ones(N)
    alpha[0] = np.sqrt(1.0 / N)

    # shape (n_frames, N)
    res = x.dot(phi.T) * alpha  

    if moved:
        res = np.moveaxis(res, 1, axis)
    return res

def get_n_frames(duration_s,window_dt,hop_dt):
    return int(1 + np.floor((duration_s - window_dt) / hop_dt))

def signal_windowing(S:Signal,
                     window_dt: float, hop_dt: float,
					 window_fn = np.hamming
					 ) -> np.ndarray:
    """
    Splits a 1D signal into overlapping frames

    Args:
        X (np.ndarray): Input signal (1D).
        Fs (float): Sampling rate in Hz.
        signal_duration (float): Duration of the signal in seconds.
        window_dt (float): Window length in seconds.
        hop_dt (float): Hop size (frame shift) in seconds.

    Returns:
        np.ndarray: 2D array of shape (n_frames, n_samples_per_frame)
                    containing the windowed signal.
    """
    signal_duration = S.duration_s
    Fs = S.fs
    X = S.samples

    n_frames = get_n_frames(signal_duration,window_dt,hop_dt)
    n_window_samples = int(np.floor(window_dt * Fs))
    n_hop_samples    = int(np.floor(hop_dt * Fs))

    expected_length = (n_frames - 1) * n_hop_samples + n_window_samples
    if len(X) < expected_length:
        raise ValueError("Input signal X is too short for the given parameters.")

    shape = (n_frames, n_window_samples)
    strides = (X.strides[0] * n_hop_samples, X.strides[0])
    windowed_signal = as_strided(X, shape=shape, strides=strides).copy()

    w = window_fn(n_window_samples)
    windowed_signal = windowed_signal * w

    return windowed_signal

def mel_filterbank(Fs, N_fft, N_filters, f_min=0.0, f_max=None):
    if f_max is None:
        f_max = Fs / 2.0

    mel_min, mel_max = f_to_qf(f_min), f_to_qf(f_max)
    mel_points = np.linspace(mel_min, mel_max, N_filters + 2)

    hz_points = qf_to_f(mel_points)
    # (N_fft + 1) is used to ensure correctly Nyquist map 
	# in the last frequency
    bins = np.floor((N_fft + 1) * hz_points / Fs).astype(int)

    n_bins = N_fft // 2 + 1
    M = np.zeros((N_filters, n_bins))

    for m in range(1, N_filters + 1):
        left, center, right = bins[m-1], bins[m], bins[m+1]
        if center == left:
            left_slope = np.zeros(0)
        else:
            denom = center - left
            for k in range(left, center):
                if 0 <= k < n_bins:
                    M[m-1, k] = (k - left) / denom
        if right == center:
            pass
        else:
            denom = right - center
            for k in range(center, right):
                if 0 <= k < n_bins:
                    M[m-1, k] = (right - k) / denom

    return M  # shape: (N_filters, N_fft//2 + 1)

def get_mfcc(signal:Signal, cfg: ConfigMFCC):
    
    assert cfg.n_mels <= cfg.n_filters, "N_mels must be <= N_filters"

    s_m_n = signal_windowing(signal, cfg.window_dt, cfg.hop_dt)

    S_m_k = np.fft.rfft(s_m_n, n=cfg.n_fft, axis=1)[:, : (cfg.n_fft // 2 + 1)]

    power_m_k = np.abs(S_m_k) ** 2

    f_i_k = mel_filterbank(signal.fs, cfg.n_fft, cfg.n_filters, cfg.bank_min_f, cfg.bank_max_f)

    E_m_i = np.dot(power_m_k, f_i_k.T)

    E_m_i_log = np.log(np.maximum(E_m_i, 1e-10))

    mfcc_all = dct_type2_ortho(E_m_i_log, axis=1)

    return mfcc_all[:, :cfg.n_mels]
from dataclasses import dataclass

# PLACEHOLDER
class Signal:
    pass

@dataclass
class ConfigMFCC:
    window_dt  : float = 25e-3
    hop_dt     : float = 10e-3
    n_fft      : int   = 1024
    n_filters  : int   = 20
    n_mels     : int   = 16
    bank_min_f: float  = 0.0
    bank_max_f: float  = 4600

def get_mfcc(s:Signal, cfg: ConfigMFCC):
    pass
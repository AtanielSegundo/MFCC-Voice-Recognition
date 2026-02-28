"""
TestModels.py
─────────────
Real-time Portuguese number word recognition from microphone.

How it works:
  - Ring buffer holds the last ~2 s of audio
  - When RMS energy crosses the VAD threshold → run the model every INFER_EVERY chunks
  - If max(softmax) >= conf_threshold → add prediction to a small vote window
  - Display the majority winner of the vote window (or "—" if no speech)
  - On silence → drain the vote window one slot at a time so the display fades to "—"

No state machine. No settling period. No event pooling.

Controls:
    Q / ESC   — quit
    SPACE     — mute / unmute
    +  /  -   — raise / lower VAD energy threshold   (step 0.001)
    C  /  V   — raise / lower confidence threshold   (step 0.05)
    H         — toggle probability bars
"""

import os, sys, json, argparse, queue, threading, time, collections
import numpy as np
import torch
import torch.nn as nn
import pygame

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    print("[WARN] sounddevice not found — running in DEMO mode.")

sys.path.insert(0, os.path.dirname(__file__))
from MFCC.cepstral import get_mfcc, Signal

# ─── Audio / feature constants (overridden from checkpoint) ──────────────────

FS          = 16_000
DURATION_S  = 0.75
WINDOW_DT   = 25e-3
HOP_DT      = 10e-3
N_FFT       = 1024
N_FILTERS   = 20
N_MELS      = 16
BANK_MIN_F  = 0.0
BANK_MAX_F  = 4600.0

CHUNK_S          = 0.03
VAD_HOLD_S       = 0.35
INFER_EVERY      = 3

# Rolling window of recent confident predictions for majority vote.
# Higher = more stable but slower to update.
VOTE_WINDOW      = 4

CONF_DEFAULT     = 0.55     # min max-prob to count a prediction

WINNER = "-"

# ─── Palette ─────────────────────────────────────────────────────────────────

C_BG        = (10,  12,  18)
C_PANEL     = (18,  22,  34)
C_BORDER    = (38,  48,  72)
C_DIM       = (80,  90, 112)
C_ACCENT    = (72, 128, 255)
C_SPEAKING  = (72, 210, 130)
C_SILENT    = (70,  80, 100)
C_BAR_BG    = (26,  32,  50)
C_BAR_FG    = (72, 128, 255)
C_BAR_BEST  = (255, 210,  60)
C_WORD_ON   = (255, 255, 255)
C_WORD_OFF  = (50,  60,  85)
C_WARN      = (255,  88,  72)

WIN_W, WIN_H = 1280, 720


# ─── Model ───────────────────────────────────────────────────────────────────

class MFCCClassifier(nn.Module):
    def __init__(self, n_classes, n_mels=N_MELS):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.conv(x)))


# ─── Feature extraction ──────────────────────────────────────────────────────

def _delta(m, width=2):
    pad = np.pad(m, ((width, width), (0, 0)), mode="edge")
    d   = 2.0 * sum(t * t for t in range(1, width + 1))
    return np.array([
        sum(w * (pad[t+width+w] - pad[t+width-w]) for w in range(1, width+1)) / d
        for t in range(m.shape[0])
    ])

def extract_features(samples, speech_samples, preemph_coef=0.0, target_rms=0.0):
    # Pad / crop to exact window length
    if len(samples) < speech_samples:
        samples = np.pad(samples, (0, speech_samples - len(samples)))
    else:
        samples = samples[-speech_samples:]

    # Match the MakeDataset processing chain (pre_emphasis → rms_normalize → MFCC)
    # NOTE: Tukey window is intentionally omitted — it was used in dataset creation
    # to smooth cut edges; applying it on a live sliding window would attenuate real
    # speech at the boundaries, hurting inference.
    samples = pre_emphasis(samples, preemph_coef)
    samples = rms_normalize(samples, target_rms)

    sig  = Signal(samples, FS, DURATION_S)
    mfcc = get_mfcc(sig, WINDOW_DT, HOP_DT, N_FFT, N_FILTERS, N_MELS,
                    BANK_MIN_F, BANK_MAX_F)
    d1   = _delta(mfcc)
    d2   = _delta(d1)
    feat = np.stack([mfcc, d1, d2], axis=0).astype(np.float32)
    return torch.from_numpy(feat).unsqueeze(0)

def rms(chunk):
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


def pre_emphasis(samples: np.ndarray, coef: float) -> np.ndarray:
    """y[n] = x[n] - coef * x[n-1]  — first-order high-pass."""
    if coef == 0.0:
        return samples
    out    = np.empty_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - coef * samples[:-1]
    return out

_MAX_GAIN = 10 ** (40.0 / 20.0)   # 40 dB ceiling

def rms_normalize(samples: np.ndarray, target: float) -> np.ndarray:
    """Scale samples to *target* RMS. No-op when target <= 0 or signal is silent."""
    if target <= 0.0:
        return samples
    r = float(np.sqrt(np.mean(samples ** 2)))
    if r < 1e-9:
        return samples
    scale = min(target / r, _MAX_GAIN)
    return np.clip(samples * scale, -1.0, 1.0)


# ─── Demo audio ──────────────────────────────────────────────────────────────

def _demo_thread(q, stop, chunk_samples):
    rng = np.random.default_rng(1)
    while not stop.is_set():
        silence = np.zeros(int(FS * 0.5), dtype=np.float32)
        speech  = rng.normal(0, 0.085, int(FS * 0.8)).astype(np.float32)
        for blk in np.array_split(np.concatenate([silence, speech]),
                                  int(FS * 1.3 / chunk_samples)):
            if stop.is_set(): break
            try: q.put_nowait(blk.astype(np.float32))
            except queue.Full: pass
            time.sleep(CHUNK_S)


# ─── Pygame helpers ───────────────────────────────────────────────────────────

def _rr(surf, rect, r, col):
    pygame.draw.rect(surf, col, rect, border_radius=r)

def _t(surf, text, font, col, x, y, anchor="center"):
    img = font.render(text, True, col)
    rct = img.get_rect()
    setattr(rct, anchor, (x, y))
    surf.blit(img, rct)

def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(model_path: str, word_set_path: str | None,
        conf_threshold: float = CONF_DEFAULT,
        _cli_preemph: float | None = None,
        _cli_target_rms: float | None = None):

    global FS, DURATION_S, WINDOW_DT, HOP_DT, N_FFT, N_FILTERS, BANK_MIN_F, BANK_MAX_F, WINNER

    # ── Load checkpoint ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(model_path, map_location=device)
    words  = ckpt["words"]
    n_mels = ckpt.get("n_mels", N_MELS)

    FS         = ckpt.get("fs",         FS)
    DURATION_S = ckpt.get("duration",   DURATION_S)
    WINDOW_DT  = ckpt.get("window_dt",  WINDOW_DT)
    HOP_DT     = ckpt.get("hop_dt",     HOP_DT)
    N_FFT      = ckpt.get("n_fft",      N_FFT)
    N_FILTERS  = ckpt.get("n_filters",  N_FILTERS)
    BANK_MIN_F = ckpt.get("bank_min_f", BANK_MIN_F)
    BANK_MAX_F = ckpt.get("bank_max_f", BANK_MAX_F)

    chunk_samples   = int(FS * CHUNK_S)
    ring_capacity   = int(FS * 2.0)
    speech_samples  = int(FS * DURATION_S)
    vad_hold_chunks = int(VAD_HOLD_S / CHUNK_S)

    if word_set_path and os.path.isfile(word_set_path):
        with open(word_set_path) as f:
            words = json.load(f)["words"]
    n_cls = len(words)

    model = MFCCClassifier(n_cls, n_mels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Pre-processing params — must match what MakeDataset used ─────────────
    # Priority: CLI arg > checkpoint > metadata.json auto-discovery > default
    _DEFAULT_PREEMPH    = 0.97
    _DEFAULT_TARGET_RMS = 0.0     # 0 = disabled (safe default when unknown)

    def _load_metadata_json(model_path: str) -> dict:
        """Try to find metadata.json in the dataset folder next to the training dir."""
        # model_path is typically <dataset>_training/model_best.pt
        # dataset folder is one level up, named <dataset>
        training_dir = os.path.dirname(os.path.abspath(model_path))
        parent       = os.path.dirname(training_dir)
        # strip trailing "_training" from the folder name
        folder_name  = os.path.basename(training_dir)
        dataset_name = folder_name[:-9] if folder_name.endswith("_training") else folder_name
        candidates   = [
            os.path.join(parent, dataset_name, "metadata.json"),
            os.path.join(parent, "metadata.json"),
            os.path.join(training_dir, "metadata.json"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                with open(p) as f:
                    return json.load(f)
        return {}

    meta = _load_metadata_json(model_path)

    # Read from checkpoint, then metadata.json, then default
    preemph_coef = ckpt.get("preemph",
                   meta.get("preemph", _DEFAULT_PREEMPH))
    target_rms   = ckpt.get("target_rms",
                   meta.get("target_rms", _DEFAULT_TARGET_RMS))

    # CLI can override
    if _cli_preemph is not None:
        preemph_coef = _cli_preemph
    if _cli_target_rms is not None:
        target_rms   = _cli_target_rms

    print(f"[MODEL] {n_cls} classes: {words}")
    print(f"[PREPROC] pre_emphasis coef : {preemph_coef}"
          f"  ({'enabled' if preemph_coef > 0 else 'DISABLED'})")
    print(f"[PREPROC] RMS target        : {target_rms:.4f}"
          f"  ({20*np.log10(target_rms):.1f} dBFS)" if target_rms > 0
          else "[PREPROC] RMS normalisation : DISABLED")

    # ── Shared state ─────────────────────────────────────────────────────────
    lock  = threading.Lock()
    state = {
        "word":     "—",
        "probs":    np.zeros(n_cls, np.float32),
        "speaking": False,
        "energy":   0.0,
        "n_infer":  0,
        "n_skip":   0,
    }

    # Mutable settings (list wrappers so inference thread sees changes)
    vad_thr   = [0.002]
    conf_box  = [conf_threshold]
    muted     = [False]
    show_bars = [True]

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_q  = queue.Queue(maxsize=400)
    stop_evt = threading.Event()
    ring     = collections.deque(maxlen=ring_capacity)

    if _SD_AVAILABLE:
        def _cb(indata, frames, t, status):
            try: audio_q.put_nowait(indata[:, 0].astype(np.float32))
            except queue.Full: pass
        stream = sd.InputStream(samplerate=FS, channels=1, dtype="float32",
                                blocksize=chunk_samples, callback=_cb)
        stream.start()
        print("[MIC] stream started.")
    else:
        threading.Thread(target=_demo_thread,
                         args=(audio_q, stop_evt, chunk_samples), daemon=True).start()
        print("[MIC] demo mode.")

    # ── Inference thread ──────────────────────────────────────────────────────
    def _infer():
        global WINNER
        vad_hold  = 0
        chunk_ctr = 0
        # Rolling window of recent confident class predictions
        vote_buf  = collections.deque(maxlen=VOTE_WINDOW)

        while not stop_evt.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            ring.extend(chunk)
            e = rms(chunk)

            with lock:
                state["energy"] = e

            if muted[0]:
                vad_hold = 0
                vote_buf.clear()
                with lock:
                    state["speaking"] = False
                    state["word"]     = "—"
                continue

            # VAD
            if e >= vad_thr[0]:
                vad_hold = vad_hold_chunks
            elif vad_hold > 0:
                vad_hold -= 1
            speaking = vad_hold > 0

            with lock:
                state["speaking"] = speaking

            if not speaking:
                # Gradually drain vote buffer → display fades to "—"
                if vote_buf:
                    vote_buf.popleft()
                if not vote_buf:
                    with lock:
                        state["word"]  = WINNER
                        state["probs"] = np.zeros(n_cls, np.float32)
                continue

            # Run model every INFER_EVERY chunks
            chunk_ctr += 1
            if chunk_ctr < INFER_EVERY:
                continue
            chunk_ctr = 0

            if len(ring) < speech_samples:
                continue

            try:
                samples = np.array(ring, dtype=np.float64)[-speech_samples:]
                with torch.no_grad():
                    feat   = extract_features(samples, speech_samples,
                                              preemph_coef, target_rms).to(device)
                    logits = model(feat)
                    probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

                conf = float(np.max(probs))
                pred = int(np.argmax(probs))

                with lock:
                    state["probs"]   = probs.copy()
                    state["n_infer"] += 1

                if conf >= conf_box[0]:
                    vote_buf.append(pred)
                else:
                    with lock:
                        state["n_skip"] += 1

                # Majority vote → update display word
                if vote_buf:
                    winner = collections.Counter(vote_buf).most_common(1)[0][0]
                    with lock:
                        state["word"] = words[winner]

            except Exception as ex:
                print(f"[INFER] {ex}")

    threading.Thread(target=_infer, daemon=True).start()

    # ── Pygame ────────────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("TestModels  ·  Word Recognition")
    clock  = pygame.time.Clock()

    def _fnt(name, sz, bold=False):
        try:    return pygame.font.SysFont(name, sz, bold=bold)
        except: return pygame.font.Font(None, sz)

    font_xl   = _fnt("DejaVuSans", 92, bold=True)
    font_md   = _fnt("DejaVuSans", 19)
    font_sm   = _fnt("DejaVuSans", 14)
    font_mono = _fnt("DejaVuSansMono", 13)

    disp_probs    = np.zeros(n_cls, np.float32)
    energy_smooth = 0.0
    prev_word     = "—"
    word_scale    = 1.0
    breathe_t     = 0.0

    running = True
    while running:
        dt = clock.tick(40) / 1000.0

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_SPACE:
                    muted[0] = not muted[0]
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    vad_thr[0] = min(vad_thr[0] + 0.001, 0.10)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    vad_thr[0] = max(vad_thr[0] - 0.001, 0.0001)
                elif k == pygame.K_c:
                    conf_box[0] = min(conf_box[0] + 0.05, 0.99)
                elif k == pygame.K_v:
                    conf_box[0] = max(conf_box[0] - 0.05, 0.10)
                elif k == pygame.K_h:
                    show_bars[0] = not show_bars[0]

        # Snapshot
        with lock:
            cur_word = state["word"]
            probs    = state["probs"].copy()
            speaking = state["speaking"]
            energy   = state["energy"]
            n_infer  = state["n_infer"]
            n_skip   = state["n_skip"]

        energy_smooth = energy_smooth * 0.72 + energy * 0.28
        breathe_t    += dt * 0.85

        # Smooth bars toward latest softmax output
        disp_probs = disp_probs * 0.80 + probs * 0.20

        # Word pop
        if cur_word != prev_word:
            word_scale = 1.40
            prev_word  = cur_word
        word_scale = max(1.0, word_scale - dt * 4.5)

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Header
        pygame.draw.rect(screen, C_PANEL, (0, 0, WIN_W, 48))
        pygame.draw.line(screen, C_BORDER, (0, 48), (WIN_W, 48), 1)
        _t(screen, "TestModels  ·  Real-Time Word Recognition",
           font_sm, C_DIM, WIN_W//2, 13, anchor="midtop")

        if muted[0]:
            s_col, s_lbl = C_WARN, "MUTED"
        elif speaking:
            s_col, s_lbl = C_SPEAKING, "● SPEAKING"
        else:
            s_col, s_lbl = C_SILENT, "○ LISTENING"

        _t(screen, s_lbl, font_sm, s_col, WIN_W-14, 16, anchor="topright")
        _t(screen, f"infer:{n_infer}  skipped:{n_skip}  conf≥{conf_box[0]:.2f}  vote:{VOTE_WINDOW}",
           font_mono, C_DIM, 14, 16, anchor="topleft")

        # Big word
        WORD_CY = 165
        ws = font_xl.render(cur_word.upper(), True, C_WORD_ON)
        if word_scale > 1.001:
            ws = pygame.transform.smoothscale(
                ws, (int(ws.get_width()*word_scale), int(ws.get_height()*word_scale)))
        screen.blit(ws, ws.get_rect(center=(WIN_W//2, WORD_CY)))

        # Ring glow
        RING_R = 82
        CX, CY = WIN_W//2, WORD_CY - 46
        ov = pygame.Surface((WIN_W, 380), pygame.SRCALPHA)
        if speaking and not muted[0]:
            p = 0.5 + 0.5 * np.sin(time.perf_counter() * 5.2)
            pygame.draw.circle(ov, (*C_SPEAKING, int(50 + 95*p)),
                               (CX, CY), int(RING_R + 13*p), 3)
        else:
            b = 0.22 + 0.16 * np.sin(breathe_t)
            pygame.draw.circle(ov, (*C_SILENT, int(40*b)), (CX, CY), RING_R, 2)
        screen.blit(ov, (0, 48))

        # Energy bar
        BX, BY, BW, BH = 52, 278, WIN_W-104, 11
        _rr(screen, (BX, BY, BW, BH), 5, C_BAR_BG)
        fw = int(min(energy_smooth/0.05, 1.0)*BW)
        if fw > 2:
            _rr(screen, (BX, BY, fw, BH), 5, C_SPEAKING if speaking else C_ACCENT)
        tx = BX + int(min(vad_thr[0]/0.05, 1.0)*BW)
        pygame.draw.line(screen, C_WARN, (tx, BY-4), (tx, BY+BH+4), 2)
        _t(screen, "energy", font_sm, C_DIM, BX, BY+BH+5, anchor="topleft")
        _t(screen, f"VAD {vad_thr[0]:.4f}  (+/- adjust)",
           font_sm, C_DIM, BX+BW, BY+BH+5, anchor="topright")

        # Probability bars
        if show_bars[0]:
            BAR_TOP = 310
            BAR_BOT = WIN_H - 72
            MAX_H   = BAR_BOT - BAR_TOP - 38
            cw = (WIN_W-80)/n_cls
            cp = max(4, int(cw*0.13))
            best = int(np.argmax(disp_probs)) if disp_probs.sum() > 0 else -1

            for i, (word, prob) in enumerate(zip(words, disp_probs)):
                cx = 40 + (i+0.5)*cw
                bx = int(40 + i*cw + cp)
                bw = max(1, int(cw - 2*cp))
                bh = max(1, int(prob * MAX_H))
                by = BAR_TOP + MAX_H - bh + 2

                _rr(screen, (bx, BAR_TOP+2, bw, MAX_H), 4, C_BAR_BG)
                is_best = (i == best)
                if is_best: WINNER = word 
                fc = C_BAR_BEST if is_best else _lerp(C_BAR_BG, C_BAR_FG, 0.30)
                if bh > 2:
                    _rr(screen, (bx, by, bw, bh), 4, fc)

                _t(screen, f"{prob*100:.0f}%", font_sm,
                   C_BAR_BEST if is_best else C_DIM,
                   int(cx), by-2, anchor="midbottom")
                _t(screen, word,
                   font_md if n_cls <= 10 else font_sm,
                   C_WORD_ON if is_best else C_WORD_OFF,
                   int(cx), BAR_TOP+MAX_H+7, anchor="midtop")

        # Footer
        pygame.draw.line(screen, C_BORDER, (0, WIN_H-32), (WIN_W, WIN_H-32), 1)
        _t(screen, "SPACE  mute    +/-  VAD    C/V  confidence    H  bars    Q/ESC  quit",
           font_sm, C_DIM, WIN_W//2, WIN_H-18)

        pygame.display.flip()

    stop_evt.set()
    if _SD_AVAILABLE:
        stream.stop(); stream.close()
    pygame.quit()
    print("[EXIT] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      "-m",   default="./numbers_training/model_best.pt")
    ap.add_argument("--word_set",   "-ws",  default=None)
    ap.add_argument("--conf",       "-c",   type=float, default=CONF_DEFAULT,
                    help=f"Min confidence to count a prediction (default {CONF_DEFAULT})")
    ap.add_argument("--preemph",    "-pe",  type=float, default=None,
                    help="Pre-emphasis coefficient (overrides checkpoint/metadata.json). "
                         "0 = disabled. Default: auto-detected.")
    ap.add_argument("--target_rms", "-rms", type=float, default=None,
                    help="RMS normalisation target (overrides checkpoint/metadata.json). "
                         "0 = disabled. Default: auto-detected.")
    args = ap.parse_args()
    if not os.path.isfile(args.model):
        print(f"[ERROR] {args.model} not found"); sys.exit(1)
    run(args.model, args.word_set, args.conf, args.preemph, args.target_rms)
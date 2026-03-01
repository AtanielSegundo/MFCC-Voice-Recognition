"""
TestModels.py
─────────────
Real-time word recognition from microphone or system loopback.

How it works:
  - Ring buffer holds the last ~2 s of audio
  - VAD = pure RMS energy threshold (reliable, same as original)
  - When VAD crosses threshold → run the model every INFER_EVERY chunks
  - Inference runs on the highest-energy window inside the ring buffer
  - Entropy gate rejects predictions where the model is genuinely confused
  - Softmax is temperature-scaled for better confidence calibration
  - If max(softmax) >= conf_threshold → add (pred, conf) to weighted vote window
  - Display the confidence-weighted majority winner (or "—" if no speech)
  - On silence → drain the vote window one slot at a time so the display fades to "—"

Controls:
    Q / ESC   — quit
    SPACE     — mute / unmute
    +  /  -   — raise / lower VAD energy threshold (step 0.001)
    C  /  V   — raise / lower confidence threshold (step 0.05)
    T  /  G   — raise / lower softmax temperature  (step 0.1)
    H         — toggle probability bars
"""

import os, sys, json, argparse, queue, threading, time, collections
import numpy as np
import torch
import pygame

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    print("[WARN] sounddevice not found — running in DEMO mode.")

sys.path.insert(0, os.path.dirname(__file__))

from core.cepstral import ConfigMFCC
from core.dsp import pre_emphasis, rms_normalize, extract_features
from core.models import AVAILABLE_MODELS

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

MFCC_CFG    = None

CHUNK_S          = 0.03
VAD_HOLD_S       = 0.35
INFER_EVERY      = 3

VOTE_WINDOW      = 4

CONF_DEFAULT     = 0.55
TEMPERATURE_DEF  = 1.5     # softmax temperature; >1 spreads probs, 1 = raw
ENTROPY_GATE     = 0.55    # reject if entropy/max_entropy exceeds this fraction

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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))


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


def _rr(surf, rect, r, col):
    pygame.draw.rect(surf, col, rect, border_radius=r)

def _t(surf, text, font, col, x, y, anchor="center"):
    img = font.render(text, True, col)
    rct = img.get_rect()
    setattr(rct, anchor, (x, y))
    surf.blit(img, rct)

def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ─── Device resolution ────────────────────────────────────────────────────────

def _resolve_device(use_loopback: bool, explicit: str | None):
    """
    Returns a sounddevice device index/name, or None for the system default mic.
    Priority: explicit --device > --loopback auto-detect > default mic
    """
    if explicit is not None:
        try:    return int(explicit)
        except: return explicit

    if not use_loopback:
        return None

    import platform
    sys_name = platform.system()

    if sys_name == "Linux":
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if "monitor" in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[LOOPBACK] Using: {d['name']}")
                return i
        print("[WARN] No monitor source found — available devices:")
        print(sd.query_devices())
        print("[TIP]  Run  pactl list sources short  and pass the name with --device")
        return None

    elif sys_name == "Windows":
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if "loopback" in d["name"].lower() and d["max_input_channels"] > 0:
                print(f"[LOOPBACK] Using: {d['name']}")
                return i
        print("[WARN] No WASAPI loopback device found.")
        print("[TIP]  Enable 'Stereo Mix' in Windows Sound settings or use --device.")
        return None

    print(f"[WARN] Loopback auto-detect not implemented for {sys_name}. Use --device.")
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(model_path: str,
        word_set_path: str | None,
        use_loopback: bool = False,
        audio_device: str | None = None,
        conf_threshold: float = CONF_DEFAULT,
        temperature: float = TEMPERATURE_DEF,
        _cli_preemph: float | None = None,
        _cli_target_rms: float | None = None):

    global FS, DURATION_S, WINDOW_DT, HOP_DT, N_FFT, N_FILTERS, \
           BANK_MIN_F, BANK_MAX_F, WINNER, MFCC_CFG

    # ── Load checkpoint ──────────────────────────────────────────────────────
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt         = torch.load(model_path, map_location=torch_device)

    words          = ckpt["words"]
    model_name     = ckpt["model"]
    n_frames       = ckpt["n_frames"]
    average_frames = ckpt["average_frames"]
    use_deltas     = ckpt["use_deltas"]
    n_mels         = ckpt.get("n_mels", N_MELS)

    FS         = ckpt.get("fs",         FS)
    DURATION_S = ckpt.get("duration",   DURATION_S)
    WINDOW_DT  = ckpt.get("window_dt",  WINDOW_DT)
    HOP_DT     = ckpt.get("hop_dt",     HOP_DT)
    N_FFT      = ckpt.get("n_fft",      N_FFT)
    N_FILTERS  = ckpt.get("n_filters",  N_FILTERS)
    BANK_MIN_F = ckpt.get("bank_min_f", BANK_MIN_F)
    BANK_MAX_F = ckpt.get("bank_max_f", BANK_MAX_F)

    if MFCC_CFG is None:
        MFCC_CFG = ConfigMFCC(WINDOW_DT, HOP_DT, N_FFT, N_FILTERS,
                               n_mels, BANK_MIN_F, BANK_MAX_F)

    chunk_samples   = int(FS * CHUNK_S)
    ring_capacity   = int(FS * 2.0)
    speech_samples  = int(FS * DURATION_S)
    vad_hold_chunks = int(VAD_HOLD_S / CHUNK_S)

    if word_set_path and os.path.isfile(word_set_path):
        with open(word_set_path) as f:
            words = json.load(f)["words"]
    n_cls = len(words)

    # ── Build model ──────────────────────────────────────────────────────────
    model = AVAILABLE_MODELS[model_name](
        n_classes=n_cls, n_frames=n_frames,
        n_mels=n_mels, use_deltas=use_deltas,
        average_frames=average_frames
    ).to(torch_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ── Pre-processing params ────────────────────────────────────────────────
    _DEFAULT_PREEMPH    = 0.97
    _DEFAULT_TARGET_RMS = 0.0

    def _load_metadata_json(mp: str) -> dict:
        training_dir = os.path.dirname(os.path.abspath(mp))
        parent       = os.path.dirname(training_dir)
        folder_name  = os.path.basename(training_dir)
        dataset_name = folder_name[:-9] if folder_name.endswith("_training") else folder_name
        for p in [os.path.join(parent, dataset_name, "metadata.json"),
                  os.path.join(parent, "metadata.json"),
                  os.path.join(training_dir, "metadata.json")]:
            if os.path.isfile(p):
                with open(p) as f:
                    return json.load(f)
        return {}

    meta         = _load_metadata_json(model_path)
    preemph_coef = ckpt.get("preemph",    meta.get("preemph",    _DEFAULT_PREEMPH))
    target_rms   = ckpt.get("target_rms", meta.get("target_rms", _DEFAULT_TARGET_RMS))
    if _cli_preemph    is not None: preemph_coef = _cli_preemph
    if _cli_target_rms is not None: target_rms   = _cli_target_rms

    print(f"[MODEL]   {n_cls} classes : {words}")
    print(f"[MODEL]   Architecture   : {model_name}")
    print(f"[PREPROC] pre_emphasis   : {preemph_coef}"
          f"  ({'enabled' if preemph_coef > 0 else 'DISABLED'})")
    if target_rms > 0:
        print(f"[PREPROC] RMS target     : {target_rms:.4f}"
              f"  ({20*np.log10(target_rms):.1f} dBFS)")
    else:
        print("[PREPROC] RMS normalisation : DISABLED")
    print(f"[INFER]   temperature    : {temperature}")
    print(f"[INFER]   entropy gate   : {ENTROPY_GATE:.0%} of max entropy")

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

    vad_thr   = [0.002]
    conf_box  = [conf_threshold]
    temp_box  = [temperature]
    muted     = [False]
    show_bars = [True]

    # ── Audio stream ─────────────────────────────────────────────────────────
    audio_q  = queue.Queue(maxsize=400)
    stop_evt = threading.Event()
    ring     = collections.deque(maxlen=ring_capacity)

    if _SD_AVAILABLE:
        def _cb(indata, frames, t, status):
            try: audio_q.put_nowait(indata[:, 0].astype(np.float32))
            except queue.Full: pass

        resolved_device = _resolve_device(use_loopback, audio_device)
        stream = sd.InputStream(
            samplerate=FS, channels=1, dtype="float32",
            blocksize=chunk_samples, callback=_cb,
            device=resolved_device
        )
        stream.start()
        src_label = "LOOPBACK" if use_loopback else "MIC"
        dev_info  = str(resolved_device) if resolved_device is not None else "default"
        print(f"[{src_label}] device={dev_info}  stream started.")
    else:
        threading.Thread(
            target=_demo_thread,
            args=(audio_q, stop_evt, chunk_samples), daemon=True
        ).start()
        print("[AUDIO] demo mode (sounddevice unavailable).")

    # ── Inference thread ──────────────────────────────────────────────────────
    def _infer():
        global WINNER, MFCC_CFG
        vad_hold  = 0
        chunk_ctr = 0
        vote_buf  = collections.deque(maxlen=VOTE_WINDOW)  # stores (class, conf)

        while not stop_evt.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            ring.extend(chunk)

            # ── VAD: pure RMS energy ─────────────────────────────────────────
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

            if e >= vad_thr[0]:
                vad_hold = vad_hold_chunks
            elif vad_hold > 0:
                vad_hold -= 1
            speaking = vad_hold > 0

            with lock:
                state["speaking"] = speaking

            if not speaking:
                # Gradually drain vote buffer so display fades to "—"
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

            # ── Pick the highest-energy window from the ring buffer ──────────
            # Handles early/late speech onset better than always taking the tail
            arr    = np.array(ring, dtype=np.float64)
            step   = chunk_samples
            starts = range(0, max(1, len(arr) - speech_samples + 1), step)
            best_start = max(
                starts,
                key=lambda s: float(np.mean(arr[s:s + speech_samples] ** 2))
            )
            samples = arr[best_start: best_start + speech_samples]

            with torch.no_grad():
                if len(samples) < speech_samples:
                    samples = np.pad(samples, (0, speech_samples - len(samples)))

                samples = pre_emphasis(samples, preemph_coef)
                samples = rms_normalize(samples, target_rms)
                feat    = extract_features(samples, FS, DURATION_S, MFCC_CFG)
                feat    = torch.from_numpy(feat).unsqueeze(0).to(torch_device)
                logits  = model(feat)

                # ── Temperature scaling ──────────────────────────────────────
                T     = max(temp_box[0], 0.1)
                probs = torch.softmax(logits / T, dim=1).squeeze().cpu().numpy()

            # ── Entropy gate: skip flat/confused distributions ───────────────
            entropy     = -float(np.sum(probs * np.log(probs + 1e-9)))
            max_entropy = np.log(n_cls)
            if entropy / max_entropy > ENTROPY_GATE:
                with lock:
                    state["n_skip"] += 1
                    state["probs"]   = probs.copy()
                continue

            conf = float(np.max(probs))
            pred = int(np.argmax(probs))

            with lock:
                state["probs"]    = probs.copy()
                state["n_infer"] += 1

            if conf >= conf_box[0]:
                vote_buf.append((pred, conf))
            else:
                with lock:
                    state["n_skip"] += 1

            # ── Confidence-weighted majority vote ────────────────────────────
            if vote_buf:
                scores = collections.defaultdict(float)
                for cls_idx, cls_conf in vote_buf:
                    scores[cls_idx] += cls_conf
                winner = max(scores, key=scores.get)
                with lock:
                    state["word"] = words[winner]

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
                elif k == pygame.K_t:
                    temp_box[0] = round(min(temp_box[0] + 0.1, 5.0), 2)
                elif k == pygame.K_g:
                    temp_box[0] = round(max(temp_box[0] - 0.1, 0.2), 2)
                elif k == pygame.K_h:
                    show_bars[0] = not show_bars[0]

        with lock:
            cur_word = state["word"]
            probs    = state["probs"].copy()
            speaking = state["speaking"]
            energy   = state["energy"]
            n_infer  = state["n_infer"]
            n_skip   = state["n_skip"]

        energy_smooth = energy_smooth * 0.72 + energy * 0.28
        breathe_t    += dt * 0.85
        disp_probs    = disp_probs * 0.80 + probs * 0.20

        if cur_word != prev_word:
            word_scale = 1.40
            prev_word  = cur_word
        word_scale = max(1.0, word_scale - dt * 4.5)

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        pygame.draw.rect(screen, C_PANEL, (0, 0, WIN_W, 48))
        pygame.draw.line(screen, C_BORDER, (0, 48), (WIN_W, 48), 1)
        _t(screen, "TestModels  ·  Real-Time Word Recognition",
           font_sm, C_DIM, WIN_W // 2, 13, anchor="midtop")

        src_tag = "LOOP" if use_loopback else "MIC"
        if muted[0]:
            s_col, s_lbl = C_WARN, f"MUTED [{src_tag}]"
        elif speaking:
            s_col, s_lbl = C_SPEAKING, f"● SPEAKING [{src_tag}]"
        else:
            s_col, s_lbl = C_SILENT, f"○ LISTENING [{src_tag}]"

        _t(screen, s_lbl, font_sm, s_col, WIN_W - 14, 16, anchor="topright")
        _t(screen,
           f"infer:{n_infer}  skip:{n_skip}  conf≥{conf_box[0]:.2f}"
           f"  T={temp_box[0]:.1f}  vote:{VOTE_WINDOW}",
           font_mono, C_DIM, 14, 16, anchor="topleft")

        WORD_CY = 165
        ws = font_xl.render(cur_word.upper(), True, C_WORD_ON)
        if word_scale > 1.001:
            ws = pygame.transform.smoothscale(
                ws, (int(ws.get_width() * word_scale),
                     int(ws.get_height() * word_scale)))
        screen.blit(ws, ws.get_rect(center=(WIN_W // 2, WORD_CY)))

        RING_R = 82
        CX, CY = WIN_W // 2, WORD_CY - 46
        ov = pygame.Surface((WIN_W, 380), pygame.SRCALPHA)
        if speaking and not muted[0]:
            p = 0.5 + 0.5 * np.sin(time.perf_counter() * 5.2)
            pygame.draw.circle(ov, (*C_SPEAKING, int(50 + 95 * p)),
                               (CX, CY), int(RING_R + 13 * p), 3)
        else:
            b = 0.22 + 0.16 * np.sin(breathe_t)
            pygame.draw.circle(ov, (*C_SILENT, int(40 * b)), (CX, CY), RING_R, 2)
        screen.blit(ov, (0, 48))

        BX, BY, BW, BH = 52, 278, WIN_W - 104, 11
        _rr(screen, (BX, BY, BW, BH), 5, C_BAR_BG)
        fw = int(min(energy_smooth / 0.05, 1.0) * BW)
        if fw > 2:
            _rr(screen, (BX, BY, fw, BH), 5, C_SPEAKING if speaking else C_ACCENT)
        tx = BX + int(min(vad_thr[0] / 0.05, 1.0) * BW)
        pygame.draw.line(screen, C_WARN, (tx, BY - 4), (tx, BY + BH + 4), 2)
        _t(screen, "energy (RMS)", font_sm, C_DIM, BX, BY + BH + 5, anchor="topleft")
        _t(screen, f"VAD {vad_thr[0]:.4f}  (+/- adjust)",
           font_sm, C_DIM, BX + BW, BY + BH + 5, anchor="topright")

        if show_bars[0]:
            BAR_TOP = 310
            BAR_BOT = WIN_H - 72
            MAX_H   = BAR_BOT - BAR_TOP - 38
            cw = (WIN_W - 80) / n_cls
            cp = max(4, int(cw * 0.13))
            best = int(np.argmax(disp_probs)) if disp_probs.sum() > 0 else -1

            for i, (word, prob) in enumerate(zip(words, disp_probs)):
                cx = 40 + (i + 0.5) * cw
                bx = int(40 + i * cw + cp)
                bw = max(1, int(cw - 2 * cp))
                bh = max(1, int(prob * MAX_H))
                by = BAR_TOP + MAX_H - bh + 2

                _rr(screen, (bx, BAR_TOP + 2, bw, MAX_H), 4, C_BAR_BG)
                is_best = (i == best)
                if is_best:
                    WINNER = word
                fc = C_BAR_BEST if is_best else _lerp(C_BAR_BG, C_BAR_FG, 0.30)
                if bh > 2:
                    _rr(screen, (bx, by, bw, bh), 4, fc)

                _t(screen, f"{prob * 100:.0f}%", font_sm,
                   C_BAR_BEST if is_best else C_DIM,
                   int(cx), by - 2, anchor="midbottom")
                _t(screen, word,
                   font_md if n_cls <= 10 else font_sm,
                   C_WORD_ON if is_best else C_WORD_OFF,
                   int(cx), BAR_TOP + MAX_H + 7, anchor="midtop")

        pygame.draw.line(screen, C_BORDER, (0, WIN_H - 32), (WIN_W, WIN_H - 32), 1)
        _t(screen,
           "SPACE mute  |  +/- VAD  |  C/V conf  |  T/G temp  |  H bars  |  Q/ESC quit",
           font_sm, C_DIM, WIN_W // 2, WIN_H - 18)

        pygame.display.flip()

    stop_evt.set()
    if _SD_AVAILABLE:
        stream.stop()
        stream.close()
    pygame.quit()
    print("[EXIT] done.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Real-time word recognition")
    ap.add_argument("--model",       "-m",    default="./numbers_training/model_best.pt")
    ap.add_argument("--word_set",    "-ws",   default=None)
    ap.add_argument("--conf",        "-c",    type=float, default=CONF_DEFAULT,
                    help=f"Min confidence to accept a prediction (default {CONF_DEFAULT})")
    ap.add_argument("--temperature", "-temp", type=float, default=TEMPERATURE_DEF,
                    help=f"Softmax temperature >1 spreads probs (default {TEMPERATURE_DEF})")
    ap.add_argument("--preemph",     "-pe",   type=float, default=None,
                    help="Pre-emphasis coefficient (overrides checkpoint). 0 = disabled.")
    ap.add_argument("--target_rms",  "-rms",  type=float, default=None,
                    help="RMS normalisation target (overrides checkpoint). 0 = disabled.")
    ap.add_argument("--loopback",             action="store_true",
                    help="Capture system audio (loopback) instead of microphone")
    ap.add_argument("--device",      "-d",    type=str, default=None,
                    help="Explicit sounddevice device name or integer index.")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    run(
        model_path      = args.model,
        word_set_path   = args.word_set,
        use_loopback    = args.loopback,
        audio_device    = args.device,
        conf_threshold  = args.conf,
        temperature     = args.temperature,
        _cli_preemph    = args.preemph,
        _cli_target_rms = args.target_rms,
    )
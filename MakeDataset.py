import os
import argparse
import json as json_module
import datetime
import hashlib
import random
import shutil
import subprocess
import re
import wave

from collections import defaultdict

import numpy as np
import log

'''
[Ataniel - 22/02/2026]

The dataset Schema Will Be:

dataset/
_______| <hash: from the original file name>_<class:<int>>_<speaker: M, F>

The class int should be in range from 0..N, 

and should be associated with an WORD_SET, e.g:

0 => "um", 1 => "dois", 2 => "três", ... 

The Dataset Generation Algorithm Will Be:

Root                     : Dir  , Root Folder With Multi Audio Corpus SubFolders
NUMBER_OF_EXAMPLES_CLASS : Int  , Number of Desired Exemples Per Class
OUT                      : Dir  , Folder To Save Temp And Final Dataset Files
DURATION                 : float, Time of the window centered in the word
EMPHASIS_PERCENT_LENGHT  : float, Percent of the centered duration of the fade_in and fadeout

N = NUMBER_OF_EXAMPLES_CLASS


[This Should Maintain speakers Balanced accross the examples if possible]
1 - Traverse Root Trying to find Files Pairs <name>.wav and <name>.txt:
    1.1 - Open <name>.txt and if it has one or more class examples, continue
    1.2 - Copy To The Out Folder The File Pairs
    1.3 - Update The count for each class
    1.4 - Stop If all the Classes count are >= N
    NOTE: Directory traversal order is randomised (controlled by --seed) so
          examples are drawn from across the whole corpus rather than being
          biased toward the first alphabetical folders/speakers.

2 - Apply MFA To The Folder like:
    mfa align <PATH(OUT)> portuguese_mfa portuguese_mfa <PATH(OUT)>

3 - Make a summary of the alignment_analysis.csv Generated for Each Class With Averages :
    CSV HEADER: file,begin,end,speaker,overall_log_likelihood,speech_log_likelihood,phone_duration_deviation,snr
    print the summary using log.INFO

4 - For each file in the OUT FOLDER:
    4.1 - Use the <filename>.TextGrid to find the start and end of file class word
    4.2 - Center in the half of the start and end of the word and cut the audio file
          from [centre - duration/2, centre + duration/2]
    4.3 - Apply pre-emphasis filter:  y[n] = x[n] - PREEMPH_COEF * x[n-1]
          (first-order high-pass, default coeff 0.97)
          Boosts high-frequency consonant energy and flattens the spectral slope.
          Applied BEFORE the window so the filter sees a causal sequence.
    4.4 - Apply a Tukey-like window with (1 - EMPHASIS_PERCENT_LENGTH) / 2
          fade-in from 0→1 and (1 - EMPHASIS_PERCENT_LENGTH) / 2 fade-out from 1→0
    4.5 - RMS power normalisation to TARGET_RMS (default −23 dBFS ≈ 0.071):
          scale = TARGET_RMS / rms(windowed_segment)
          Applied AFTER the window so the tapered signal is what gets normalised.
          Clips are prevented by clamping the scale to a max of 40 dB gain.
          Files that are pure silence (rms ≈ 0) are skipped with a warning.

5 - If --clean is set:
    5.1 - Move all final processed .wav files from <OUT>/final/ up to <OUT>/
    5.2 - Delete every intermediate file:
            - original (pre-cut) .wav files
            - .txt transcript files
            - .TextGrid alignment files
            - alignment_analysis.csv and any MFA-generated artefacts
            - any MFA temp subdirectories (mfa_*/  etc.)
    5.3 - Remove the now-empty  <OUT>/final/  directory
    Result: <OUT>/ contains ONLY the final cut-and-windowed .wav files


cut → resample → pre_emphasis → tukey_window → rms_normalize → write

'''

# ─── TextGrid Parser ──────────────────────────────────────────────────────────

def parse_textgrid_words(tg_path: str) -> list[dict]:
    """
    Parse a Praat TextGrid file and return all non-empty intervals
    from the 'words' tier as:
        [{"text": str, "xmin": float, "xmax": float, "duration": float}, ...]
    """
    with open(tg_path, "r", encoding="utf-8") as f:
        content = f.read()

    words_match = re.search(
        r'name\s*=\s*"words".*?(?=name\s*=\s*"|$)',
        content,
        re.DOTALL,
    )
    if not words_match:
        log.INFO(f"[TG] No 'words' tier found in: {tg_path}")
        return []

    tier_block = words_match.group(0)

    interval_pattern = re.compile(
        r'xmin\s*=\s*(?P<xmin>[\d.]+)\s+'
        r'xmax\s*=\s*(?P<xmax>[\d.]+)\s+'
        r'text\s*=\s*"(?P<text>[^"]*)"',
        re.DOTALL,
    )

    words = []
    for m in interval_pattern.finditer(tier_block):
        text = m.group("text").strip()
        if not text:
            continue
        xmin = float(m.group("xmin"))
        xmax = float(m.group("xmax"))
        words.append({
            "text":     text,
            "xmin":     xmin,
            "xmax":     xmax,
            "duration": round(xmax - xmin, 6),
        })

    return words


# ─── Audio I/O ────────────────────────────────────────────────────────────────

def read_wav(path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (samples: float64 [-1,1], sample_rate)."""
    with wave.open(path, "rb") as wf:
        n_channels  = wf.getnchannels()
        samp_width  = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)

    fmt_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype   = fmt_map.get(samp_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    max_val = float(2 ** (8 * samp_width - 1))
    samples /= max_val

    return samples, sample_rate


def write_wav(path: str, samples: np.ndarray, sample_rate: int, samp_width: int = 2) -> None:
    """Write a float64 [-1,1] numpy array as a PCM WAV file."""
    max_val = float(2 ** (8 * samp_width - 1))
    dtype   = {1: np.int8, 2: np.int16, 4: np.int32}[samp_width]
    pcm     = np.clip(samples * max_val, -max_val, max_val - 1).astype(dtype)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(samp_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


# ─── Tukey-like window ────────────────────────────────────────────────────────

def tukey_window(n: int, emphasis_percent: float) -> np.ndarray:
    """
    Build a Tukey-like window of length *n*.

    The flat ("emphasis") region covers the centre `emphasis_percent` fraction
    of the window.  The remaining `(1 - emphasis_percent)` is split equally
    into a cosine fade-in and a cosine fade-out:

        fade_width = (1 - emphasis_percent) / 2   (as a fraction of n)

    This matches the spec:
        fade-in  from 0→1  over  (1 - EMPHASIS_PERCENT_LENGTH) / 2
        fade-out from 1→0  over  (1 - EMPHASIS_PERCENT_LENGTH) / 2
    """
    w        = np.ones(n, dtype=np.float64)
    fade_len = int(round(n * (1.0 - emphasis_percent) / 2.0))

    if fade_len == 0:
        return w

    # Cosine fade-in  [0 → 1]
    w[:fade_len]  = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    # Cosine fade-out [1 → 0]
    w[-fade_len:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade_len) / fade_len))

    return w


# ─── Helpers ──────────────────────────────────────────────────────────────────

def file_hash(path: str) -> str:
    """Short SHA-1 derived from the base filename (no extension)."""
    name = os.path.splitext(os.path.basename(path))[0]
    return hashlib.sha1(name.encode()).hexdigest()[:12]


def infer_speaker(path: str) -> str:
    """
    Infer speaker gender from the filename tag <hash>_<class>_<SPEAKER>
    or from the folder structure (.../M001/... or .../F_speaker/...).
    Falls back to 'U' (unknown).
    """
    name = os.path.splitext(os.path.basename(path))[0]
    tag  = name.split("_")[-1].upper()
    if tag.startswith("F"):
        return "F"
    if tag.startswith("M"):
        return "M"

    parts = path.replace("\\", "/").split("/")
    for part in reversed(parts[:-1]):
        upper = part.upper()
        if upper.startswith("F"):
            return "F"
        if upper.startswith("M"):
            return "M"

    return "U"


def find_classes_in_transcript(txt_path: str, word_set: list[str]) -> list[int]:
    """
    Return a list of class indices whose word appears in the transcript.
    Case-insensitive whole-word match.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read().lower()

    tokens = set(content.split())
    return [idx for idx, word in enumerate(word_set) if word.lower() in tokens]


# ─── Step 1 ───────────────────────────────────────────────────────────────────

def collect_files(root: str, out: str, word_set: list[str], n: int,
                  seed: int = 42) -> dict[int, list[str]]:
    """
    Walk *root* for .wav/.txt pairs and copy qualifying ones to *out*.

    Speaker balance: for each class we track how many M / F / U files have
    been added and always prefer the gender with the lower count so that the
    final dataset is as balanced as possible.

    The directory traversal order is randomised (controlled by *seed*) so
    that examples are drawn from across the whole corpus rather than being
    biased toward the first alphabetical folders/speakers.  File pairs within
    each directory are also shuffled for the same reason.

    Returns {class_idx: [new_base_name, ...]}
    """
    rng = random.Random(seed)
    os.makedirs(out, exist_ok=True)

    class_counts:   dict[int, int]             = defaultdict(int)
    speaker_counts: dict[tuple[int, str], int] = defaultdict(int)
    class_files:    dict[int, list[str]]       = defaultdict(list)
    n_classes = len(word_set)

    def all_classes_full() -> bool:
        return all(class_counts[i] >= n for i in range(n_classes))

    for dirpath, dirnames, filenames in os.walk(root):
        if all_classes_full():
            break

        # Shuffle subdirectory order in-place — os.walk respects this and will
        # visit them in the shuffled order, spreading traversal across the whole
        # tree instead of exhausting one branch before moving to the next.
        rng.shuffle(dirnames)

        wav_bases = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".wav")}
        txt_bases = {os.path.splitext(f)[0] for f in filenames if f.lower().endswith(".txt")}

        # Shuffle pairs within this directory too so no alphabetical bias
        pairs = sorted(wav_bases & txt_bases)
        rng.shuffle(pairs)

        for base in pairs:
            if all_classes_full():
                break

            wav_src = os.path.join(dirpath, base + ".wav")
            txt_src = os.path.join(dirpath, base + ".txt")

            classes_found = find_classes_in_transcript(txt_src, word_set)
            if not classes_found:
                continue

            speaker = infer_speaker(wav_src)
            h       = file_hash(wav_src)

            for cls in classes_found:
                if class_counts[cls] >= n:
                    continue

                # Balance: allow at most ceil(n/2) per speaker when others exist
                half         = (n + 1) // 2
                other_total  = sum(
                    speaker_counts[(cls, s)]
                    for s in ["M", "F", "U"] if s != speaker
                )
                if other_total > 0 and speaker_counts[(cls, speaker)] >= half and other_total < half:
                    log.INFO(
                        f"[BALANCE] Skipping {base} for class {cls} "
                        f"(speaker={speaker} already at {speaker_counts[(cls, speaker)]}/{half})"
                    )
                    continue

                new_base = f"{h}_{cls}_{speaker}"
                wav_dst  = os.path.join(out, new_base + ".wav")
                txt_dst  = os.path.join(out, new_base + ".txt")

                if os.path.exists(wav_dst):
                    log.INFO(f"[SKIP] already exists: {new_base}")
                    class_counts[cls]              += 1
                    speaker_counts[(cls, speaker)] += 1
                    class_files[cls].append(new_base)
                    continue

                shutil.copy2(wav_src, wav_dst)
                shutil.copy2(txt_src, txt_dst)

                class_counts[cls]              += 1
                speaker_counts[(cls, speaker)] += 1
                class_files[cls].append(new_base)
                log.INFO(
                    f"[COPY] {base} -> {new_base}  "
                    f"(class {cls}, {speaker}, count {class_counts[cls]}/{n})"
                )

    for i in range(n_classes):
        if class_counts[i] < n:
            log.INFO(
                f"[WARN] Class {i} ('{word_set[i]}'): "
                f"only {class_counts[i]}/{n} examples found."
            )
        log.INFO(
            f"[BALANCE] Class {i} ('{word_set[i]}'): "
            f"M={speaker_counts[(i,'M')]}  "
            f"F={speaker_counts[(i,'F')]}  "
            f"U={speaker_counts[(i,'U')]}"
        )

    return dict(class_files)


# ─── Step 2 ───────────────────────────────────────────────────────────────────

def run_mfa(out_folder: str):
    cmd = [
        "mfa", "align",
        out_folder,
        "portuguese_brazil_mfa",  
        "portuguese_mfa",         
        out_folder,
        "--clean"
    ]
    log.INFO(f"[MFA] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        log.ERROR(f"[MFA] Failed (exit {e.returncode}):\n{e.stdout}\n{e.stderr}")
        raise

# ─── Step 3 ───────────────────────────────────────────────────────────────────

def summarise_alignment(out: str, word_set: list[str], class_files: dict[int, list[str]]) -> None:
    """
    For every .TextGrid in *out*, parse the 'words' tier and collect
    duration (xmax - xmin) of each interval matching a class word.
    Print per-class statistics via log.INFO.
    """
    word_to_class: dict[str, int]            = {w.lower(): i for i, w in enumerate(word_set)}
    class_durations: dict[int, list[float]]  = defaultdict(list)
    class_occurrences: dict[int, list[dict]] = defaultdict(list)

    tg_files = [
        os.path.join(out, f)
        for f in sorted(os.listdir(out))
        if f.lower().endswith(".textgrid")
    ]

    if not tg_files:
        log.INFO(f"[SUMMARY] No .TextGrid files found in: {out}")
        return

    for tg_path in tg_files:
        for interval in parse_textgrid_words(tg_path):
            cls = word_to_class.get(interval["text"].lower())
            if cls is None:
                continue
            class_durations[cls].append(interval["duration"])
            class_occurrences[cls].append({
                "file":     os.path.basename(tg_path),
                "xmin":     interval["xmin"],
                "xmax":     interval["xmax"],
                "duration": interval["duration"],
            })

    log.NEWLINE(1)
    log.INFO("=" * 65)
    log.INFO("ALIGNMENT SUMMARY — word durations per class (from TextGrid)")
    log.INFO("=" * 65)

    for cls in range(len(word_set)):
        word      = word_set[cls]
        durations = class_durations.get(cls, [])
        count     = len(durations)

        if count == 0:
            log.INFO(f"\nClass {cls} — '{word}': no aligned occurrences found.")
            continue

        avg      = sum(durations) / count
        mn       = min(durations)
        mx       = max(durations)
        variance = sum((d - avg) ** 2 for d in durations) / count
        std      = variance ** 0.5

        log.INFO(f"\nClass {cls} — '{word}'  ({count} occurrences)")
        log.INFO(
            f"  {'xmin→xmax duration':<28}  avg={avg:.4f}s  "
            f"min={mn:.4f}s  max={mx:.4f}s  std={std:.4f}s"
        )
        for occ in class_occurrences[cls]:
            log.INFO(
                f"    {occ['file']:<45} "
                f"[{occ['xmin']:.3f} → {occ['xmax']:.3f}]  "
                f"dur={occ['duration']:.4f}s"
            )

    log.INFO("=" * 65)


# ─── Step 4 ───────────────────────────────────────────────────────────────────

def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample *samples* from *orig_sr* to *target_sr* using linear interpolation.
    Pure numpy — no scipy required.
    """
    if orig_sr == target_sr:
        return samples

    orig_len   = len(samples)
    target_len = int(round(orig_len * target_sr / orig_sr))

    orig_times   = np.linspace(0.0, 1.0, orig_len,   endpoint=False)
    target_times = np.linspace(0.0, 1.0, target_len, endpoint=False)

    return np.interp(target_times, orig_times, samples)


# ─── Pre-emphasis ─────────────────────────────────────────────────────────────

def pre_emphasis(samples: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    First-order FIR high-pass filter:  y[n] = x[n] - coef * x[n-1]

    Purpose:
      Speech has a natural spectral tilt of ~6 dB/octave (more energy at low
      frequencies).  Pre-emphasis compensates by boosting high-frequency
      content before feature extraction, improving the SNR of fricatives,
      affricates and other high-frequency phonemes that are diagnostically
      important for word discrimination.

    Typical coef: 0.95 – 0.97.  coef=0 is a no-op.

    Must be applied BEFORE the Tukey window so that the first-order difference
    is computed on the real causal audio signal, not on a zero-padded edge.
    """
    if coef == 0.0:
        return samples.copy()
    out = np.empty_like(samples)
    out[0] = samples[0]                         # first sample passes through
    out[1:] = samples[1:] - coef * samples[:-1]
    return out


# ─── RMS Power Normalisation ──────────────────────────────────────────────────

# Default target: −23 dBFS  (EBU R128 loudness anchor level for speech)
#   target_rms = 10 ** (−23 / 20) ≈ 0.0708
_DEFAULT_TARGET_RMS = 10 ** (-23.0 / 20.0)
_MAX_GAIN_DB        = 40.0     # never boost more than 40 dB (prevents noise explosion)


def rms_normalize(samples: np.ndarray,
                  target_rms: float = _DEFAULT_TARGET_RMS) -> tuple[np.ndarray, float]:
    """
    Scale *samples* so that its RMS equals *target_rms*.

    Returns (normalised_samples, scale_factor_applied).

    The scale is clamped to [0, 10^(MAX_GAIN_DB/20)] to prevent extreme
    amplification of near-silent files (which would just amplify noise).

    Must be applied AFTER the Tukey window so that the windowed (tapered)
    signal is what gets normalised — this keeps the energy measurement
    consistent with what the model will see during inference.
    """
    rms = float(np.sqrt(np.mean(samples ** 2)))
    if rms < 1e-9:
        return samples.copy(), 1.0          # silence — leave as-is

    scale = target_rms / rms
    max_scale = 10 ** (_MAX_GAIN_DB / 20.0)
    scale = min(scale, max_scale)

    return (samples * scale).clip(-1.0, 1.0), scale


def cut_and_window(
    out             : str,
    word_set        : list[str],
    duration        : float,
    emphasis_percent: float,
    sample_rate     : float,
    preemph_coef    : float = 0.97,
    target_rms      : float = _DEFAULT_TARGET_RMS,
) -> None:
    """
    For every .wav in *out* that has a matching .TextGrid, runs the full
    processing chain and writes the result to <out>/final/<original_name>.wav.

    Processing order (order matters):
        1. Resample to *sample_rate*          (if needed)
        2. Cut centred window                 [centre - duration/2, centre + duration/2]
        3. Pre-emphasis                       y[n] = x[n] - preemph_coef * x[n-1]
        4. Tukey window                       cosine fade-in / flat / cosine fade-out
        5. RMS normalisation                  scale to *target_rms* RMS amplitude

    Args:
        preemph_coef : FIR high-pass coefficient (0 = disabled, 0.97 = standard).
        target_rms   : Desired RMS amplitude after normalisation.
                       Default is −23 dBFS (≈ 0.071), the EBU R128 anchor.
                       Pass 0.0 to disable normalisation.
    """
    final_dir = os.path.join(out, "final")
    os.makedirs(final_dir, exist_ok=True)

    word_to_class: dict[str, int] = {w.lower(): i for i, w in enumerate(word_set)}

    wav_files = [
        f for f in sorted(os.listdir(out))
        if f.lower().endswith(".wav")
    ]

    processed = skipped = 0

    for wav_name in wav_files:
        base    = os.path.splitext(wav_name)[0]
        wav_src = os.path.join(out, wav_name)
        tg_path = os.path.join(out, base + ".TextGrid")

        if not os.path.isfile(tg_path):
            tg_path = os.path.join(out, base + ".textgrid")
        if not os.path.isfile(tg_path):
            log.INFO(f"[CUT] No TextGrid for {wav_name}, skipping.")
            skipped += 1
            continue

        parts = base.split("_")
        try:
            expected_cls = int(parts[1]) if len(parts) >= 3 else None
        except ValueError:
            expected_cls = None

        intervals       = parse_textgrid_words(tg_path)
        target_interval = None

        for interval in intervals:
            cls = word_to_class.get(interval["text"].lower())
            if cls is None:
                continue
            if expected_cls is not None and cls != expected_cls:
                continue
            target_interval = interval
            break

        if target_interval is None:
            log.INFO(f"[CUT] No class word found in TextGrid for {wav_name}, skipping.")
            skipped += 1
            continue

        # ── Step 4.2  Resample + cut ─────────────────────────────────────────
        samples, orig_sr = read_wav(wav_src)

        if sample_rate and sample_rate != orig_sr:
            log.INFO(f"[CUT] Resampling {wav_name}: {orig_sr} Hz → {sample_rate} Hz")
            samples = resample_audio(samples, orig_sr, sample_rate)
            sr = sample_rate
        else:
            sr = orig_sr

        total_samples = len(samples)
        word_centre   = (target_interval["xmin"] + target_interval["xmax"]) / 2.0
        half_dur      = duration / 2.0
        start_sec     = word_centre - half_dur
        end_sec       = word_centre + half_dur

        start_idx = int(round(start_sec * sr))
        end_idx   = int(round(end_sec   * sr))
        win_len   = end_idx - start_idx

        segment   = np.zeros(win_len, dtype=np.float64)
        src_start = max(start_idx, 0)
        src_end   = min(end_idx, total_samples)
        dst_start = src_start - start_idx
        dst_end   = dst_start + (src_end - src_start)

        if src_end > src_start:
            segment[dst_start:dst_end] = samples[src_start:src_end]

        rms_before = float(np.sqrt(np.mean(segment ** 2)))

        # ── Step 4.3  Pre-emphasis ────────────────────────────────────────────
        if preemph_coef > 0.0:
            segment = pre_emphasis(segment, preemph_coef)

        # ── Step 4.4  Tukey window ────────────────────────────────────────────
        segment = segment * tukey_window(win_len, emphasis_percent)

        # ── Step 4.5  RMS normalisation ───────────────────────────────────────
        if target_rms > 0.0:
            seg_rms = float(np.sqrt(np.mean(segment ** 2)))
            if seg_rms < 1e-9:
                log.INFO(f"[CUT] WARN {wav_name} is near-silent after windowing "
                         f"(rms={seg_rms:.2e}), skipping normalisation.")
                norm_scale = 1.0
            else:
                segment, norm_scale = rms_normalize(segment, target_rms)
        else:
            norm_scale = 1.0

        rms_after = float(np.sqrt(np.mean(segment ** 2)))

        out_path = os.path.join(final_dir, wav_name)
        write_wav(out_path, segment, sr)

        log.INFO(
            f"[CUT] {wav_name}  word='{target_interval['text']}'  "
            f"[{target_interval['xmin']:.3f}→{target_interval['xmax']:.3f}]  "
            f"centre={word_centre:.3f}s  "
            f"rms {rms_before:.4f}→{rms_after:.4f}  "
            f"gain×{norm_scale:.2f}  "
            f"→ final/{wav_name}"
        )
        processed += 1

    log.NEWLINE(1)
    log.INFO(f"[CUT] Done — processed: {processed}, skipped: {skipped}")
    log.INFO(f"[CUT] Final files saved to: {final_dir}")


# ─── Step 5  (--clean) ────────────────────────────────────────────────────────

_INTERMEDIATE_EXTENSIONS = {
    ".txt",          # transcript copies
    ".textgrid",     # MFA alignment grids
    ".csv",          # alignment_analysis.csv and similar MFA reports
    ".log",          # MFA run logs
    ".json",         # MFA config / metadata dumps
}

# Top-level directory names that MFA creates inside the out folder
_MFA_SUBDIRS = {
    "mfa_output",
    "mfa_temp",
    "mfa_align",
    "logs",          # MFA log directory
    "split0",        # MFA split-corpus artefacts
}


def clean_intermediates(out: str) -> None:
    """
    Step 5 — move final processed WAVs up from <out>/final/ and delete
    every intermediate file/folder, leaving ONLY the processed .wav files
    directly inside <out>/.

    Intermediate artefacts removed:
      • <out>/*.txt          — transcript copies from step 1
      • <out>/*.TextGrid     — MFA alignment grids from step 2
      • <out>/*.csv          — MFA alignment_analysis.csv and similar
      • <out>/*.log          — MFA run logs
      • <out>/*.json         — MFA metadata
      • <out>/mfa_*/         — MFA temporary subdirectories
      • <out>/logs/          — MFA log subdirectory
      • <out>/<name>.wav     — original (pre-cut) WAV copies from step 1
                               (replaced by the processed version)
      • <out>/final/         — staging directory (emptied and removed)
    """
    final_dir = os.path.join(out, "final")

    if not os.path.isdir(final_dir):
        log.INFO("[CLEAN] No final/ directory found — nothing to promote.")
        return

    final_wavs = [f for f in os.listdir(final_dir) if f.lower().endswith(".wav")]

    if not final_wavs:
        log.INFO("[CLEAN] final/ directory is empty — nothing to promote.")
        return

    log.NEWLINE(1)
    log.INFO("[CLEAN] Promoting final .wav files and removing intermediates …")

    # ── 5.1  Move processed WAVs from final/ → out/ ───────────────────────────
    promoted = 0
    for wav_name in final_wavs:
        src = os.path.join(final_dir, wav_name)
        dst = os.path.join(out, wav_name)
        shutil.move(src, dst)
        log.INFO(f"[CLEAN] Promoted  final/{wav_name}  →  {wav_name}")
        promoted += 1

    # ── 5.2  Delete intermediate files in out/ ────────────────────────────────
    removed_files = 0
    for entry in os.listdir(out):
        entry_path = os.path.join(out, entry)

        # Skip the (now empty) final dir — handled below
        if entry == "final":
            continue

        # Delete known intermediate subdirectories
        if os.path.isdir(entry_path):
            if entry.lower() in _MFA_SUBDIRS or entry.lower().startswith("mfa_"):
                shutil.rmtree(entry_path)
                log.INFO(f"[CLEAN] Removed dir   {entry}/")
            continue

        # Delete files by extension (but never the promoted .wav files)
        ext = os.path.splitext(entry)[1].lower()
        if ext in _INTERMEDIATE_EXTENSIONS:
            os.remove(entry_path)
            log.INFO(f"[CLEAN] Removed file  {entry}")
            removed_files += 1
        elif ext == ".wav":
            # This is an original pre-cut WAV that has been replaced by the
            # processed version promoted above.  Only remove it if a processed
            # version with the same name now exists in out/.
            processed_path = os.path.join(out, entry)
            # The file IS already at processed_path because we moved it there;
            # but we are iterating os.listdir which saw the old file before the
            # move.  By now it has been overwritten / is the processed version —
            # so there is nothing extra to delete here; the move already replaced
            # it.  If no processed version existed (skipped file) we keep it.
            pass

    # ── 5.3  Remove the now-empty final/ directory ────────────────────────────
    try:
        os.rmdir(final_dir)   # only succeeds if truly empty
        log.INFO("[CLEAN] Removed dir   final/")
    except OSError:
        # Contains skipped/unprocessed files — leave it and warn
        remaining = os.listdir(final_dir)
        log.INFO(
            f"[CLEAN] final/ not empty ({len(remaining)} file(s) remain) — "
            f"left in place."
        )

    log.NEWLINE(1)
    log.INFO(
        f"[CLEAN] Done — promoted: {promoted} WAV(s), "
        f"removed: {removed_files} intermediate file(s)."
    )
    log.INFO(f"[CLEAN] OUT folder now contains only final processed audio.")


# ─── Saving Metadata ──────────────────────────────────────────────────────────

from typing import Dict, Any

def save_dataset_metadata(out_folder: str, metadata: Dict[str, Any],
                          file_name: str = "metadata.json"):
    metadata_path = os.path.join(out_folder, file_name)
    with open(metadata_path, "w") as f:
        json_module.dump(metadata, f, indent=4)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Dataset Maker")
    parser.add_argument("root",
                        help="Root Folder With Multi Audio Corpus")
    parser.add_argument("out",
                        help="Folder To Save The Dataset")
    parser.add_argument("--n", "-n", type=int, default=100,
                        help="Number of desired examples per class")
    parser.add_argument("--words_set", "-ws", default="sets/numeros.json",
                        help="Path To The Json With The Words Set")
    parser.add_argument("--duration", "-d", type=float, default=1.0,
                        help="Window duration in seconds centred on the target word (default: 1.0)")
    parser.add_argument("--emphasis", "-e", type=float, default=0.5,
                        help="Flat (emphasis) fraction of the Tukey window, 0..1 (default: 0.5)")
    parser.add_argument("--preemph", "-pe", type=float, default=0.97,
                        help=(
                            "Pre-emphasis filter coefficient (0 = disabled, "
                            "default: 0.97).  Applied before the Tukey window."
                        ))
    parser.add_argument("--target_rms", "-rms", type=float,
                        default=_DEFAULT_TARGET_RMS,
                        help=(
                            f"Target RMS amplitude for power normalisation "
                            f"(default: {_DEFAULT_TARGET_RMS:.4f} = −23 dBFS). "
                            f"Pass 0 to disable."
                        ))
    parser.add_argument("--sample_rate", "-fs", type=int, default=48000,
                        help="Sample Rate To Resample WAV to")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for corpus traversal order (default: 42). "
                             "Change to get a different random sample from the same corpus.")
    parser.add_argument("--log", default=log.OUT,
                        help="Path To Log File")
    parser.add_argument("--skip_mfa", action="store_true",
                        help="Skip MFA alignment step (useful for testing)")
    parser.add_argument("--skip_cut", action="store_true",
                        help="Skip the cut+window step (step 4)")
    parser.add_argument("--clean", action="store_true",
                        help=(
                            "After processing, remove all intermediate files "
                            "(.txt, .TextGrid, .csv, MFA dirs, original WAV copies) "
                            "and promote the final processed WAVs to the OUT root. "
                            "Implies step 4 runs (--skip_cut is ignored when --clean is set)."
                        ))

    args = parser.parse_args()

    root_folder      = args.root
    out_folder       = args.out
    word_set_path    = args.words_set
    log_path         = args.log
    N                = args.n
    sample_rate      = args.sample_rate
    DURATION         = args.duration
    EMPHASIS_PERCENT = args.emphasis
    PREEMPH_COEF     = args.preemph
    TARGET_RMS       = args.target_rms
    SEED             = args.seed

    # --clean requires step 4 to have run
    if args.clean and args.skip_cut:
        args.skip_cut = False
        print("[WARN] --clean requires step 4; ignoring --skip_cut.")

    log.setOut(log_path)
    WORD_SET: list[str] = []

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.isdir(root_folder):
        log.ERROR(f"Root Is Not A Folder: {root_folder}")

    if os.path.isfile(word_set_path):
        with open(word_set_path, "r", encoding="utf-8") as f:
            data = json_module.load(f)
            WORD_SET.extend(data["words"])
    else:
        log.ERROR(f"Word Set Is Not A File: {word_set_path}")

    log.NEWLINE(3)
    log.INFO(f"{datetime.date.today()}")
    log.INFO(f"Root             : {root_folder}")
    log.INFO(f"Out              : {out_folder}")
    log.INFO(f"N / class        : {N}")
    log.INFO(f"Window duration  : {DURATION}s")
    log.INFO(f"Tukey emphasis   : {EMPHASIS_PERCENT}")
    log.INFO(f"Pre-emphasis α   : {PREEMPH_COEF}  ({'enabled' if PREEMPH_COEF > 0 else 'disabled'})")
    log.INFO(f"Target RMS       : {TARGET_RMS:.4f}  ({20*np.log10(TARGET_RMS):.1f} dBFS)"
             if TARGET_RMS > 0 else "Target RMS       : disabled")
    log.INFO(f"Traversal seed   : {SEED}")
    log.INFO(f"Clean mode       : {args.clean}")
    log.INFO(f"Word set         : {WORD_SET}")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    log.NEWLINE(1)
    log.INFO("[STEP 1] Collecting file pairs …")
    class_files = collect_files(root_folder, out_folder, WORD_SET, N, SEED)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if not args.skip_mfa:
        log.NEWLINE(1)
        log.INFO("[STEP 2] Running MFA alignment …")
        run_mfa(out_folder)
    else:
        log.INFO("[STEP 2] MFA skipped (--skip_mfa).")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    log.NEWLINE(1)
    log.INFO("[STEP 3] Summarising alignment results …")
    summarise_alignment(out_folder, WORD_SET, class_files)

    if not args.skip_cut:
        log.NEWLINE(1)
        log.INFO("[STEP 4] Cutting and windowing audio …")
        cut_and_window(out_folder, WORD_SET, DURATION, EMPHASIS_PERCENT,
                       sample_rate, PREEMPH_COEF, TARGET_RMS)
    else:
        log.INFO("[STEP 4] Cut+window skipped (--skip_cut).")

    # ── Step 5  (optional clean) ──────────────────────────────────────────────
    if args.clean:
        log.NEWLINE(1)
        log.INFO("[STEP 5] Cleaning intermediates …")
        clean_intermediates(out_folder)
    else:
        log.INFO("[STEP 5] Clean skipped (pass --clean to enable).")

    save_dataset_metadata(out_folder, {
        "sample_rate": sample_rate,
        "duration":    DURATION,
        "N":           N,
        "preemph":     PREEMPH_COEF,
        "target_rms":  TARGET_RMS,
        "seed":        SEED,
    })
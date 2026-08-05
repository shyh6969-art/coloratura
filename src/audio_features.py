"""
Coloratura — audio feature extraction (music -> painting direction).

The signal-processing counterpart to feature_extraction.py, which pulls CV
features out of a painting. Deliberately hand-rolled with numpy/scipy
rather than pulling in librosa — this project already builds its own core
machinery rather than depending on someone else's black box (own harmony
engine instead of a sample library, own synthesizer instead of a
SoundFont), and the same instinct applies here: tempo/key/spectral
features from first principles, using only what's already a dependency.

soundfile (via libsndfile) decodes MP3 directly — no ffmpeg needed, and no
new heavy dependency beyond one small C-extension package.

Known limitation, found by testing against this project's own Stage A
renders rather than assumed away: estimate_tempo() has an "octave error"
mitigation (see its docstring), but it doesn't catch everything — a
known-108bpm Stage A track came back at 53.8. Traced the cause: our own
synth.py uses soft, sustained attacks for most instrument families and no
percussion, so the audio genuinely lacks strong onset transients at the
notional tempo grid (the secondary autocorrelation peak at double-tempo
was only 4.5% of the chosen peak's strength — not an ambiguous case a
threshold could reasonably resolve, the signal itself is just weak there).
Tempo estimates on real percussive/produced audio (e.g. the Stage B/Suno
renders) are more reliable. Not chasing this further right now — full
beat-tracking is its own research area, and "roughly plausible" is the bar
this project is aiming for here, same as elsewhere.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf

N_FFT = 2048
HOP = 512

# Krumhansl-Kessler key profiles — the standard reference profiles for
# correlating a chroma vector against all 24 major/minor key candidates.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def load_mono(path: str, max_duration_s: float | None = None) -> tuple[np.ndarray, int]:
    """max_duration_s trims to a leading excerpt rather than loading the
    whole file — found necessary after a real production failure: a
    12-minute upload drove audio_semantic.py's CLAP pass through ~72
    sequential 10s-window forward passes, easily enough to blow past a
    request timeout on Render's proxy (and its own runtime scales with
    file length the same way audio_features.py's own STFT/RMS passes do).
    Mood/style doesn't need the whole song either way — a fixed-length
    excerpt is a legitimate design choice here, not just a workaround."""
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if max_duration_s is not None:
        data = data[: int(max_duration_s * sr)]
    return data.astype(np.float64), sr


def _stft_magnitude(y: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram, shape (n_frames, N_FFT//2+1). Plain numpy —
    windowed frames, real FFT, no scipy.signal dependency needed."""
    window = np.hanning(N_FFT)
    n_frames = max(1, 1 + (len(y) - N_FFT) // HOP)
    frames = np.empty((n_frames, N_FFT))
    for i in range(n_frames):
        start = i * HOP
        seg = y[start:start + N_FFT]
        if len(seg) < N_FFT:
            seg = np.pad(seg, (0, N_FFT - len(seg)))
        frames[i] = seg * window
    spec = np.fft.rfft(frames, axis=1)
    return np.abs(spec)


def _rms_envelope(y: np.ndarray) -> np.ndarray:
    n_frames = max(1, 1 + (len(y) - N_FFT) // HOP)
    rms = np.empty(n_frames)
    for i in range(n_frames):
        start = i * HOP
        seg = y[start:start + N_FFT]
        rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0
    return rms


def estimate_tempo(rms: np.ndarray, sr: int) -> float:
    """Autocorrelation of the onset-strength envelope (half-wave-rectified
    energy flux) — a simple, standard-enough tempo estimator without
    needing a dedicated beat-tracking library."""
    onset = np.diff(rms, prepend=rms[0])
    onset[onset < 0] = 0
    onset = onset - onset.mean()

    frame_rate = sr / HOP
    min_lag = int(frame_rate * 60 / 200)  # 200 BPM upper bound
    max_lag = int(frame_rate * 60 / 50)   # 50 BPM lower bound
    if len(onset) <= max_lag:
        return 100.0  # too short a clip to estimate reliably — a plain default

    corr = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    window = corr[min_lag:max_lag]
    if window.size == 0 or not np.any(window > 0):
        return 100.0
    best_lag = min_lag + int(np.argmax(window))

    # autocorrelation tempo estimators are prone to a well-known "octave
    # error" — locking onto half the true tempo, since a half-time pulse is
    # also a real periodicity in most music. Caught this directly: a known
    # 108bpm track came back as 53.8. Cheap, standard mitigation: if the
    # correlation strength one octave up (half the lag = double the tempo)
    # is nearly as strong as the chosen peak, prefer the faster tempo,
    # since sub-70bpm is the less common case for the kind of material this
    # project generates.
    bpm = 60.0 * frame_rate / best_lag
    half_lag = best_lag // 2
    if bpm < 70 and half_lag >= min_lag and corr[half_lag] > 0.7 * corr[best_lag]:
        best_lag = half_lag
        bpm = 60.0 * frame_rate / best_lag

    return float(np.clip(bpm, 50, 200))


def estimate_key(magnitude: np.ndarray, sr: int) -> dict:
    """Chroma vector (sum of spectral magnitude per pitch class) correlated
    against all 24 Krumhansl-Kessler major/minor profiles."""
    freqs = np.fft.rfftfreq(N_FFT, d=1 / sr)
    chroma = np.zeros(12)
    with np.errstate(divide="ignore"):
        midi = 69 + 12 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
    pitch_class = np.round(midi).astype(int) % 12
    valid = freqs > 20  # ignore DC/sub-audible bins, whose pitch class is meaningless noise
    total_mag = magnitude.sum(axis=0)
    for pc in range(12):
        chroma[pc] = total_mag[valid & (pitch_class == pc)].sum()
    if chroma.sum() > 0:
        chroma = chroma / chroma.sum()

    best = {"score": -1e9}
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = np.roll(profile, tonic)
            score = float(np.corrcoef(chroma, rotated)[0, 1]) if chroma.std() > 0 else 0.0
            if score > best["score"]:
                best = {"score": score, "tonic": tonic, "mode": mode}
    return {
        "tonic_pc": best["tonic"],
        "tonic_name": PITCH_CLASS_NAMES[best["tonic"]],
        "mode": best["mode"],
        "confidence": round(max(0.0, best["score"]), 3),
        "chroma": [round(float(c), 4) for c in chroma],
    }


def spectral_centroid(magnitude: np.ndarray, sr: int) -> float:
    freqs = np.fft.rfftfreq(N_FFT, d=1 / sr)
    total = magnitude.sum(axis=1)
    centroid = np.zeros(magnitude.shape[0])
    nonzero = total > 0
    centroid[nonzero] = (magnitude[nonzero] @ freqs) / total[nonzero]
    return float(np.mean(centroid)) if len(centroid) else 0.0


def spectral_rolloff(magnitude: np.ndarray, sr: int, pct: float = 0.85) -> float:
    freqs = np.fft.rfftfreq(N_FFT, d=1 / sr)
    total = magnitude.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(magnitude, axis=1)
    threshold = pct * total
    rolloff = np.zeros(magnitude.shape[0])
    for i in range(magnitude.shape[0]):
        idx = np.searchsorted(cumulative[i], threshold[i, 0])
        rolloff[i] = freqs[min(idx, len(freqs) - 1)]
    return float(np.mean(rolloff)) if len(rolloff) else 0.0


def zero_crossing_rate(y: np.ndarray) -> float:
    signs = np.sign(y)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs)) > 0
    return float(np.mean(crossings))


def onset_density(rms: np.ndarray, sr: int, duration_s: float) -> float:
    onset = np.diff(rms, prepend=rms[0])
    onset[onset < 0] = 0
    if onset.max() > 0:
        onset = onset / onset.max()
    threshold = 0.15
    peaks = 0
    above = False
    for v in onset:
        if v > threshold and not above:
            peaks += 1
            above = True
        elif v <= threshold:
            above = False
    return peaks / max(duration_s, 0.1)


def dynamic_envelope(rms: np.ndarray, n: int = 6) -> list[float]:
    if len(rms) == 0:
        return [0.5] * n
    chunks = np.array_split(rms, n)
    means = np.array([c.mean() if len(c) else 0.0 for c in chunks])
    peak = means.max()
    if peak > 0:
        means = means / peak
    return [round(float(v), 3) for v in means]


def extract_features(path: str, max_duration_s: float | None = None) -> dict:
    y, sr = load_mono(path, max_duration_s)
    duration_s = len(y) / sr if sr else 0.0
    magnitude = _stft_magnitude(y)
    rms = _rms_envelope(y)

    tempo = estimate_tempo(rms, sr)
    key = estimate_key(magnitude, sr)
    centroid = spectral_centroid(magnitude, sr)
    rolloff = spectral_rolloff(magnitude, sr)
    zcr = zero_crossing_rate(y)
    density = onset_density(rms, sr, duration_s)
    envelope = dynamic_envelope(rms)

    nyquist = sr / 2
    return {
        "duration_s": round(duration_s, 2),
        "tempo_bpm": round(tempo, 1),
        "key": key,
        # normalized 0-1 versions, calibrated against an 8-file reference
        # set (the project's own Stage A + Stage B renders) the same way
        # feature_extraction.py's CV constants were — the first divisors
        # tried here (4.0, *6) saturated rhythmic_density and loudness_rms
        # at 1.0 for half the set, caught by checking actual unclipped
        # values before shipping rather than after; these replacements
        # spread the real observed range instead. Revisit with a larger,
        # more varied sample later, same caveat as the CV side.
        "brightness": float(np.clip(centroid / (nyquist * 0.35), 0, 1)),
        "texture_richness": float(np.clip(rolloff / (nyquist * 0.5), 0, 1)),
        "noisiness": float(np.clip(zcr * 8, 0, 1)),
        "rhythmic_density": float(np.clip(density / 15.0, 0, 1)),
        "loudness_rms": float(np.clip(np.sqrt(np.mean(rms ** 2)) * 4.5, 0, 1)),
        "dynamic_curve": envelope,
        "raw": {
            "spectral_centroid_hz": round(centroid, 1),
            "spectral_rolloff_hz": round(rolloff, 1),
            "zero_crossing_rate": round(zcr, 4),
            "onset_density_per_sec": round(density, 3),
        },
    }

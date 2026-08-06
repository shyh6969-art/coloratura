"""
Coloratura — MIDI -> WAV, without FluidSynth.

No soundfont-based synthesizer (FluidSynth) was available in this
environment — not in pip, and winget had no package for it either, and
installing a system binary + a soundfont file isn't something to force
through a non-interactive sandbox. So this is a small self-contained
additive synthesizer instead: numpy generates a handful of harmonic sine
partials per note, shaped by a per-GM-instrument-family envelope (piano
decays continuously from the attack; organ holds flat until released;
strings/brass get a slower attack and a bit of vibrato; etc.), mixed to a
stereo buffer and written out with scipy — no external binaries, no
downloaded assets.

Revised after direct user feedback that the first version sounded "too
synthesized... like an 80s synthesizer" and lacked the roundness of real
instruments. That's an accurate description of what pure clean-oscillator
additive synthesis with no room, no ensemble detuning, and no attack noise
actually sounds like — those are exactly the things separating a clean
digital tone from a real instrument, so this version adds them, still
entirely self-built (no sample libraries, no impulse-response assets,
same instinct as before):
  - a hand-rolled Schroeder-style algorithmic reverb (parallel combs into
    series allpasses, implemented as vectorized IIR filters via
    scipy.signal.lfilter rather than a per-sample Python loop, which would
    be far too slow at 44.1kHz over a multi-minute track)
  - unison detuning (2-3 slightly mistuned voices per note) on sustained
    families, the same trick real synths and orchestras get "bigness" and
    "beating" from that a single pure oscillator structurally cannot have
  - vibrato extended to every sustained family, not just strings/brass
  - a short filtered noise burst at note onset for bowed/blown families
    (strings/brass/reed/pipe), approximating bow/breath noise
  - gentle tanh soft-saturation on the final mix instead of pure linear
    summing, for a touch of analog-style warmth

Anyone who later wires in a real soundfont (composer.py's Stage B, per the
spec doc) will still get a categorically better sound than this. This
module's job is to make the MIDI files linked from other modules sound as
good as a hand-built synthesizer reasonably can, not to reach sample-
library realism.
"""

from __future__ import annotations

import numpy as np
import pretty_midi
from scipy.io import wavfile
from scipy.signal import lfilter

SR = 44100


def _family_for_program(program: int) -> str:
    if 0 <= program <= 7:
        return "piano"
    if 8 <= program <= 15:
        return "bell"
    if 16 <= program <= 23:
        return "organ"
    if program in (44, 45, 46) or 104 <= program <= 111:
        return "guitar"
    if 24 <= program <= 31:
        return "guitar"
    if 32 <= program <= 39:
        return "bass"
    if 40 <= program <= 54:
        return "strings"
    if 56 <= program <= 63:
        return "brass"
    if 64 <= program <= 71:
        return "reed"
    if 72 <= program <= 79:
        return "pipe"
    if 80 <= program <= 103:
        return "pad"
    if 112 <= program <= 127:
        return "mallet"
    return "piano"


# (harmonic ratio, relative amplitude) pairs per family, plus envelope shape.
# "percussive" envelopes decay continuously from the attack peak across the
# note's own duration; "sustained" envelopes hold a plateau and release
# after note-off, extending the render past the note's nominal end.
# unison_cents: detune spread for a 3-voice unison (0, +-cents) on sustained
# families -- the single biggest lever for "one pure tone" vs. "an ensemble
# with body," per the user feedback this revision responds to.
# attack_noise: True for bowed/blown families -- a short filtered noise
# burst at onset, approximating bow/breath noise real instruments have and
# pure harmonic synthesis structurally lacks.
FAMILIES = {
    "piano":   dict(harm=[(1, 1.0), (2, .5), (3, .25), (4, .15), (5, .1), (6, .06)],
                     shape="percussive", attack=0.004, decay_to=0.04),
    "bell":    dict(harm=[(1, 1.0), (2.4, .5), (3.8, .3), (5.4, .18), (7.1, .1)],
                     shape="percussive", attack=0.002, decay_to=0.02),
    "mallet":  dict(harm=[(1, 1.0), (2.0, .4), (3.9, .25), (5.2, .12)],
                     shape="percussive", attack=0.002, decay_to=0.03),
    "guitar":  dict(harm=[(1, 1.0), (2, .7), (3, .5), (4, .3), (5, .18), (6, .1)],
                     shape="percussive", attack=0.003, decay_to=0.06),
    "organ":   dict(harm=[(1, 1.0), (2, .6), (3, .4), (4, .3), (5, .2), (6, .15), (8, .1)],
                     shape="sustained", attack=0.02, release=0.05, sustain=1.0,
                     unison_cents=6, vibrato_rate=4.2, vibrato_depth=0.0025),
    "bass":    dict(harm=[(1, 1.0), (2, .4), (3, .15)],
                     shape="sustained", attack=0.02, release=0.08, sustain=0.85,
                     unison_cents=4),
    "strings": dict(harm=[(1, 1.0), (2, .55), (3, .35), (4, .22), (5, .14), (6, .08)],
                     shape="sustained", attack=0.10, release=0.15, sustain=0.75,
                     vibrato_rate=5.3, vibrato_depth=0.007, unison_cents=9, attack_noise=True),
    "brass":   dict(harm=[(1, 1.0), (2, .7), (3, .55), (4, .4), (5, .28), (6, .18), (7, .1)],
                     shape="sustained", attack=0.035, release=0.10, sustain=0.85,
                     vibrato_rate=5.0, vibrato_depth=0.005, unison_cents=7, attack_noise=True),
    "reed":    dict(harm=[(1, 1.0), (2, .2), (3, .5), (4, .1), (5, .3), (7, .12)],
                     shape="sustained", attack=0.045, release=0.09, sustain=0.8,
                     vibrato_rate=4.6, vibrato_depth=0.005, unison_cents=6, attack_noise=True),
    "pipe":    dict(harm=[(1, 1.0), (2, .18), (3, .08)],
                     shape="sustained", attack=0.06, release=0.12, sustain=0.75,
                     vibrato_rate=4.0, vibrato_depth=0.004, attack_noise=True),
    "pad":     dict(harm=[(1, 1.0), (2, .4), (3, .25), (4, .15)],
                     shape="sustained", attack=0.18, release=0.25, sustain=0.7,
                     vibrato_rate=3.6, vibrato_depth=0.004, unison_cents=8),
}


def _envelope(n: int, sr: int, attack_s: float, body_shape: str, **kw) -> np.ndarray:
    env = np.ones(n)
    a = min(n, int(attack_s * sr))
    if a > 0:
        # slightly curved attack (x**1.6) rather than a dead-straight linear
        # ramp — a small change, but a perfectly linear attack is one more
        # small "too clean" tell; real transients aren't linear.
        env[:a] = np.linspace(0, 1, a) ** 1.6
    if body_shape == "percussive":
        decay_to = kw.get("decay_to", 0.05)
        tail = n - a
        if tail > 0:
            rate = -np.log(max(decay_to, 1e-4)) / tail
            env[a:] = np.exp(-rate * np.arange(tail))
    else:  # sustained: attack -> flat sustain -> release ramp at the very end
        sustain = kw.get("sustain", 0.8)
        release_n = min(n - a, int(kw.get("release", 0.1) * sr))
        body_end = n - release_n
        if body_end > a:
            env[a:body_end] = sustain
        if release_n > 0:
            env[body_end:] = np.linspace(sustain, 0, release_n) ** 1.3
    return env


def _onset_noise(n: int, sr: int, rng: np.random.Generator, center_hz: float) -> np.ndarray:
    """A short band-limited noise burst at note onset -- bow/breath noise,
    the transient-noise component pure harmonic synthesis has no way to
    produce on its own. Band-passed loosely around the note's own
    fundamental via a simple leaky-integrator + differencer pair (cheap,
    no scipy filter design needed) so it reads as breathy/bowed rather than
    as a broadband hiss or click."""
    burst_len = min(n, int(sr * 0.02))
    if burst_len < 4:
        return np.zeros(n)
    noise = rng.standard_normal(burst_len)
    # crude bandpass: a short moving-average lowpass, then a first-difference
    # highpass, tuned loosely by the note's own pitch register
    smooth_n = max(1, int(sr / max(center_hz, 80) / 2))
    kernel = np.ones(smooth_n) / smooth_n
    low = np.convolve(noise, kernel, mode="same")
    band = np.diff(low, prepend=low[0])
    band = band / (np.max(np.abs(band)) + 1e-9)
    env = np.exp(-np.arange(burst_len) / (burst_len * 0.35))
    out = np.zeros(n)
    out[:burst_len] = band * env
    return out


def _render_note(pitch: int, velocity: int, duration: float, family: str, rng: np.random.Generator) -> np.ndarray:
    fam = FAMILIES[family]
    tail = fam.get("release", 0.0) if fam["shape"] == "sustained" else 0.0
    n = max(1, int(SR * (duration + tail)))
    t = np.arange(n) / SR
    base_freq = 440.0 * 2 ** ((pitch - 69) / 12)

    vib_rate, vib_depth = fam.get("vibrato_rate"), fam.get("vibrato_depth")
    unison_cents = fam.get("unison_cents", 0)
    # 1 voice if no unison, else 3 (center + two detuned) -- a real 3-voice
    # unison rather than just 2, since 2 detuned voices alone tend to
    # produce a single slow beat rather than the denser "chorus" texture
    # 3+ voices give.
    detunes = [0.0] if not unison_cents else [-unison_cents, 0.0, unison_cents]
    # slight per-note phase jitter so identical pitches played twice aren't
    # bit-identical -- a small humanization touch, not load-bearing for
    # anything else in the pipeline, so an unseeded RNG draw is fine here.
    phase_jitter = rng.uniform(0, 2 * np.pi, size=len(detunes))

    wave = np.zeros(n)
    for voice_i, cents in enumerate(detunes):
        freq = base_freq * (2 ** (cents / 1200))
        for ratio, amp in fam["harm"]:
            f = freq * ratio
            if vib_rate:
                inst_freq = f * (1 + vib_depth * np.sin(2 * np.pi * vib_rate * t + voice_i))
                phase = phase_jitter[voice_i] + 2 * np.pi * np.cumsum(inst_freq) / SR
            else:
                phase = phase_jitter[voice_i] + 2 * np.pi * f * t
            wave += amp * np.sin(phase)
    wave /= sum(a for _, a in fam["harm"]) * len(detunes)

    if fam.get("attack_noise"):
        wave = wave + 0.10 * _onset_noise(n, SR, rng, base_freq)

    env = _envelope(n, SR, fam["attack"], fam["shape"], **{k: v for k, v in fam.items()
                     if k in ("decay_to", "sustain", "release")})
    return wave * env * (velocity / 127.0)


def _delay_recursive_filter(x: np.ndarray, delay: int, b: list[float], a: list[float]) -> np.ndarray:
    """A single feedback/feedforward tap at `delay` samples, applied
    efficiently. A direct scipy.signal.lfilter call with a delay-length
    coefficient array works but costs O(delay) work per output sample
    regardless of how many of those coefficients are actually zero (lfilter
    doesn't exploit sparsity) -- measured directly at 32s to render one
    short piece with delays in the ~1300-1900 sample range this reverb
    uses, an unacceptable cost for something that runs synchronously in
    /api/analyze's request cycle. Fix: de-interleave the signal into
    `delay` independent subsequences (each exactly `delay` samples apart in
    the original), each of which reduces the same recursion to an ordinary
    2-tap IIR filter; run all of them at once via lfilter's axis parameter,
    then re-interleave. Same math, O(1) work per output sample instead of
    O(delay)."""
    n = len(x)
    pad = (-n) % delay
    xp = np.concatenate([x, np.zeros(pad)]) if pad else x
    blocks = xp.reshape(-1, delay).T  # shape (delay, n_blocks); row j = x[j::delay]
    filtered = lfilter(b, a, blocks, axis=1)
    out = filtered.T.reshape(-1)
    return out[:n]


def _comb_filter(x: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    return _delay_recursive_filter(x, delay, [1.0], [1.0, -feedback])


def _allpass_filter(x: np.ndarray, delay: int, feedback: float) -> np.ndarray:
    return _delay_recursive_filter(x, delay, [-feedback, 1.0], [1.0, -feedback])


def _reverb_channel(x: np.ndarray, sr: int, comb_ms: tuple[float, ...]) -> np.ndarray:
    """Schroeder reverb: 4 parallel feedback comb filters summed, then 2
    series allpass filters for diffusion. Implemented as vectorized IIR
    filters (scipy.signal.lfilter) rather than a per-sample Python loop --
    a naive sample-by-sample loop would take minutes per track at 44.1kHz,
    lfilter runs the same recursion in compiled code."""
    comb_delays = [max(1, int(sr * ms / 1000)) for ms in comb_ms]
    comb_sum = sum(_comb_filter(x, d, 0.77) for d in comb_delays) / len(comb_delays)
    ap = _allpass_filter(comb_sum, max(1, int(sr * 0.005)), 0.5)
    ap = _allpass_filter(ap, max(1, int(sr * 0.0017)), 0.5)
    return ap


def _apply_reverb(left: np.ndarray, right: np.ndarray, sr: int, wet: float = 0.16) -> tuple[np.ndarray, np.ndarray]:
    # slightly different comb delay sets per channel decorrelates the two
    # reverb tails instead of collapsing them to the same mono-ish smear --
    # a standard trick for a wider-sounding stereo reverb.
    rev_l = _reverb_channel(left, sr, (29.7, 37.1, 41.1, 43.7))
    rev_r = _reverb_channel(right, sr, (31.3, 35.9, 42.5, 44.9))
    out_l = left * (1 - wet) + rev_l * wet
    out_r = right * (1 - wet) + rev_r * wet
    return out_l, out_r


def render_midi_to_wav(mid_path: str, wav_path: str, pans: dict[str, float] | None = None) -> None:
    """pans: optional {instrument_name: pan} in [-1, 1] (equal-power); any
    instrument not named there gets 0 (center)."""
    pm = pretty_midi.PrettyMIDI(mid_path)
    end_time = pm.get_end_time() + 0.9  # a touch more tail room than before, for the reverb's own decay
    n_total = int(SR * end_time) + 1
    left = np.zeros(n_total)
    right = np.zeros(n_total)
    rng = np.random.default_rng()

    for inst in pm.instruments:
        family = _family_for_program(inst.program)
        pan = (pans or {}).get(inst.name, 0.0)
        lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        for note in inst.notes:
            dur = note.end - note.start
            if dur <= 0:
                continue
            audio = _render_note(note.pitch, note.velocity, dur, family, rng)
            start_i = int(note.start * SR)
            end_i = start_i + len(audio)
            if end_i > n_total:
                audio = audio[: n_total - start_i]
                end_i = n_total
            left[start_i:end_i] += audio * lg
            right[start_i:end_i] += audio * rg

    left, right = _apply_reverb(left, right, SR)

    # gentle tanh soft-saturation for a touch of analog-style warmth,
    # instead of pure linear summing straight to a hard peak-normalize
    drive = 1.35
    left = np.tanh(left * drive) / np.tanh(drive)
    right = np.tanh(right * drive) / np.tanh(drive)

    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.92
    wavfile.write(wav_path, SR, (stereo * 32767).astype(np.int16))


LITE_PANS = {"chords": -0.25, "melody": 0.3}
STAGE_A_PANS = {"bass": -0.5, "tenor": -0.2, "alto": 0.2, "soprano": 0.5}

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

Anyone who later wires in a real soundfont (composer.py's Stage B, per the
spec doc) will get a categorically better sound than this. This module's
job is only to make the MIDI files linked from other modules audible
without installing anything.
"""

from __future__ import annotations

import numpy as np
import pretty_midi
from scipy.io import wavfile

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
                     shape="sustained", attack=0.02, release=0.05, sustain=1.0),
    "bass":    dict(harm=[(1, 1.0), (2, .4), (3, .15)],
                     shape="sustained", attack=0.02, release=0.08, sustain=0.85),
    "strings": dict(harm=[(1, 1.0), (2, .55), (3, .35), (4, .22), (5, .14), (6, .08)],
                     shape="sustained", attack=0.10, release=0.15, sustain=0.75,
                     vibrato_rate=5.3, vibrato_depth=0.006),
    "brass":   dict(harm=[(1, 1.0), (2, .7), (3, .55), (4, .4), (5, .28), (6, .18), (7, .1)],
                     shape="sustained", attack=0.035, release=0.10, sustain=0.85,
                     vibrato_rate=5.0, vibrato_depth=0.004),
    "reed":    dict(harm=[(1, 1.0), (2, .2), (3, .5), (4, .1), (5, .3), (7, .12)],
                     shape="sustained", attack=0.045, release=0.09, sustain=0.8),
    "pipe":    dict(harm=[(1, 1.0), (2, .18), (3, .08)],
                     shape="sustained", attack=0.06, release=0.12, sustain=0.75),
    "pad":     dict(harm=[(1, 1.0), (2, .4), (3, .25), (4, .15)],
                     shape="sustained", attack=0.18, release=0.25, sustain=0.7),
}


def _envelope(n: int, sr: int, attack_s: float, body_shape: str, **kw) -> np.ndarray:
    env = np.ones(n)
    a = min(n, int(attack_s * sr))
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
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
            env[body_end:] = np.linspace(sustain, 0, release_n)
    return env


def _render_note(pitch: int, velocity: int, duration: float, family: str) -> np.ndarray:
    fam = FAMILIES[family]
    tail = fam.get("release", 0.0) if fam["shape"] == "sustained" else 0.0
    n = max(1, int(SR * (duration + tail)))
    t = np.arange(n) / SR
    freq = 440.0 * 2 ** ((pitch - 69) / 12)

    vib_rate, vib_depth = fam.get("vibrato_rate"), fam.get("vibrato_depth")
    wave = np.zeros(n)
    for ratio, amp in fam["harm"]:
        f = freq * ratio
        if vib_rate:
            inst_freq = f * (1 + vib_depth * np.sin(2 * np.pi * vib_rate * t))
            phase = 2 * np.pi * np.cumsum(inst_freq) / SR
        else:
            phase = 2 * np.pi * f * t
        wave += amp * np.sin(phase)
    wave /= sum(a for _, a in fam["harm"])

    env = _envelope(n, SR, fam["attack"], fam["shape"], **{k: v for k, v in fam.items()
                     if k in ("decay_to", "sustain", "release")})
    return wave * env * (velocity / 127.0)


def render_midi_to_wav(mid_path: str, wav_path: str, pans: dict[str, float] | None = None) -> None:
    """pans: optional {instrument_name: pan} in [-1, 1] (equal-power); any
    instrument not named there gets 0 (center)."""
    pm = pretty_midi.PrettyMIDI(mid_path)
    end_time = pm.get_end_time() + 0.6
    n_total = int(SR * end_time) + 1
    left = np.zeros(n_total)
    right = np.zeros(n_total)

    for inst in pm.instruments:
        family = _family_for_program(inst.program)
        pan = (pans or {}).get(inst.name, 0.0)
        lg, rg = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        for note in inst.notes:
            dur = note.end - note.start
            if dur <= 0:
                continue
            audio = _render_note(note.pitch, note.velocity, dur, family)
            start_i = int(note.start * SR)
            end_i = start_i + len(audio)
            if end_i > n_total:
                audio = audio[: n_total - start_i]
                end_i = n_total
            left[start_i:end_i] += audio * lg
            right[start_i:end_i] += audio * rg

    stereo = np.stack([left, right], axis=1)
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.92
    wavfile.write(wav_path, SR, (stereo * 32767).astype(np.int16))


LITE_PANS = {"chords": -0.25, "melody": 0.3}
STAGE_A_PANS = {"bass": -0.5, "tenor": -0.2, "alto": 0.2, "soprano": 0.5}

"""
Coloratura — Stage A "lite": brief -> playable MIDI.

Not the real composition engine described in the spec doc (section ח,
Stage A) — that would enforce actual voice-leading and counterpoint rules.
This is deliberately smaller: enough harmonic and melodic logic to make the
brief *audible*, so the mapping decisions can be sanity-checked by ear the
same way the JSON briefs were sanity-checked by eye (output/report.txt).
If it turns out even this lite version sounds wrong for some painting,
that's cheaper to learn now than after building the real engine.

One bar per dynamic_curve sample — the brief's own timeline becomes the
piece's bar structure directly, so climax_position and harmonic_resolution
are audible, not just numbers in a JSON file:
  - a fixed 6-position scale-degree progression (I-IV-V-vi-IV-I-ish, spelled
    diatonically in whatever mode the brief picked) walks through the bars
  - if engine_params.resolved is False, the final bar's degree is swapped
    for the 7th scale degree instead of the tonic, so the piece harmonically
    just stops rather than resolving
  - the bar closest to climax_position gets extra chord tones when
    harmonic_complexity_level allows it, and the loudest velocity
  - articulation and note density come from engine_params.staccato and the
    brief's arousal, not fixed values
"""

from __future__ import annotations

import hashlib
import random

import pretty_midi

# Hebrew instrument names (mapping_engine._instrumentation) -> General MIDI
# program number. Approximate on purpose — this is the "lite" render tier;
# real orchestration belongs to Stage B (spec doc, section ח) if this ever
# needs to sound genuinely good rather than just legible.
GM_PROGRAM = {
    "מיתרים": 48, "קוורטט כלי-קשת": 48, "פאד מיתרים גבוה": 50,
    "קרן יער": 60, "פסנתר": 0, "פסנתר פרפטואום מוביל": 0,
    "פעמוני צינור": 14, "קלרינט": 71, "נבל": 46, "חלילית": 72,
    "נחושת מעוותת (con sordino)": 59, "פרקאשן לא-מכוון": 47,
    "מרימבה פעימתית": 12, "כלי נשיפה-עץ": 68, "קונטרבס פיציקטו": 45,
    "תופים חופשיים": 116, "סקסופון אלט": 65,
}
DEFAULT_PROGRAM = 0  # acoustic grand piano, if an instrument name is unmapped

# Scale-degree progression (0-indexed into the 7-note mode scale), applied
# generically across whichever mode _mode() picked. Deliberately plain —
# I-IV-V-vi-IV-I in Roman-numeral terms — this is the "lite" harmonic
# vocabulary, not an attempt at idiomatic voice-leading per style.
DEGREE_PROGRESSION = [0, 3, 4, 5, 3, 0]
UNRESOLVED_FINAL_DEGREE = 6  # leading-tone-ish chord, used when resolved=False


def _scale_pitch(tonic_pc: int, intervals: tuple[int, ...], octave: int, degree: int) -> int:
    """MIDI pitch for a scale degree, wrapping into higher octaves past the
    7th degree (so stacking thirds past the octave boundary works)."""
    octave_shift, deg = divmod(degree, len(intervals))
    return 12 * (octave + 1 + octave_shift) + tonic_pc + intervals[deg]


def _chord_tones(tonic_pc, intervals, octave, degree, extra_tones):
    """Stack diatonic thirds on top of `degree`: root, 3rd, 5th, optionally
    7th and 9th (extra_tones 0/1/2) — all still inside the given mode's own
    interval set, not an ad hoc chromatic chord."""
    stack = [degree, degree + 2, degree + 4]
    if extra_tones >= 1:
        stack.append(degree + 6)
    if extra_tones >= 2:
        stack.append(degree + 8)
    return [_scale_pitch(tonic_pc, intervals, octave, d) for d in stack]


def compose_midi(brief: dict, out_path: str) -> None:
    ep = brief["engine_params"]
    tonic_pc, intervals, octave = ep["tonic_pc"], ep["mode_intervals"], ep["register_octave"]
    tempo = brief["tempo_bpm"]
    arousal = brief["vat"]["arousal"]
    staccato = ep["staccato"]
    complexity = ep["harmonic_complexity_level"]
    resolved = ep["resolved"]
    curve = brief["dynamic_curve"]
    n_bars = len(curve)
    climax_bar = round(brief["climax_position"] * (n_bars - 1))

    rng = random.Random(int(hashlib.sha256(brief["source_image"].encode("utf-8")).hexdigest(), 16))

    # This is the "lite" simplification flagged in the module docstring:
    # every meter is counted in quarter-note beats regardless of the
    # denominator the brief picked (so a 7/8 bar plays as 7 quarters, not 7
    # eighths) — irregular bar *lengths* come through, true compound-meter
    # subdivision doesn't yet.
    sec_per_beat = 60.0 / tempo
    bar_len = ep["meter_numerator"] * sec_per_beat

    programs = brief["instrumentation"]
    chord_program = GM_PROGRAM.get(programs[0], DEFAULT_PROGRAM) if programs else DEFAULT_PROGRAM
    melody_program = GM_PROGRAM.get(programs[1], chord_program) if len(programs) > 1 else chord_program

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    chords = pretty_midi.Instrument(program=chord_program, name="chords")
    melody = pretty_midi.Instrument(program=melody_program, name="melody")

    note_count_by_arousal = 2 if arousal < 0.35 else (4 if arousal < 0.65 else 8)

    for i in range(n_bars):
        bar_start = i * bar_len
        velocity = int(round(30 + _clip01(curve[i]) * 90))
        is_climax = i == climax_bar

        degree = DEGREE_PROGRESSION[i % len(DEGREE_PROGRESSION)]
        if i == n_bars - 1 and not resolved:
            degree = UNRESOLVED_FINAL_DEGREE
        extra_tones = complexity if (complexity >= 2 or is_climax) else 0

        for pitch in _chord_tones(tonic_pc, intervals, octave - 1, degree, extra_tones):
            chords.notes.append(pretty_midi.Note(
                velocity=max(20, velocity - 15), pitch=pitch,
                start=bar_start, end=bar_start + bar_len * 0.95,
            ))

        # melody: walk the scale around the chord tones for this bar
        n_notes = note_count_by_arousal + (2 if is_climax else 0)
        slot = bar_len / n_notes
        gate = 0.55 if staccato >= 0.55 else 0.9  # fraction of the slot actually sounded
        deg = degree
        for j in range(n_notes):
            deg += rng.choice([-2, -1, 0, 1, 1, 2]) if j > 0 else 0
            pitch = _scale_pitch(tonic_pc, intervals, octave, deg)
            note_start = bar_start + j * slot
            melody.notes.append(pretty_midi.Note(
                velocity=min(127, velocity + rng.randint(-5, 8)),
                pitch=pitch, start=note_start, end=note_start + slot * gate,
            ))

    pm.instruments.append(chords)
    pm.instruments.append(melody)
    pm.write(out_path)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

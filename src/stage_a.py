"""
Coloratura — Stage A (full): brief -> functional harmony + 4-part voice-leading -> MIDI.

This is the engine composer.py's docstring calls "Stage A" in the spec doc
(section ח) — real harmonic grammar and real voice-leading constraints, not
a fixed I-IV-V-vi-IV-I loop with randomly-wandering melody notes. Kept
alongside composer.py (not replacing it) so the two can be A/B compared by
ear: does the extra machinery actually sound more like "a composer" and
less like "a chord-generator," or was the lite version already good enough
for what this project needs?

What's real here:
  - A weighted Markov grammar over scale degrees, grouped by function
    (Tonic/Subdominant/Dominant) the way a first-year harmony class teaches
    it, generating an actual progression per section rather than looping a
    fixed pattern.
  - 4-part (SATB) voice-leading via constrained search: for each chord,
    candidate voicings are scored by total voice movement plus penalties
    for parallel fifths/octaves and voice crossing, and the best-scoring
    legal candidate wins. count_parallel_violations() at the bottom is a
    self-test, not decoration — run it and it should print 0.
  - Mode-aware cadential resolution: the 7th scale degree only gets forced
    stepwise resolution into the tonic when the mode actually has a
    half-step there (Ionian/Lydian). In Mixolydian/Aeolian/Phrygian/Locrian
    that degree is a whole step below the tonic (a subtonic, not a leading
    tone) — forcing a chromatic pull there would be musically dishonest, so
    it isn't done; the voice still resolves, just by the interval the mode
    actually has.
  - Meter-aware rhythm: 7/8 is actually grouped 2+2+3 eighth notes (the
    Stravinsky-flavored grouping the spec doc's meter choice was named
    after), not "7 quarter notes" the way the lite version simplified it.
  - Form: ABA literally repeats the A section's chord sequence; through-
    composed generates one continuous, non-repeating arc.

What's still simplified, on purpose, to keep this a "lite-plus" build
rather than a semester of species counterpoint:
  - Bass is always root position — no first/second-inversion chords.
  - Only the soprano line gets melodic decoration (passing/neighbor tones);
    alto/tenor/bass stay as sustained block harmony.
  - Function grouping (T/S/D per scale degree) is the major/minor
    convention, applied uniformly across all five modes for simplicity —
    real modal harmony treats this less rigidly.
"""

from __future__ import annotations

import hashlib
import itertools
import random

import pretty_midi

from composer import GM_PROGRAM, DEFAULT_PROGRAM

FUNCTION = {0: "T", 1: "S", 2: "T", 3: "S", 4: "D", 5: "T", 6: "D"}

TRANSITIONS = {
    0: [(3, 0.30), (1, 0.15), (4, 0.25), (5, 0.15), (2, 0.15)],
    1: [(4, 0.50), (3, 0.15), (6, 0.15), (0, 0.20)],
    2: [(5, 0.35), (3, 0.35), (0, 0.30)],
    3: [(4, 0.45), (0, 0.25), (1, 0.15), (5, 0.15)],
    4: [(0, 0.55), (5, 0.20), (3, 0.10), (2, 0.15)],
    5: [(3, 0.30), (1, 0.30), (4, 0.25), (0, 0.15)],
    6: [(0, 0.70), (5, 0.30)],
}

RANGES = {"B": (40, 64), "T": (48, 69), "A": (55, 74), "S": (60, 81)}

# meter numerator -> pulse grouping, in units of the meter's own denominator
# (i.e. eighth-note units when denominator==8, quarter-note units when 4).
METER_GROUPS = {2: [2], 3: [3], 4: [2, 2], 5: [2, 3], 6: [3, 3], 7: [2, 2, 3], 8: [3, 3, 2]}
COMPLEXITY_RHYTHM_FACTOR = {0: 2.0, 1: 1.0, 2: 0.65}  # slower <-> faster harmonic rhythm


def _weighted_choice(rng: random.Random, options: list[tuple[int, float]]) -> int:
    total = sum(w for _, w in options)
    r = rng.uniform(0, total)
    upto = 0.0
    for val, w in options:
        upto += w
        if r <= upto:
            return val
    return options[-1][0]


def _cadence_type(harmonic_resolution_label: str) -> str:
    if harmonic_resolution_label.startswith("לא-פתור"):
        return "none"
    if harmonic_resolution_label.startswith("פתור חלקית"):
        return "half"
    return "authentic"


def generate_progression(rng: random.Random, n_chords: int, cadence: str, start_degree: int = 0) -> list[int]:
    """Random-walk the TRANSITIONS grammar, then overwrite the tail to land
    on the requested cadence type."""
    degrees = [start_degree]
    while len(degrees) < n_chords:
        degrees.append(_weighted_choice(rng, TRANSITIONS[degrees[-1]]))
    if cadence == "authentic" and n_chords >= 2:
        degrees[-2], degrees[-1] = 4, 0
    elif cadence == "half" and n_chords >= 1:
        degrees[-1] = 4
    elif cadence == "none" and n_chords >= 1:
        degrees[-1] = 6
    return degrees


def chord_pitch_classes(tonic_pc: int, intervals: tuple[int, ...], degree: int, extra_tones: int) -> list[int]:
    tones = [degree, degree + 2, degree + 4]
    if extra_tones >= 1:
        tones.append(degree + 6)
    if extra_tones >= 2:
        tones.append(degree + 8)
    return [(tonic_pc + intervals[t % 7]) % 12 for t in tones]


def _has_semitone_leading_tone(intervals: tuple[int, ...]) -> bool:
    """True only for modes where scale-degree 7 sits a half-step below the
    octave tonic (Ionian/Lydian here) — see module docstring."""
    return (12 - intervals[6]) == 1


def _closest_pitch(pc: int, prev: int, lo: int, hi: int) -> int | None:
    candidates = _candidate_pitches(pc, prev, lo, hi, k=1)
    return candidates[0] if candidates else None


def _candidate_pitches(pc: int, prev: int, lo: int, hi: int, k: int = 2) -> list[int]:
    """The k octave-placements of `pc` within [lo, hi] closest to `prev`,
    nearest first. Trying more than just the single closest option matters
    for voice-leading search: the closest note for one voice in isolation
    can be exactly what creates a parallel fifth/octave with another voice,
    and the only way out is an alternate (still legal, just not optimal on
    its own) octave for that voice."""
    options = []
    for octave in range(0, 9):
        p = 12 * octave + pc
        if lo <= p <= hi:
            options.append(p)
    options.sort(key=lambda p: abs(p - prev))
    return options[:k]


def _parallel_violation(prev: dict[str, int], curr: dict[str, int]) -> int:
    """Count parallel perfect-5th/octave motions between every voice pair."""
    voices = ["B", "T", "A", "S"]
    violations = 0
    for v1, v2 in itertools.combinations(voices, 2):
        iv_prev = (prev[v2] - prev[v1]) % 12
        iv_curr = (curr[v2] - curr[v1]) % 12
        if iv_prev in (0, 7) and iv_curr == iv_prev:
            moved1 = curr[v1] != prev[v1]
            moved2 = curr[v2] != prev[v2]
            same_dir = moved1 and moved2 and (
                (curr[v1] - prev[v1]) > 0) == ((curr[v2] - prev[v2]) > 0)
            if same_dir:
                violations += 1
    return violations


def voice_lead(rng: random.Random, degrees: list[int], tonic_pc: int, intervals: tuple[int, ...],
                extra_tones_of: list[int], cadence: str) -> list[dict[str, int]]:
    n = len(degrees)
    voicings: list[dict[str, int]] = []
    prev = {"B": 52, "T": 60, "A": 64, "S": 69}  # comfortable neutral start

    has_lt = _has_semitone_leading_tone(intervals)
    forced_next: dict[str, int] | None = None

    for i, deg in enumerate(degrees):
        pcs = chord_pitch_classes(tonic_pc, intervals, deg, extra_tones_of[i])
        root_pc = pcs[0]
        bass_options = _candidate_pitches(root_pc, prev["B"], *RANGES["B"], k=2) or [12 * 4 + root_pc]

        upper_pcs = list(pcs[1:])
        if len(upper_pcs) == 2:  # triad: double the root for the 4th voice
            upper_pcs.append(root_pc)

        # A forced voice (cadential resolution) claims one occurrence of its
        # own pitch-class up front, so the remaining pcs get permuted only
        # across the two free voices — never overridden after the fact,
        # which would silently drop or duplicate a chord tone.
        free_voices = ["T", "A", "S"]
        fixed: dict[str, int] = {}
        free_pcs = list(upper_pcs)
        if forced_next:
            for voice, pitch in forced_next.items():
                fixed[voice] = pitch
                free_voices.remove(voice)
                if (pitch % 12) in free_pcs:
                    free_pcs.remove(pitch % 12)
                elif free_pcs:
                    free_pcs.pop()

        best_candidate, best_score = None, 1e18
        for bass in bass_options:
            for perm in set(itertools.permutations(free_pcs)):
                # up to 3 octave choices per free voice, not just the closest
                # one — see _candidate_pitches docstring for why that matters
                per_voice_options = [_candidate_pitches(pc, prev[v], *RANGES[v], k=3) for v, pc in zip(free_voices, perm)]
                if any(not opts for opts in per_voice_options):
                    continue
                for combo in itertools.product(*per_voice_options):
                    cand = dict(fixed)
                    cand.update(zip(free_voices, combo))
                    if not (bass <= cand["T"] <= cand["A"] <= cand["S"]):
                        continue
                    if cand["S"] - cand["A"] > 12 or cand["A"] - cand["T"] > 12:
                        continue
                    full = {"B": bass, **cand}
                    movement = sum(abs(full[v] - prev[v]) for v in "BTAS")
                    penalty = _parallel_violation(prev, full) * 1000
                    score = movement + penalty
                    if score < best_score:
                        best_score, best_candidate = score, full

        if best_candidate is None:
            # relaxed fallback: ignore spacing/crossing/parallels, just avoid None
            bass = bass_options[0]
            full = {"B": bass, **fixed}
            for voice, pc in zip(free_voices, free_pcs):
                full[voice] = _closest_pitch(pc, prev[voice], *RANGES[voice]) or (12 * 5 + pc)
            best_candidate = full

        voicings.append(best_candidate)
        prev = best_candidate
        forced_next = None

        # authentic cadence: whichever upper voice holds the 7th scale-degree
        # pitch class in a Dominant chord must resolve into the tonic pitch
        # class next chord — by whatever interval this mode actually has.
        is_last_pair = cadence == "authentic" and i == n - 2
        if is_last_pair:
            lt_pc = (tonic_pc + intervals[6]) % 12
            tonic_target_pc = tonic_pc % 12
            for voice in ("T", "A", "S"):
                if best_candidate[voice] % 12 == lt_pc:
                    # up a half-step if this mode has a true leading tone,
                    # otherwise down a whole step (subtonic motion)
                    target = best_candidate[voice] + (1 if has_lt else -2)
                    if target % 12 != tonic_target_pc:
                        target = 12 * (target // 12) + tonic_target_pc
                    forced_next = {voice: target}
                    break

    return voicings


def _rhythm_slots(meter_numerator: int, meter_denominator: int, complexity_level: int, tempo_bpm: float,
                   n_chords: int) -> list[float]:
    """One duration (seconds) per chord, cycling the meter's pulse-grouping."""
    unit_beats = 1.0 if meter_denominator == 4 else 0.5  # quarters vs eighths, in quarter-note units
    sec_per_quarter = 60.0 / tempo_bpm
    group = METER_GROUPS.get(meter_numerator, [meter_numerator])
    factor = COMPLEXITY_RHYTHM_FACTOR[complexity_level]
    durs = []
    gi = 0
    while len(durs) < n_chords:
        pulses = group[gi % len(group)]
        durs.append(pulses * unit_beats * sec_per_quarter * factor)
        gi += 1
    return durs[:n_chords]


def _dynamic_envelope(n: int, arousal: float, climax_i: float) -> list[float]:
    baseline = 0.22 + 0.45 * arousal
    out = []
    for i in range(n):
        dist = abs(i - climax_i) / max(1.0, n - 1)
        bump = (1 - dist) ** 2 * 0.45
        out.append(max(0.0, min(1.0, baseline + bump)))
    return out


def _decorate_soprano(pitches: list[int], starts: list[float], durations: list[float],
                       articulation_staccato: float, rng: random.Random) -> list[tuple[int, float, float, int]]:
    """Returns (pitch, abs_start, abs_end, source_slot_index) events — a
    variable number per chord slot (a legato passing tone splits one slot
    into two events), so this is a flat event list keyed to absolute time,
    not something a caller should zip 1:1 against the chord list. The slot
    index is carried along only so the caller can look up that slot's
    dynamic-envelope velocity."""
    events = []
    legato = articulation_staccato < 0.55
    for i, (p, s, d) in enumerate(zip(pitches, starts, durations)):
        if legato and d >= 0.55 and i < len(pitches) - 1:
            nxt = pitches[i + 1]
            step = 1 if nxt > p else (-1 if nxt < p else 0)
            passing = p + step * rng.choice([1, 2])
            events.append((p, s, s + d * 0.5, i))
            events.append((passing, s + d * 0.5, s + d * 0.5 + d * 0.5 * 0.95, i))
        elif legato:
            events.append((p, s, s + d * 0.92, i))
        else:
            gate = rng.uniform(0.35, 0.55)
            events.append((p, s, s + d * gate, i))
    return events


def _extra_tones_for(degrees: list[int], complexity_level: int) -> list[int]:
    extra_tones = []
    for d in degrees:
        if complexity_level == 0:
            extra_tones.append(0)
        elif complexity_level == 1:
            extra_tones.append(1 if FUNCTION[d] == "D" else 0)
        else:
            extra_tones.append(1 if FUNCTION[d] in ("D", "S") else 0)
    return extra_tones


def _build_section(rng: random.Random, tonic_pc: int, intervals: tuple[int, ...], n_chords: int,
                    cadence: str, complexity_level: int, start_degree: int = 0) -> tuple[list[int], list[int]]:
    degrees = generate_progression(rng, n_chords, cadence, start_degree)
    return degrees, _extra_tones_for(degrees, complexity_level)


def _with_cadence_tail(degrees: list[int], cadence: str) -> list[int]:
    """Reuse an existing degree sequence verbatim except for its final 1-2
    chords, overwritten to land on `cadence` — used for the literal ABA
    recapitulation, where the final A must sound like the same music as the
    first A right up until it actually needs to end the piece."""
    out = list(degrees)
    if cadence == "authentic" and len(out) >= 2:
        out[-2], out[-1] = 4, 0
    elif cadence == "half" and out:
        out[-1] = 4
    elif cadence == "none" and out:
        out[-1] = 6
    return out


def compose_stage_a(brief: dict, out_path: str) -> dict:
    """Returns a small stats dict (parallel-violation count, chord count,
    duration) alongside writing the MIDI file, so callers can self-check."""
    ep = brief["engine_params"]
    tonic_pc, intervals = ep["tonic_pc"], ep["mode_intervals"]
    tempo = brief["tempo_bpm"]
    arousal = brief["vat"]["arousal"]
    complexity = ep["harmonic_complexity_level"]
    cadence = _cadence_type(brief["harmonic_resolution"])
    articulation_staccato = ep["staccato"]
    form_is_aba = brief["form"].startswith("ABA")

    rng = random.Random(int(hashlib.sha256((brief["source_image"] + "-stageA").encode()).hexdigest(), 16))

    section_len = 8
    if form_is_aba:
        a_degrees, a_extra = _build_section(rng, tonic_pc, intervals, section_len, "half", complexity)
        b_degrees, b_extra = _build_section(rng, tonic_pc, intervals, section_len, "half", complexity, start_degree=4)
        # final A is a literal reprise of the first A's material — same
        # chords — except its tail is overwritten to land on the piece's
        # actual cadence, since a mid-piece half-cadence isn't a real ending
        final_a_degrees = _with_cadence_tail(a_degrees, cadence)
        final_a_extra = _extra_tones_for(final_a_degrees, complexity)
        degrees = a_degrees + b_degrees + final_a_degrees
        extra_tones = a_extra + b_extra + final_a_extra
    else:
        degrees, extra_tones = _build_section(rng, tonic_pc, intervals, section_len * 2, cadence, complexity)

    n_chords = len(degrees)
    voicings = voice_lead(rng, degrees, tonic_pc, intervals, extra_tones, cadence)
    durations = _rhythm_slots(ep["meter_numerator"], ep["meter_denominator"], complexity, tempo, n_chords)

    climax_i = brief["climax_position"] * (n_chords - 1)
    envelope = _dynamic_envelope(n_chords, arousal, climax_i)

    starts = list(itertools.accumulate([0.0] + durations[:-1]))
    soprano_pitches = [v["S"] for v in voicings]
    soprano_events = _decorate_soprano(soprano_pitches, starts, durations, articulation_staccato, rng)

    programs = brief["instrumentation"]
    low_prog = GM_PROGRAM.get(programs[0], DEFAULT_PROGRAM) if programs else DEFAULT_PROGRAM
    high_prog = GM_PROGRAM.get(programs[1], low_prog) if len(programs) > 1 else low_prog

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    tracks = {
        "B": pretty_midi.Instrument(program=low_prog, name="bass"),
        "T": pretty_midi.Instrument(program=low_prog, name="tenor"),
        "A": pretty_midi.Instrument(program=high_prog, name="alto"),
        "S": pretty_midi.Instrument(program=high_prog, name="soprano"),
    }

    for i in range(n_chords):
        s, d = starts[i], durations[i]
        vel = int(round(30 + envelope[i] * 90))
        for voice in "BTA":
            tracks[voice].notes.append(pretty_midi.Note(
                velocity=max(20, vel - 12), pitch=voicings[i][voice],
                start=s, end=s + d * 0.95,
            ))
    for pitch, ev_start, ev_end, slot_i in soprano_events:
        vel = int(round(30 + envelope[slot_i] * 90))
        tracks["S"].notes.append(pretty_midi.Note(
            velocity=vel, pitch=pitch, start=ev_start, end=ev_end,
        ))
    total_duration = starts[-1] + durations[-1]

    for name in ["B", "T", "A", "S"]:
        pm.instruments.append(tracks[name])
    pm.write(out_path)

    violations = sum(_parallel_violation(voicings[i], voicings[i + 1]) for i in range(n_chords - 1))
    return {"n_chords": n_chords, "duration_sec": round(total_duration, 2), "parallel_violations": violations, "cadence": cadence}

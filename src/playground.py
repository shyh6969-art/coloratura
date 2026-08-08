"""
Coloratura — manual VAT playground.

Idea #1 of a second "5 substantial upgrades" round the user asked for: a
control panel where valence/arousal/tension_formal/tension_emotional are
set directly by hand (plus a style choice and a major/minor toggle), and
the SAME rendering code a real upload goes through produces a real
painting or a real piece of music from those numbers — no audio or image
ever analyzed. Turns the engine into an instrument you can play, not just
a black box you feed files into.

mapping_engine.build_brief_from_vat() and visual_mapping_engine.
build_visual_brief_from_vat() already do "vat + style -> full brief" as a
standalone step (refactored out of build_brief()/build_visual_brief() for
exactly this reuse — the real upload path is unchanged, just now calls the
same shared second half). This module only has to supply the handful of
non-VAT fields those two functions still need, using honest, documented
approximations derived from the same sliders rather than a second set of
sliders for every underlying raw feature — matches the spirit of
mapping_engine.py's own docstring (no naive 1:1 tables) applied to a
different problem: approximating "what would a real analysis have measured"
well enough to get a real, differentiated render, not read literally as a
prediction of real pixel/audio statistics.
"""

from __future__ import annotations

from mapping_engine import build_brief_from_vat
from visual_mapping_engine import build_visual_brief_from_vat

STYLE_IDIOMS = [
    "אימפרסיוניזם",
    "אקספרסיוניזם",
    "קוביזם / אבסטרקט-גאומטרי",
    "מינימליזם",
    "ריאליזם",
    "סוריאליזם",
    "אבסטרקט-גסטורלי",
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _synth_dynamic_curve(arousal: float, n: int = 6) -> list[float]:
    """A plausible energy-over-time arc for a piece that was never actually
    played — same "baseline + bump at the midpoint" shape mapping_engine.
    _dynamic_curve() uses for a real climax position, just centered since
    there's no real audio to find a climax in."""
    baseline = 0.22 + 0.45 * arousal
    mid = (n - 1) / 2
    return [round(_clip01(baseline + (1 - abs(i - mid) / mid) ** 2 * 0.45), 2) for i in range(n)]


def _vat_from_sliders(valence: float, arousal: float, tension_formal: float, tension_emotional: float) -> dict:
    return {
        "valence": round(_clip01(valence), 2),
        "arousal": round(_clip01(arousal), 2),
        "tension_formal": round(_clip01(tension_formal), 2),
        "tension_emotional": round(_clip01(tension_emotional), 2),
    }


def painting_from_sliders(valence: float, arousal: float, tension_formal: float,
                           tension_emotional: float, style: str, mode_major: bool) -> dict:
    """Music -> image direction: produces a visual_brief exactly like a
    real audio upload would, from sliders instead of a real track.
    loudness_rms/rhythmic_density default to arousal itself (each is
    literally one of signal-side arousal's own biggest weighted inputs in
    visual_mapping_engine.compute_signal_vat, so reusing arousal directly
    is a defensible stand-in, not an arbitrary guess); noisiness defaults
    to tension_formal for the same reason (its biggest weighted input
    there too); spectral brightness defaults to valence (valence's own
    biggest weighted input)."""
    if style not in STYLE_IDIOMS:
        raise ValueError(f"unknown style: {style}")
    vat = _vat_from_sliders(valence, arousal, tension_formal, tension_emotional)
    audio_extra = {
        "mode_major": bool(mode_major),
        "loudness_rms": vat["arousal"],
        "rhythmic_density": vat["arousal"],
        "noisiness": vat["tension_formal"],
        "brightness": vat["valence"],
        "dynamic_curve": _synth_dynamic_curve(vat["arousal"]),
    }
    engine_notes = {
        "source": "פינת המשחק — הוזן ידנית, לא נותח קובץ אמיתי",
        "mode_major": audio_extra["mode_major"],
    }
    return build_visual_brief_from_vat(
        "playground", vat, style, audio_extra,
        engine_notes_extra=engine_notes,
        raw_audio_features=None,
    )


def music_from_sliders(valence: float, arousal: float, tension_formal: float,
                        tension_emotional: float, style: str, mode_major: bool, seed: str) -> dict:
    """Image -> music direction: produces a musical brief exactly like a
    real image upload would, from sliders instead of a real painting.
    symmetry/hue_variety/color_temperature reuse visual_mapping_engine's
    OWN vat->target_features formulas for those same three quantities
    (build_visual_brief_from_vat computes them identically from vat alone
    on the other side of the pipeline) rather than inventing a second,
    inconsistent mapping — the two directions agree on what a given vat
    'looks like' by construction. line_angularity defaults to
    tension_formal directly (it IS the formal-tension axis's own visual
    reading). focal_point.x stays centered — there's no real dynamic
    curve to find a climax in.

    seed stands in for the file path build_brief_from_vat() hashes for a
    stable tonic — sliders have no file, so the caller passes a value that
    only changes when the user asks for a fresh tonic (see webapp.py)."""
    if style not in STYLE_IDIOMS:
        raise ValueError(f"unknown style: {style}")
    vat = _vat_from_sliders(valence, arousal, tension_formal, tension_emotional)
    pixel_extra = {
        "symmetry": round(_clip01(1 - vat["tension_formal"]), 3),
        "hue_variety": round(_clip01(0.15 + 0.75 * vat["tension_formal"]), 3),
        "color_temperature": round(_clip01(0.5 * vat["valence"] + 0.35 * (1 if mode_major else 0) + 0.15 * (1 - vat["tension_formal"])), 3),
        "brightness": round(_clip01(0.20 + 0.80 * vat["valence"]), 3),
        "line_angularity": vat["tension_formal"],
        "focal_point": {"x": 0.5, "y": 0.5},
    }
    engine_notes = {
        "source": "פינת המשחק — הוזן ידנית, לא נותח קובץ אמיתי",
        "mode_major": bool(mode_major),
    }
    return build_brief_from_vat(
        f"playground-{seed}", vat, style, pixel_extra,
        engine_notes_extra=engine_notes,
        raw_features=None,
    )

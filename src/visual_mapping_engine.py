"""
Coloratura — visual mapping engine (music -> painting direction).

The mirror of mapping_engine.py: takes audio_features.py's signal-processing
numbers plus audio_semantic.py's CLAP scores, reduces them to the same
four-axis VAT space (valence, arousal, tension_formal, tension_emotional —
see mapping_engine.py's own docstring for why formal/visual tension and
emotional/narrative tension are kept as separate axes), then produces a
structured "visual brief".

Closed-loop vocabulary, on purpose: the brief's target_features dict uses
the EXACT same keys feature_extraction.py measures from a real painting
(brightness, saturation, color_temperature, hue_variety, color_clash,
value_contrast, line_density, line_thickness, line_angularity,
composition_density, symmetry, negative_space_ratio, focal_point,
movement). image_stage_a.py's job is then simply to render a canvas whose
OWN feature_extraction.py reading would land close to these targets — the
two directions of the whole project share one measurement vocabulary
instead of each inventing its own, and a round-trip (painting -> music ->
painting) becomes something that can actually be checked against real
numbers rather than just eyeballed.

Trust in the audio-semantic signal, revised after a real recalibration
pass, not left at its first (too pessimistic) reading: the original version
of this module found CLAP's embeddings nearly collapsed (0.89-0.97 pairwise
similarity) on four of this project's own Stage A renders, and weighted it
down hard as a result. Testing that conclusion against a 34-track sample of
real, diverse, commercially produced music (classical through metal through
hip-hop, gathered via itunes_source.py) told a different story: pairwise
embedding similarity spread 0.115-0.969 (median 0.503) — genuine
discrimination — and the earlier collapse was a property of this project's
own narrow-timbre synthesized audio (synth.py's own documented limitation,
see audio_features.py), not of CLAP itself or of audio_semantic.py. So the
weights below were raised back toward mapping_engine.py's own values
(0.5 / 0.85) rather than left where the Stage-A-only finding first put
them — SEMANTIC_WEIGHT=0.45 and TENSION_EMOTIONAL_SEMANTIC_WEIGHT=0.75,
still a shade more conservative than the image side to leave room for the
one input path that genuinely does still resemble Stage A's narrow timbre
(the live sequencer, synth.py-style oscillators) rather than real recorded
audio.

style_idiom now matches the image side's own preference order for the same
reason: the same 34-track sample showed CLAP's style classification
actually discriminating (5 of 7 buckets used as the top pick, with
sensible correlations — e.g. the three metal tracks all landed on
אקספרסיוניזם/אבסטרקט-גסטורלי, the two most intense/dissonant buckets, not
a coincidence) rather than collapsing onto one answer regardless of input,
which is what the original four-Stage-A-file test had shown. CLAP's
classification is primary again; _signal_style_bucket (the audio-side
sibling of feature_extraction.py's own crude style_bucket() placeholder)
is demoted to runner-up context in engine_notes, mirroring
mapping_engine.py exactly — because the evidence now supports it, not
because symmetry is inherently more elegant than asymmetry.

Same honesty carried over from mapping_engine.py's own section א: no pitch
class is mapped to a hue. hue_deg (the one field beyond feature_extraction's
own vocabulary, needed because image_stage_a.py has to pick an actual base
color, not just a 0-1 warmth scalar) is driven by valence, not by the
detected key/tonic — the same naive color<->note table the spec doc
rejects one way is just as naive rejected the other way.
"""

from __future__ import annotations

import hashlib
from typing import Any

SEMANTIC_WEIGHT = 0.45
TENSION_EMOTIONAL_SEMANTIC_WEIGHT = 0.75


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _dynamic_curve_volatility(curve: list[float]) -> float:
    """Std of consecutive differences, normalized against a typical observed
    swing (~0.4) — the audio-side analog of the pixel side's value_contrast:
    how much the piece's energy jumps around rather than sitting still."""
    if len(curve) < 2:
        return 0.0
    diffs = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    std = var ** 0.5
    return _clip01(std / 0.4)


def compute_signal_vat(audio_feats: dict) -> dict:
    """The four axes from audio_features.py's numbers alone — no CLAP
    involved. Kept separate from compute_vat() below for the same reason
    compute_pixel_vat() is kept separate on the image side: so the two
    signals stay independently inspectable (see engine_notes)."""
    mode_major = 1.0 if audio_feats["key"]["mode"] == "major" else 0.0
    tempo_norm = _clip01((audio_feats["tempo_bpm"] - 50) / 150)
    dyn_volatility = _dynamic_curve_volatility(audio_feats["dynamic_curve"])

    valence = (
        0.40 * audio_feats["brightness"]
        + 0.35 * mode_major
        + 0.25 * (1 - audio_feats["noisiness"])
    )
    arousal = (
        0.35 * audio_feats["rhythmic_density"]
        + 0.30 * tempo_norm
        + 0.20 * audio_feats["loudness_rms"]
        + 0.15 * audio_feats["texture_richness"]
    )
    tension_formal = (
        0.40 * audio_feats["noisiness"]
        + 0.35 * audio_feats["texture_richness"]
        + 0.25 * dyn_volatility
    )
    # fallback-only proxy for tension_emotional, used when no semantic score
    # is available — deliberately different features than tension_formal
    # (mode darkness rather than roughness/texture), mirroring how
    # mapping_engine.py's own pixel fallback for emotional tension draws on
    # different features than its formal-tension formula does.
    tension_emotional = _clip01(0.5 * audio_feats["noisiness"] + 0.5 * (1 - mode_major))

    return {
        "valence": _clip01(valence),
        "arousal": _clip01(arousal),
        "tension_formal": _clip01(tension_formal),
        "tension_emotional": tension_emotional,
    }


def compute_vat(audio_feats: dict, semantic: dict | None = None) -> dict:
    """Blend of the signal-processing axes and the CLAP semantic axes. See
    module docstring for why the semantic weights here are well below
    mapping_engine.py's — tension_formal stays signal-only regardless, for
    the same reason it does on the image side: it's a property of the
    sound's own structure/roughness, and blending in CLAP's 'ominous' score
    would reintroduce the exact formal/emotional conflation the split
    exists to prevent."""
    signal = compute_signal_vat(audio_feats)
    if semantic is None:
        return {k: round(v, 2) for k, v in signal.items()}
    w = SEMANTIC_WEIGHT
    w2 = TENSION_EMOTIONAL_SEMANTIC_WEIGHT
    blended = {
        "valence": _clip01((1 - w) * signal["valence"] + w * semantic["valence"]),
        "arousal": _clip01((1 - w) * signal["arousal"] + w * semantic["arousal"]),
        "tension_formal": signal["tension_formal"],
        "tension_emotional": _clip01((1 - w2) * signal["tension_emotional"] + w2 * semantic["tension"]),
    }
    return {k: round(v, 2) for k, v in blended.items()}


def _signal_style_bucket(audio_feats: dict) -> dict:
    """Crude, audio-only style heuristic — the sibling of
    feature_extraction.py's own style_bucket() placeholder, same status
    (intentionally rough, meant to be replaced by something better, not to
    be right). Six buckets, not seven: feature_extraction.py's pixel
    heuristic never included סוריאליזם either, for the same underlying
    reason — dreamlike/uncanny content is a semantic judgment, not
    something low-level statistics can approximate, on either side of the
    pipeline."""
    mode_major = 1.0 if audio_feats["key"]["mode"] == "major" else 0.0
    rd = audio_feats["rhythmic_density"]
    tr = audio_feats["texture_richness"]
    ns = audio_feats["noisiness"]
    dyn_vol = _dynamic_curve_volatility(audio_feats["dynamic_curve"])
    tension_formal = _clip01(0.40 * ns + 0.35 * tr + 0.25 * dyn_vol)

    scores = {
        "מינימליזם": (1 - rd) * 1.0 + (1 - tr) * 0.6 + (1 - ns) * 0.4,
        "קוביזם / אבסטרקט-גאומטרי": dyn_vol * 1.1 + (1 - ns) * 0.3,
        "אקספרסיוניזם": ns * 1.0 + tension_formal * 0.9 + tr * 0.6,
        "אימפרסיוניזם": (1 - ns) * 0.8 + tr * 0.7 + (1 - rd) * 0.6,
        "אבסטרקט-גסטורלי": rd * 1.0 + dyn_vol * 0.8 + tr * 0.5,
        "ריאליזם": (1 - tension_formal) * 0.6 + (1 - ns) * 0.4 + (1 - abs(rd - 0.5) * 2) * 0.4,
    }
    best = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {"bucket": best, "runner_up": ranked[1][0], "scores": {k: round(v, 2) for k, v in scores.items()}}


# Musical-idiom -> visual mark-making, keyed by the same seven style-bucket
# names as semantic_features.STYLE_PROMPTS / audio_semantic.STYLE_PROMPTS,
# so a style bucket landed on from either direction means the same thing.
MARK_MAKING = {
    "אימפרסיוניזם": {"description": "מכחול רך, נקודות צבע מעורבבות ואור מפוזר, ללא קווי מתאר חדים", "brush_type_id": "soft_dab"},
    "אקספרסיוניזם": {"description": "משיכות מכחול גסות ומעוותות, קווי מתאר שחורים חדים וצבע לא-נטורליסטי", "brush_type_id": "jagged"},
    "קוביזם / אבסטרקט-גאומטרי": {"description": "צורות גאומטריות מפוצלות וזוויתיות, ריבוי נקודות מבט", "brush_type_id": "fragmented_geometric"},
    "מינימליזם": {"description": "שדות צבע שטוחים ואחידים, מעט מאוד אלמנטים, המון שטח ריק", "brush_type_id": "flat_field"},
    "ריאליזם": {"description": "מעברי גוון חלקים, צורות מוגדרות וקומפוזיציה מאוזנת", "brush_type_id": "smooth_realistic"},
    "סוריאליזם": {"description": "צורות חלומיות ומעורפלות, גבולות נמסים, שילובים בלתי-אפשריים", "brush_type_id": "dreamlike_blend"},
    "אבסטרקט-גסטורלי": {"description": "משיכות מכחול רחבות וספונטניות, ללא מוקד קומפוזיציוני יחיד", "brush_type_id": "gestural_sweep"},
}


def _hue_deg(valence: float, arousal: float) -> int:
    """Warm/cool base ramp from blue (valence=0) through magenta to
    red/orange (valence=1), with arousal rotating on top — not a pitch-
    class lookup (see module docstring).

    The arousal term is a second real-data fix, not present in the first
    version: found directly by testing, not assumed — six deliberately
    different genres (classical, metal, hip-hop, electronic, folk, reggae)
    from the 34-track calibration sample rendered hue_deg values of
    327/327/343/344/334/359, a ~30-degree band, despite sounding nothing
    alike. Root cause: valence-alone drove hue, and this project's own
    blended valence (0.55 signal / 0.45 CLAP) clusters in a much narrower
    real-world range (~0.5-0.75 across that six-genre sample) than arousal
    does (0.06-0.99, the same six songs) — real music is mostly not read
    as strongly sad or strongly joyful by either signal, but varies hugely
    in energy. A user-reported symptom of exactly this ("all the songs
    turn into the same type of painting, even very different ones") led
    directly to finding it. Arousal now rotates the base hue up to +-70
    degrees — high-arousal tracks push toward more vivid/alarm hues, low-
    arousal toward calmer ones — separating e.g. aggressive-high-arousal
    from mellow-high-arousal music that valence alone reads as similar.
    An 8-song retest (same six genres plus ambient and a second metal
    track) after this change spread hue_deg across 333 degrees instead of
    ~30, confirmed visually as genuinely different paintings (color
    palette and composition), not just a wider number range. A first pass
    at +-35 degrees (rotation coefficient 70) only widened the spread to
    ~39 degrees and was judged insufficient before this larger coefficient
    was tried. Still a two-axis approximation tuned against one sample,
    not a general solution; the same "revisit with a larger, more varied
    sample" caveat as the CV/audio calibration constants elsewhere still
    applies."""
    base = 200 + (valence ** 0.6) * 190
    rotation = (arousal - 0.5) * 140
    return round((base + rotation) % 360)


def _movement(dynamic_curve: list[float]) -> dict:
    trend = dynamic_curve[-1] - dynamic_curve[0]
    if trend > 0.08:
        label, angle = "אלכסוני עולה", 45.0
    elif trend < -0.08:
        label, angle = "אלכסוני יורד", 225.0
    else:
        label, angle = "אופקי", 90.0
    dynamism = _clip01(abs(trend) / 0.5)
    return {"label": label, "dominant_angle_deg": angle, "dynamism": dynamism}


def _focal_point(dynamic_curve: list[float], brightness_raw: float) -> dict:
    """x from the dynamic curve's climax (loudest/busiest moment maps to
    where the eye should land horizontally, same role focal_point.x plays
    in mapping_engine.py's own climax_position). y from spectral brightness
    via the pitch-height cross-modal correspondence (bright/high timbre ->
    upper canvas) — one of the more robustly replicated cross-modal effects
    in the perception literature (Spence, 'Crossmodal Correspondences',
    2011), not an invented rule."""
    climax_i = max(range(len(dynamic_curve)), key=lambda i: dynamic_curve[i])
    x = climax_i / max(1, len(dynamic_curve) - 1)
    y = _clip01(1 - brightness_raw)
    return {"x": round(x, 2), "y": round(y, 2)}


def _stable_seed(path: str) -> int:
    """Stable hash of the source path so re-running the same audio file
    reproduces the same procedural render — a determinism convenience for
    image_stage_a.py's RNG, not the tonic-hash's naive-mapping-avoidance
    principle (there's no equivalent naive trap in picking an RNG seed)."""
    h = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_visual_brief_from_vat(
    path: str,
    vat: dict,
    style: str,
    audio_extra: dict,
    engine_notes_extra: dict | None = None,
    raw_audio_features: dict | None = None,
) -> dict:
    """The shared second half of build_visual_brief(): everything downstream
    of already having a vat dict and a style bucket. Factored out so the
    manual VAT playground (webapp.py's /api/playground_image) can produce a
    real, fully-formed brief — and a real rendered painting — from user-set
    sliders, no audio ever analyzed, using the exact same target_features
    formulas a real upload goes through.

    audio_extra carries the handful of raw audio_feats fields these formulas
    need beyond vat+style: mode_major (bool), loudness_rms, rhythmic_density,
    noisiness, brightness (spectral), dynamic_curve. build_visual_brief()
    below passes the real measured values; the playground route passes
    vat-derived approximations (see webapp.py), documented there as
    playground simplifications, not a real audio analysis."""
    mark = MARK_MAKING[style]

    brightness_raw = audio_extra["brightness"]
    composition_density_t = vat["arousal"]
    line_angularity_t = _clip01(0.5 * vat["tension_formal"] + 0.5 * audio_extra["noisiness"])

    target_features: dict[str, Any] = {
        "brightness": round(_clip01(0.20 + 0.55 * vat["valence"] + 0.25 * brightness_raw), 3),
        "saturation": round(_clip01(0.25 + 0.65 * vat["arousal"]), 3),
        "color_temperature": round(_clip01(0.5 * vat["valence"] + 0.35 * (1 if audio_extra["mode_major"] else 0) + 0.15 * (1 - vat["tension_formal"])), 3),
        "hue_variety": round(_clip01(0.15 + 0.75 * vat["tension_formal"]), 3),
        "color_clash": round(vat["tension_formal"], 3),
        "value_contrast": round(_clip01(0.25 + 0.65 * vat["tension_formal"] + 0.10 * audio_extra["loudness_rms"]), 3),
        "line_density": round(_clip01(0.20 + 0.65 * audio_extra["rhythmic_density"]), 3),
        "line_thickness": round(_clip01(1 - audio_extra["rhythmic_density"]), 3),
        "line_angularity": round(line_angularity_t, 3),
        "composition_density": round(composition_density_t, 3),
        "symmetry": round(_clip01(1 - vat["tension_formal"]), 3),
        "negative_space_ratio": round(_clip01(1 - composition_density_t), 3),
        "focal_point": _focal_point(audio_extra["dynamic_curve"], brightness_raw),
        "movement": _movement(audio_extra["dynamic_curve"]),
    }

    brief: dict[str, Any] = {
        "source_audio": path,
        "vat": vat,
        "style_idiom": style,
        "target_features": target_features,
        "mark_making": mark,
        "engine_notes": engine_notes_extra or {},
        "raw_audio_features": raw_audio_features,
        "engine_params": {
            "hue_deg": _hue_deg(vat["valence"], vat["arousal"]),
            "brush_type_id": mark["brush_type_id"],
            "seed": _stable_seed(path),
        },
    }
    return brief


def build_visual_brief(path: str, audio_feats: dict, semantic: dict | None = None) -> dict:
    vat = compute_vat(audio_feats, semantic)
    # style_idiom: CLAP is primary again (mirrors mapping_engine.py), with
    # the signal-only heuristic as runner-up context — see module docstring
    # for the 34-track calibration finding that restored this preference.
    style_signal = _signal_style_bucket(audio_feats)
    style = semantic["style_bucket"] if semantic else style_signal["bucket"]
    audio_extra = {
        "mode_major": audio_feats["key"]["mode"] == "major",
        "loudness_rms": audio_feats["loudness_rms"],
        "rhythmic_density": audio_feats["rhythmic_density"],
        "noisiness": audio_feats["noisiness"],
        "brightness": audio_feats["brightness"],
        "dynamic_curve": audio_feats["dynamic_curve"],
    }
    engine_notes = {
        "semantic_weight_valence_arousal": SEMANTIC_WEIGHT,
        "semantic_weight_tension_emotional": TENSION_EMOTIONAL_SEMANTIC_WEIGHT,
        "weights_calibrated_against": "34-track real-music sample (output/audio_reference_large) — see module docstring",
        "style_source": "CLAP (audio_semantic.py)" if semantic else "signal heuristic fallback — no semantic available",
        "style_signal_bucket_context": style_signal["bucket"],
        "style_signal_runner_up_context": style_signal["runner_up"],
        "style_signal_scores_context": style_signal["scores"],
        "style_semantic_scores": semantic["style_scores"] if semantic else None,
        "vat_signal_only": {k: round(v, 2) for k, v in compute_signal_vat(audio_feats).items()},
        "vat_semantic_only": semantic and {
            "valence": semantic["valence"],
            "arousal": semantic["arousal"],
            "tension_emotional": semantic["tension"],
        },
    }
    return build_visual_brief_from_vat(
        path, vat, style, audio_extra,
        engine_notes_extra=engine_notes,
        raw_audio_features={k: v for k, v in audio_feats.items() if k != "raw"},
    )

"""
Coloratura — mapping engine.

Takes the raw visual features from feature_extraction.py plus the CLIP
semantic scores from semantic_features.py, reduces them to a mediating
emotional space, then applies the detailed parameter table (spec doc,
section ה) to produce a structured "musical brief" (section ח).

The mediating space is VA-FE, not the VAT from the original spec doc:
valence, arousal, tension-formal, tension-emotional. The split happened
after the 5-painting sanity check (output/report.txt) surfaced Mondrian:
its pixel-measured color-clash tension was high (0.72) while its CLIP
"does this feel ominous" tension was near zero (0.065) — both readings are
correct, they're just answering different questions. A grid of clashing
primary colors is formally/visually tense (the eye vibrates at the
boundaries) without being remotely emotionally threatening. One scalar
"tension" axis was silently averaging two different things:

  tension_formal    — visual/color clash, asymmetry, harsh geometry.
                       Pixel-only; CLIP doesn't add much here, this is
                       exactly what edge/color statistics are good at.
                       Feeds harmonic complexity and metric irregularity.
  tension_emotional  — is the mood ominous, threatening, unresolved, vs.
                        safe and settled. This is the axis the semantic
                        layer exists for — pixel stats structurally can't
                        read "ominous." Feeds mode darkness and whether the
                        piece resolves harmonically.

Deliberate honesty, matching section א of the spec: nothing here maps hue
directly to a pitch letter. The brief's tonic is chosen from a stable hash
of the file, not from color — a literal color->note table is exactly the
naive approach the spec doc rejects.
"""

from __future__ import annotations

import hashlib
from typing import Any

TONICS = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# Interval sets (semitones from tonic) for each mode label _mode() can return.
# Composer.py consumes these directly rather than re-deriving them from the
# Hebrew label, so the sound and the doc's mode names can never drift apart.
MODE_INTERVALS = {
    "מאז'ורי (יוני) / לידי": (0, 2, 4, 5, 7, 9, 11),
    "מיקסולידי": (0, 2, 4, 5, 7, 9, 10),
    "אאולי (מינורי טבעי)": (0, 2, 3, 5, 7, 8, 10),
    "פריגי": (0, 1, 3, 5, 7, 8, 10),
    "לוקרי (הכהה והלא-יציב ביותר)": (0, 1, 3, 5, 6, 8, 10),
}

# Octave number (MIDI convention, C4=60) each register label anchors to.
REGISTER_OCTAVE = {"נמוך": 3, "בינוני-נמוך": 4, "בינוני": 4, "בינוני-גבוה": 5, "גבוה": 5}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _emotional_tension_pixel_proxy(feats: dict) -> float:
    """Fallback used only when no semantic score is available. Angularity
    and light/dark contrast carry a little emotional-unease signal on their
    own (jagged harsh shapes, stark contrast), but this is a much weaker
    proxy than CLIP reading actual pictorial content — see module docstring."""
    return _clip01(0.55 * feats["line_angularity"] + 0.45 * feats["value_contrast"])


def compute_pixel_vat(feats: dict) -> dict:
    """The four axes from pixel statistics alone. Kept separate from the
    blended compute_vat() below so the two signals stay inspectable
    independently — see engine_notes in the brief."""
    valence = (
        0.40 * feats["brightness"]
        + 0.25 * feats["color_temperature"]
        + 0.35 * (1 - feats["color_clash"])
    )
    arousal = (
        0.35 * feats["composition_density"]
        + 0.25 * feats["line_density"]
        + 0.15 * feats["movement"]["dynamism"]
        + 0.25 * feats["saturation"]
    )
    tension_formal = (
        0.35 * feats["color_clash"]
        + 0.30 * feats["value_contrast"]
        + 0.20 * (1 - feats["symmetry"])
        + 0.15 * feats["line_angularity"]
    )
    tension_emotional = _emotional_tension_pixel_proxy(feats)
    return {
        "valence": _clip01(valence),
        "arousal": _clip01(arousal),
        "tension_formal": _clip01(tension_formal),
        "tension_emotional": tension_emotional,
    }


def compute_vat(feats: dict, semantic: dict | None = None, semantic_weight: float = 0.5) -> dict:
    """Blend of the pixel-statistics axes and the CLIP semantic axes
    (semantic_features.semantic_scores). semantic_weight=0.5 is a
    deliberately plain, unturned default for valence/arousal — not fit to
    the 5-painting test set.

    tension_formal is pixel-only regardless of semantic_weight: it's a
    geometric/color property, and blending in the semantic 'ominous' score
    here would just re-introduce the exact conflation this split exists to
    remove.

    tension_emotional leans much harder on the semantic score (0.85) than
    the other axes do, because that prompt pair is precisely the thing
    pixel statistics cannot see — the pixel proxy is kept in at low weight
    only so the axis degrades gracefully when semantic is unavailable
    rather than becoming a discontinuous special case.
    """
    pixel = compute_pixel_vat(feats)
    if semantic is None:
        return {k: round(v, 2) for k, v in pixel.items()}
    w = semantic_weight
    blended = {
        "valence": _clip01((1 - w) * pixel["valence"] + w * semantic["valence"]),
        "arousal": _clip01((1 - w) * pixel["arousal"] + w * semantic["arousal"]),
        "tension_formal": pixel["tension_formal"],
        "tension_emotional": _clip01(0.15 * pixel["tension_emotional"] + 0.85 * semantic["tension"]),
    }
    return {k: round(v, 2) for k, v in blended.items()}


def _mode(valence: float, tension_emotional: float) -> str:
    """Mode darkness now driven by emotional tension (is this ominous?),
    not formal tension (is the palette clashing?) — a clashing-but-cheerful
    Kandinsky and a blended-but-dreadful Munch sky should not pick the same
    kind of darkness for the same reason."""
    if valence >= 0.62:
        return "מאז'ורי (יוני) / לידי"
    if valence >= 0.45:
        return "מיקסולידי"
    if valence >= 0.30:
        return "אאולי (מינורי טבעי)"
    if tension_emotional >= 0.55:
        return "לוקרי (הכהה והלא-יציב ביותר)"
    return "פריגי"


def _register(brightness: float) -> str:
    if brightness >= 0.66:
        return "גבוה"
    if brightness >= 0.50:
        return "בינוני-גבוה"
    if brightness >= 0.35:
        return "בינוני"
    if brightness >= 0.20:
        return "בינוני-נמוך"
    return "נמוך"


def _instrumentation(temp: float, style: str) -> list[str]:
    base = ["מיתרים", "קרן יער", "פסנתר"] if temp >= 0.5 else ["פעמוני צינור", "קלרינט", "פאד מיתרים גבוה"]
    flavor = {
        "אימפרסיוניזם": ["נבל", "חלילית"],
        "אקספרסיוניזם": ["נחושת מעוותת (con sordino)", "פרקאשן לא-מכוון"],
        "מינימליזם": ["מרימבה פעימתית", "פסנתר פרפטואום מוביל"],
        "קוביזם / אבסטרקט-גאומטרי": ["כלי נשיפה-עץ", "קונטרבס פיציקטו"],
        "אבסטרקט-גסטורלי": ["תופים חופשיים", "סקסופון אלט"],
        "ריאליזם": ["קוורטט כלי-קשת"],
    }.get(style, [])
    return base + flavor


def _articulation(angularity: float) -> str:
    if angularity >= 0.6:
        return "סטקאטו דומיננטי, מבטאים חדים ומרקאטו"
    if angularity <= 0.3:
        return "לגאטו דומיננטי, קשירות ארוכות"
    return "לגאטו בבסיס, סטקאטו נקודתי בשיאים"


def _meter(style: str, arousal: float, tension_formal: float) -> str:
    """Metric irregularity from *formal* tension — visual clash/asymmetry
    reads as rhythmic dislocation (Stravinsky-esque), independent of mood."""
    if style == "קוביזם / אבסטרקט-גאומטרי" and arousal >= 0.45:
        return "7/8 (מטר א-סימטרי, השראת סטרווינסקי)"
    if tension_formal >= 0.65 and arousal >= 0.55:
        return "5/4"
    return "4/4"


def _harmonic_complexity(hue_variety: float, tension_formal: float) -> str:
    """Chromatic complexity from palette variety + *formal* clash — a busy,
    color-clashing canvas implies a chromatically busy harmony regardless of
    whether the mood itself is threatening."""
    score = 0.5 * hue_variety + 0.5 * tension_formal
    if score < 0.32:
        return "פשוט / דיאטוני"
    if score < 0.6:
        return "בינוני / כרומטיות מקומית"
    return "מורכב / כרומטי-מודולטיבי"


def _harmonic_resolution(tension_emotional: float) -> str:
    """New field: whether the piece resolves. This is squarely an
    emotional-tension question — a dissonant-but-not-ominous painting
    (Kandinsky) can still land on a clear cadence; an ominous one shouldn't."""
    if tension_emotional >= 0.6:
        return "לא-פתור — מסתיים על הרמוניה דיסוננטית/מוקטנת, ללא קדנצה סופית ברורה"
    if tension_emotional >= 0.35:
        return "פתור חלקית — קדנצה מרומזת אך לא חד-משמעית"
    return "פתור — קדנצה סופית ברורה וסגירה הרמונית מלאה"


def _form(symmetry: float) -> str:
    if symmetry >= 0.55:
        return "ABA מאוזנת, קדנצות ברורות"
    return "דרך-מולחן (through-composed), אורכי משפט לא סדירים"


def _dynamic_curve(arousal: float, climax_x: float, n: int = 6) -> list[float]:
    baseline = 0.22 + 0.45 * arousal
    climax_i = climax_x * (n - 1)
    curve = []
    for i in range(n):
        dist = abs(i - climax_i) / max(1.0, n - 1)
        bump = (1 - dist) ** 2 * 0.45
        curve.append(round(_clip01(baseline + bump), 2))
    return curve


def _tonic(path: str) -> str:
    h = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return TONICS[int(h, 16) % len(TONICS)]


def _parse_meter(meter_label: str) -> tuple[int, int]:
    num, _, rest = meter_label.partition("/")
    den = rest.split(" ", 1)[0]
    return int(num), int(den)


def _complexity_level(harmonic_complexity_label: str) -> int:
    """0/1/2 read back off the Hebrew label _harmonic_complexity() produced,
    so the string and the number can't drift out of sync with each other."""
    if harmonic_complexity_label.startswith("פשוט"):
        return 0
    if harmonic_complexity_label.startswith("בינוני"):
        return 1
    return 2


def _is_resolved(harmonic_resolution_label: str) -> bool:
    return not harmonic_resolution_label.startswith("לא-פתור")


def _engine_params(brief_partial: dict, feats: dict) -> dict:
    """Everything composer.py needs as plain numbers/ids — no Hebrew string
    parsing downstream. Derived from the same values already used to build
    the human-readable fields above, not recomputed independently."""
    tonic_name = brief_partial["_tonic_name"]
    mode_label = brief_partial["_mode_label"]
    num, den = _parse_meter(brief_partial["meter"])
    return {
        "tonic_pc": NOTE_TO_PC[tonic_name],
        "tonic_name": tonic_name,
        "mode_id": mode_label,
        "mode_intervals": MODE_INTERVALS[mode_label],
        "register_octave": REGISTER_OCTAVE[brief_partial["register_range"]],
        "meter_numerator": num,
        "meter_denominator": den,
        "harmonic_complexity_level": _complexity_level(brief_partial["harmonic_complexity"]),
        "resolved": _is_resolved(brief_partial["harmonic_resolution"]),
        "staccato": feats["line_angularity"],
    }


def build_brief_from_vat(
    path: str,
    vat: dict,
    style: str,
    pixel_extra: dict,
    engine_notes_extra: dict | None = None,
    raw_features: dict | None = None,
) -> dict:
    """The shared second half of build_brief(): everything downstream of
    already having a vat dict and a style bucket. Factored out so the
    manual VAT playground (webapp.py's /api/playground_music) can produce
    a real, fully-formed brief from user-set sliders without a real image
    ever being analyzed — same formulas, same code path as a real upload,
    just fed a vat/style/pixel_extra that came from sliders instead of
    feature_extraction.py + semantic_features.py.

    pixel_extra carries the handful of specific pixel-measurement fields
    build_brief's downstream helpers need beyond vat+style: symmetry,
    hue_variety, color_temperature, brightness, line_angularity, and
    focal_point (with an "x" key). build_brief() below passes the real
    measured values; the playground route passes vat-derived approximations
    (see webapp.py) documented there as playground simplifications, not a
    real image analysis."""
    tonic = _tonic(path)
    mode = _mode(vat["valence"], vat["tension_emotional"])

    brief: dict[str, Any] = {
        "source_image": path,
        "vat": vat,
        "key": f"{tonic} — {mode}",
        "tempo_bpm": round(58 + vat["arousal"] * 92),
        "meter": _meter(style, vat["arousal"], vat["tension_formal"]),
        "form": _form(pixel_extra["symmetry"]),
        "harmonic_complexity": _harmonic_complexity(pixel_extra["hue_variety"], vat["tension_formal"]),
        "harmonic_resolution": _harmonic_resolution(vat["tension_emotional"]),
        "instrumentation": _instrumentation(pixel_extra["color_temperature"], style),
        "register_range": _register(pixel_extra["brightness"]),
        "articulation": _articulation(pixel_extra["line_angularity"]),
        "dynamic_curve": _dynamic_curve(vat["arousal"], pixel_extra["focal_point"]["x"]),
        "climax_position": round(pixel_extra["focal_point"]["x"], 2),
        "style_idiom": style,
        "_tonic_name": tonic,
        "_mode_label": mode,
        "engine_notes": engine_notes_extra or {},
        "raw_features": raw_features,
    }
    brief["engine_params"] = _engine_params(brief, {"line_angularity": pixel_extra["line_angularity"]})
    del brief["_tonic_name"], brief["_mode_label"]
    return brief


def build_brief(path: str, feats: dict, semantic: dict | None = None) -> dict:
    vat = compute_vat(feats, semantic)
    # Style: prefer the CLIP zero-shot bucket once available — the pixel
    # heuristic in feature_extraction.style_bucket was explicitly a
    # placeholder for a real classifier (see its docstring). Kept as
    # runner-up context either way for transparency.
    style = semantic["style_bucket"] if semantic else feats["style"]["bucket"]
    pixel_extra = {
        "symmetry": feats["symmetry"],
        "hue_variety": feats["hue_variety"],
        "color_temperature": feats["color_temperature"],
        "brightness": feats["brightness"],
        "line_angularity": feats["line_angularity"],
        "focal_point": feats["focal_point"],
    }
    engine_notes = {
        "tonic_source": "hash יציב של שם הקובץ — לא נגזר מגוון (ר' פרק א של המסמך)",
        "style_pixel_heuristic": feats["style"]["bucket"],
        "style_pixel_runner_up": feats["style"]["runner_up"],
        "style_semantic_scores": semantic["style_scores"] if semantic else None,
        "vat_pixel_only": {k: round(v, 2) for k, v in compute_pixel_vat(feats).items()},
        "vat_semantic_only": semantic and {
            "valence": semantic["valence"],
            "arousal": semantic["arousal"],
            "tension_emotional": semantic["tension"],
        },
    }
    return build_brief_from_vat(
        path, vat, style, pixel_extra,
        engine_notes_extra=engine_notes,
        raw_features={k: v for k, v in feats.items() if k != "style"},
    )

"""
Coloratura — Stage A for the music -> painting direction: a self-built
procedural renderer (PIL + numpy). The visual counterpart to
stage_a.py + synth.py: no external image-generation API here, same
instinct as the rest of the project — build the mark-making from scratch
rather than call out to someone else's model. An AI-quality "Stage B" for
images (an external image-gen API) is planned as a later, separate,
explicitly-costed addition, the same role Suno plays for the forward
direction — not built here.

Takes a visual_mapping_engine.build_visual_brief() brief and paints toward
its target_features using primitives (soft dabs, jagged polygons,
fragmented geometric facets, flat fields, blended gradients, gestural
sweeps) selected by mark_making.brush_type_id — one renderer per style
bucket, in the same spirit as synth.py's per-instrument-family timbre
profiles.

Self-verifying, the same way stage_a.py verifies voice-leading with a real
parallel-5th count rather than trusting the algorithm: after rendering,
feature_extraction.py re-measures the actual PNG and the deltas against
target_features are returned in stats, so "did this land near its targets"
is a number, not a feeling. The brief's target_features use the exact
vocabulary feature_extraction.py measures (see visual_mapping_engine.py's
docstring) precisely so this comparison is possible at all.

Known limitations, found by actually reading that verification output
against 4 test pieces rather than assuming the mapping "should" work:
  - composition_density under-shoots by ~0.13-0.15 even after adding
    _add_canvas_grain(): vector-clean procedural shapes read as far lower
    pixel-level texture (Laplacian variance) than a real painting photo
    until a fair amount of luminance noise is layered on top.
  - symmetry under-shoots, and by more than expected from the mirroring
    fixes alone: the same grain noise that helps composition_density is,
    by construction, uncorrelated between the two mirrored halves, which
    directly works against feature_extraction.symmetry()'s cross-
    correlation measurement. These two targets pull in opposite
    directions through the same mechanism — not a bug to fix so much as a
    real tradeoff between "textured like a painting" and "measurably
    symmetric," worth knowing about rather than silently accepting
    whichever way the constants happen to lean.
  - color_temperature swings from under- to over-shooting depending on the
    piece (see visual_mapping_engine._hue_deg's own docstring) — a
    4-painting calibration, not a general solution.
None of this is hidden in the output: compose_image_stage_a()'s returned
stats always include the actual re-measured values and their deltas.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

CANVAS_SIZE = (900, 700)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hsv01_to_rgb(h_deg: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys
    r, g, b = colorsys.hsv_to_rgb((h_deg % 360) / 360.0, _clip01(s), _clip01(v))
    return (int(r * 255), int(g * 255), int(b * 255))


def _build_palette(brief: dict, rng: random.Random) -> list[tuple[int, int, int]]:
    """Analogous colors clustered around engine_params.hue_deg, spread by
    hue_variety, plus complementary accent colors when color_clash is high
    — mirrors what color_clash/hue_variety actually measure on a real
    painting (feature_extraction.py's color_clash/hue_variety), run in the
    generative direction."""
    tf = brief["target_features"]
    hue_deg = brief["engine_params"]["hue_deg"]
    hue_variety, color_clash = tf["hue_variety"], tf["color_clash"]
    sat, bri = tf["saturation"], tf["brightness"]

    n_base = 3 + round(hue_variety * 5)
    spread = 15 + hue_variety * 130
    colors = []
    for _ in range(n_base):
        h = hue_deg + rng.uniform(-spread, spread)
        s = _clip01(sat + rng.uniform(-0.15, 0.15))
        v = _clip01(bri + rng.uniform(-0.2, 0.25))
        colors.append(_hsv01_to_rgb(h, s, v))

    if color_clash > 0.35:
        n_clash = 1 + round(color_clash * 4)
        for _ in range(n_clash):
            h = hue_deg + 180 + rng.uniform(-20, 20)
            s = _clip01(sat + rng.uniform(0.0, 0.25))
            v = _clip01(bri + rng.uniform(-0.1, 0.3))
            colors.append(_hsv01_to_rgb(h, s, v))
    return colors


def _background_color(brief: dict) -> tuple[int, int, int]:
    tf = brief["target_features"]
    hue_deg = brief["engine_params"]["hue_deg"]
    return _hsv01_to_rgb(hue_deg, tf["saturation"] * 0.35, _clip01(tf["brightness"] * 0.9 + 0.05))


def _sample_position(rng: random.Random, size: tuple[int, int], focal_point: dict, spread: float) -> tuple[float, float]:
    w, h = size
    fx, fy = focal_point["x"] * w, focal_point["y"] * h
    x = rng.gauss(fx, spread * w)
    y = rng.gauss(fy, spread * h)
    return max(0.0, min(w, x)), max(0.0, min(h, y))


# --- drawing primitives, each takes an RGBA ImageDraw layer -----------------

def _draw_soft_dab(layer: ImageDraw.ImageDraw, center, radius, color, alpha) -> None:
    x, y = center
    layer.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, alpha))


def _draw_jagged(layer: ImageDraw.ImageDraw, center, radius, color, alpha, angularity, rng) -> None:
    x, y = center
    n = 5 + int(angularity * 5)
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n + rng.uniform(-0.3, 0.3)
        r = radius * (0.5 + 0.5 * rng.random()) * (0.6 + 0.4 * angularity)
        pts.append((x + r * math.cos(a), y + r * math.sin(a)))
    layer.polygon(pts, fill=(*color, alpha), outline=(20, 20, 20, min(255, alpha + 70)))


def _draw_facet(layer: ImageDraw.ImageDraw, center, radius, color, alpha, rng) -> None:
    x, y = center
    n = rng.choice([3, 4])
    base = rng.uniform(0, 2 * math.pi)
    pts = []
    for i in range(n):
        a = base + 2 * math.pi * i / n + rng.uniform(-0.2, 0.2)
        r = radius * rng.uniform(0.7, 1.15)
        pts.append((x + r * math.cos(a), y + r * math.sin(a)))
    layer.polygon(pts, fill=(*color, alpha), outline=(30, 30, 30, 180))


def _draw_flat_shape(layer: ImageDraw.ImageDraw, center, radius, color, alpha, rng) -> None:
    x, y = center
    rx, ry = radius * rng.uniform(0.8, 1.4), radius * rng.uniform(0.6, 1.3)
    layer.ellipse([x - rx, y - ry, x + rx, y + ry], fill=(*color, alpha))


def _sweep_points(start, angle_deg, length, curviness, rng) -> list[tuple[float, float]]:
    x, y = start
    cur_angle = math.radians(angle_deg)
    pts = [(x, y)]
    steps = 12
    px, py = x, y
    for _ in range(steps):
        cur_angle += rng.uniform(-curviness, curviness)
        px += (length / steps) * math.cos(cur_angle)
        py += (length / steps) * math.sin(cur_angle)
        pts.append((px, py))
    return pts


def _draw_sweep_pts(layer: ImageDraw.ImageDraw, pts, width, color, alpha) -> None:
    layer.line(pts, fill=(*color, alpha), width=max(1, int(width)), joint="curve")


# --- one renderer per style bucket (mark_making.brush_type_id) -------------

def _render_soft_dab(layer, brief, rng, palette, size) -> None:
    """אימפרסיוניזם — many small, soft, semi-transparent dabs; blurred as a
    post-process for the diffused-light look."""
    tf = brief["target_features"]
    n = int(70 + tf["composition_density"] * 420)
    base_r = 6 + (1 - tf["composition_density"]) * 24
    spread = 0.20 + 0.30 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.5, 1.6)
        color, alpha = rng.choice(palette), int(60 + rng.random() * 80)
        _draw_soft_dab(layer, pos, r, color, alpha)
        if rng.random() < tf["symmetry"]:
            _draw_soft_dab(layer, (size[0] - pos[0], pos[1]), r, color, alpha)


def _render_jagged(layer, brief, rng, palette, size) -> None:
    """אקספרסיוניזם — fewer, larger, sharp-edged jagged shapes with dark
    outlines; no blur, kept raw and harsh."""
    tf = brief["target_features"]
    n = int(25 + tf["composition_density"] * 140)
    base_r = 18 + (1 - tf["composition_density"]) * 55
    spread = 0.22 + 0.28 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.6, 1.5)
        color, alpha = rng.choice(palette), int(150 + rng.random() * 105)
        _draw_jagged(layer, pos, r, color, alpha, tf["line_angularity"], rng)
        if rng.random() < tf["symmetry"]:
            _draw_jagged(layer, (size[0] - pos[0], pos[1]), r, color, alpha, tf["line_angularity"], rng)


def _render_facet(layer, brief, rng, palette, size) -> None:
    """קוביזם / אבסטרקט-גאומטרי — flat-shaded triangular/quad facets
    tessellating the canvas, thin dark outlines."""
    tf = brief["target_features"]
    n = int(50 + tf["composition_density"] * 260)
    base_r = 20 + (1 - tf["composition_density"]) * 50
    spread = 0.28 + 0.24 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.6, 1.4)
        color, alpha = rng.choice(palette), int(180 + rng.random() * 75)
        _draw_facet(layer, pos, r, color, alpha, rng)
        if rng.random() < tf["symmetry"]:
            _draw_facet(layer, (size[0] - pos[0], pos[1]), r, color, alpha, rng)


def _render_flat_field(layer, brief, rng, palette, size) -> None:
    """מינימליזם — very few large flat shapes, deliberately overriding
    composition_density downward: mark_making's own description says 'מעט
    מאוד אלמנטים' (very few elements), and that's the one property this
    style is defined by more than anything composition_density alone would
    produce."""
    tf = brief["target_features"]
    n = max(2, int(2 + tf["composition_density"] * 8))
    base_r = 90 + (1 - tf["composition_density"]) * 160
    spread = 0.15 + 0.20 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.7, 1.3)
        color, alpha = rng.choice(palette), 255
        _draw_flat_shape(layer, pos, r, color, alpha, rng)
        if rng.random() < tf["symmetry"]:
            _draw_flat_shape(layer, (size[0] - pos[0], pos[1]), r, color, alpha, rng)


def _render_smooth_realistic(layer, brief, rng, palette, size) -> None:
    """ריאליזם — larger, heavily overlapping soft shapes blended together
    (light blur as post-process) for continuous gradients rather than
    visible discrete marks."""
    tf = brief["target_features"]
    n = int(40 + tf["composition_density"] * 200)
    base_r = 25 + (1 - tf["composition_density"]) * 70
    spread = 0.22 + 0.26 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.6, 1.5)
        color, alpha = rng.choice(palette), int(50 + rng.random() * 60)
        _draw_soft_dab(layer, pos, r, color, alpha)
        if rng.random() < tf["symmetry"]:
            _draw_soft_dab(layer, (size[0] - pos[0], pos[1]), r, color, alpha)


def _render_dreamlike_blend(layer, brief, rng, palette, size) -> None:
    """סוריאליזם — large amorphous blobs with occasional illogical hue
    jumps (ignoring the analogous palette on purpose sometimes), heavy blur
    as post-process for melting/uncanny edges."""
    tf = brief["target_features"]
    n = int(15 + tf["composition_density"] * 70)
    base_r = 60 + (1 - tf["composition_density"]) * 140
    spread = 0.18 + 0.30 * tf["composition_density"]
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], spread)
        r = base_r * rng.uniform(0.7, 1.6)
        if rng.random() < 0.25:
            color = _hsv01_to_rgb(rng.uniform(0, 360), tf["saturation"], tf["brightness"])
        else:
            color = rng.choice(palette)
        alpha = int(70 + rng.random() * 90)
        _draw_soft_dab(layer, pos, r, color, alpha)
        if rng.random() < tf["symmetry"]:
            _draw_soft_dab(layer, (size[0] - pos[0], pos[1]), r, color, alpha)


def _render_gestural_sweep(layer, brief, rng, palette, size) -> None:
    """אבסטרקט-גסטורלי — few, long, sweeping curved strokes following
    movement.dominant_angle_deg, no single focal point dominating."""
    tf = brief["target_features"]
    n = int(20 + tf["composition_density"] * 60)
    angle = tf["movement"]["dominant_angle_deg"]
    curviness = 0.15 + tf["movement"]["dynamism"] * 0.35
    for _ in range(n):
        pos = _sample_position(rng, size, tf["focal_point"], 0.45)
        length = (120 + tf["composition_density"] * 260) * rng.uniform(0.6, 1.4)
        width = 6 + (1 - tf["line_thickness"]) * 26
        color, alpha = rng.choice(palette), int(120 + rng.random() * 110)
        this_angle = angle + rng.uniform(-25, 25)
        pts = _sweep_points(pos, this_angle, length, curviness, rng)
        _draw_sweep_pts(layer, pts, width, color, alpha)
        if rng.random() < tf["symmetry"]:
            # mirror the actual point path (x -> size[0]-x), not a
            # re-randomized stroke with a flipped starting angle — a fresh
            # curviness-perturbed walk doesn't reliably retrace the
            # original's shape even with a mirrored angle, which is what
            # was actually measured as low symmetry before this fix.
            mirrored_pts = [(size[0] - px, py) for px, py in pts]
            _draw_sweep_pts(layer, mirrored_pts, width, color, alpha)


STYLE_RENDERERS = {
    "soft_dab": _render_soft_dab,
    "jagged": _render_jagged,
    "fragmented_geometric": _render_facet,
    "flat_field": _render_flat_field,
    "smooth_realistic": _render_smooth_realistic,
    "dreamlike_blend": _render_dreamlike_blend,
    "gestural_sweep": _render_gestural_sweep,
}

# post-process blur radius (px) per style — the diffused/blended styles get
# real blur, the harsh/flat ones stay crisp on purpose.
STYLE_BLUR = {
    "soft_dab": 2.5,
    "jagged": 0.0,
    "fragmented_geometric": 0.0,
    "flat_field": 0.0,
    "smooth_realistic": 3.5,
    "dreamlike_blend": 9.0,
    "gestural_sweep": 0.8,
}


def _add_canvas_grain(img: Image.Image, target_composition_density: float, seed: int) -> Image.Image:
    """Real photographed paintings carry canvas/brushstroke texture that
    keeps feature_extraction.composition_density (Laplacian-variance-based
    pixel-level 'business') nontrivial even in visually calm passages. This
    module's vector-clean shapes read as near-zero texture regardless of
    target — found by the verification step below, not assumed — so a
    subtle luminance-noise overlay is added, scaled toward the target
    rather than a fixed constant. Same noise value across R/G/B so color
    balance doesn't shift, closer to film/canvas grain than color noise."""
    arr = np.array(img).astype(np.float64)
    rng = np.random.default_rng(seed)
    sigma = 8 + target_composition_density * 28
    noise = rng.normal(0, sigma, size=arr.shape[:2])[:, :, None]
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def compose_image_stage_a(brief: dict, out_path: str, size: tuple[int, int] = CANVAS_SIZE) -> dict:
    tf = brief["target_features"]
    brush_id = brief["engine_params"]["brush_type_id"]
    seed = brief["engine_params"]["seed"]
    rng = random.Random(seed)

    bg = Image.new("RGB", size, _background_color(brief))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    palette = _build_palette(brief, rng)
    renderer = STYLE_RENDERERS[brush_id]
    renderer(draw, brief, rng, palette, size)

    blur = STYLE_BLUR[brush_id]
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))

    img = Image.alpha_composite(bg.convert("RGBA"), layer).convert("RGB")

    # value_contrast / saturation as final global adjustments — the marks
    # above set the *composition*, these two set the overall punch.
    img = ImageEnhance.Contrast(img).enhance(0.75 + tf["value_contrast"] * 0.7)
    img = ImageEnhance.Color(img).enhance(0.6 + tf["saturation"] * 0.8)
    img = _add_canvas_grain(img, tf["composition_density"], seed)

    img.save(out_path)

    stats: dict[str, Any] = {
        "canvas_size": size,
        "style_idiom": brief["style_idiom"],
        "brush_type_id": brush_id,
        "palette_size": len(palette),
        "seed": seed,
    }
    stats["verification"] = _verify_against_targets(out_path, tf)
    return stats


def _verify_against_targets(out_path: str, target_features: dict) -> dict:
    """Re-measures the rendered PNG with feature_extraction.py (the same
    module used on real paintings) and reports the delta against what this
    render was aiming for — a number, not an assumption that the mapping
    'should' have worked. Imported lazily to avoid a hard OpenCV dependency
    for callers that only want to render, not verify."""
    from feature_extraction import extract_features

    actual = extract_features(out_path)
    deltas = {}
    for key in ("brightness", "saturation", "color_temperature", "hue_variety",
                "color_clash", "value_contrast", "composition_density", "symmetry",
                "negative_space_ratio"):
        deltas[key] = round(actual[key] - target_features[key], 3)
    return {"actual": {k: round(actual[k], 3) for k in deltas}, "delta": deltas}

"""
Coloratura — visual feature extraction.

Pulls the raw visual parameters listed in the spec doc (section ה) out of a
painting: color (brightness, saturation, temperature, hue variety), line
quality (thickness, curvature, density), composition (density, symmetry,
negative space, focal contrast), movement direction, and a heuristic style
bucket.

This module only measures. It does not decide what any of it *means*
musically — that translation lives in mapping_engine.py, mediated through
the valence/arousal/tension space described in section ג of the spec.
"""

from __future__ import annotations

import numpy as np
import cv2


def load_image(path: str, max_dim: int = 640) -> np.ndarray:
    """Load as BGR uint8, downscaled so the longer edge is max_dim (keeps CV fast)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


def brightness_value(hsv: np.ndarray) -> float:
    return float(np.mean(hsv[..., 2]) / 255.0)


def saturation(hsv: np.ndarray) -> float:
    return float(np.mean(hsv[..., 1]) / 255.0)


def color_temperature(hsv: np.ndarray) -> float:
    """0 = fully cool, 1 = fully warm. Weighted by saturation*value so near-gray
    pixels (which have an arbitrary hue) don't skew the result."""
    h = hsv[..., 0] * 2.0  # OpenCV hue is 0-179; convert to 0-359 degrees
    weight = (hsv[..., 1] / 255.0) * (hsv[..., 2] / 255.0)
    warm = ((h <= 70) | (h >= 320)).astype(np.float32)  # reds, oranges, yellows
    cool = ((h >= 90) & (h <= 280)).astype(np.float32)  # cyans, blues, violets, greens
    w_sum = float(np.sum(warm * weight))
    c_sum = float(np.sum(cool * weight))
    total = w_sum + c_sum
    if total < 1e-6:
        return 0.5
    return w_sum / total


def value_contrast(hsv: np.ndarray) -> float:
    """0-1 spread of light/dark (std of the V channel, normalized). Captures
    dramatic light-vs-dark tension (e.g. Munch's fiery sky against a near-
    black fjord) that a purely hue-based clash score misses, since near-black
    or near-white pixels carry almost no reliable hue signal."""
    v = hsv[..., 2] / 255.0
    return float(np.clip(np.std(v) / 0.30, 0, 1))


def hue_variety(hsv: np.ndarray, n_bins: int = 24) -> float:
    """Normalized entropy (0-1) of the saturation-weighted hue histogram.
    Low = near-monochrome palette. High = many competing hues."""
    h = hsv[..., 0].astype(np.int32)
    weight = (hsv[..., 1] / 255.0) * (hsv[..., 2] / 255.0)
    hist, _ = np.histogram(h, bins=n_bins, range=(0, 180), weights=weight)
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(entropy / np.log2(n_bins))


def color_clash(hsv: np.ndarray, n_bins: int = 24) -> float:
    """0 = harmonious/analogous dominant hues, 1 = strong complementary clash.
    Looks at the two strongest saturation-weighted hue clusters and checks how
    close their circular distance is to 180 degrees (complementary), scaled by
    how saturated both clusters are (muted complementary colors clash less)."""
    h = hsv[..., 0].astype(np.int32)
    weight = (hsv[..., 1] / 255.0) * (hsv[..., 2] / 255.0)
    hist, edges = np.histogram(h, bins=n_bins, range=(0, 180), weights=weight)
    if hist.sum() < 1e-6:
        return 0.0
    order = np.argsort(hist)[::-1]
    top1, top2 = order[0], order[1]
    centers = (edges[:-1] + edges[1:]) / 2.0 * 2.0  # degrees, 0-360
    d = abs(centers[top1] - centers[top2])
    circ_dist = min(d, 360 - d)  # 0-180
    complementary_closeness = 1.0 - abs(circ_dist - 180.0) / 180.0
    strength = min(1.0, (hist[top1] + hist[top2]) / (hist.sum() + 1e-9) * 1.6)
    sat_weight = float(np.mean(hsv[..., 1]) / 255.0)
    return float(np.clip(complementary_closeness * strength * (0.4 + 0.6 * sat_weight), 0, 1))


def _edge_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    return cv2.Canny(gray, lo, hi)


def line_density(edges: np.ndarray) -> float:
    return float(np.count_nonzero(edges) / edges.size)


def line_thickness(edges: np.ndarray) -> float:
    """Average stroke width in pixels, estimated by skeletonizing the edge map
    and sampling the distance-transform of the dilated edge mask at skeleton
    points (distance*2 ~= local width). Normalized against image diagonal so
    it's resolution-independent, then rescaled to a 0-1 'thin -> thick' score."""
    if np.count_nonzero(edges) == 0:
        return 0.0
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    dist = cv2.distanceTransform(dilated, cv2.DIST_L2, 5)
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(dilated > 0)
        widths = dist[skel] * 2.0
    except Exception:
        widths = dist[dilated > 0] * 2.0
    if widths.size == 0:
        return 0.0
    diag = np.hypot(*edges.shape)
    mean_width_norm = float(np.mean(widths)) / diag
    # typical range observed empirically ~0.002 (hairline) to ~0.012 (bold); rescale to 0-1
    return float(np.clip((mean_width_norm - 0.002) / 0.010, 0, 1))


def line_angularity(edges: np.ndarray) -> float:
    """0 = smooth/curved contours, 1 = sharp/angular. Approximates each
    contour with polyDP, measures turning angle at each vertex, and reports
    the fraction of turns sharper than 55 degrees, weighted by contour length."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    total_len = 0.0
    sharp_len = 0.0
    for c in contours:
        length = cv2.arcLength(c, False)
        if length < 12:
            continue
        eps = 0.01 * length
        approx = cv2.approxPolyDP(c, eps, False).reshape(-1, 2)
        if len(approx) < 3:
            total_len += length
            continue
        sharp_frac = 0.0
        for i in range(1, len(approx) - 1):
            v1 = approx[i - 1] - approx[i]
            v2 = approx[i + 1] - approx[i]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
            angle = np.degrees(np.arccos(cos_a))
            turn = 180 - angle  # 0 = straight through, 180 = full reversal
            if turn > 55:
                sharp_frac += 1
        sharp_frac /= max(1, len(approx) - 2)
        total_len += length
        sharp_len += length * sharp_frac
    if total_len < 1e-6:
        return 0.0
    return float(np.clip(sharp_len / total_len, 0, 1))


def composition_density(img_bgr: np.ndarray) -> float:
    """Detail/business via variance of the Laplacian. Divisor calibrated
    against the 5-painting reference set (range ~340-5700) rather than
    guessed — revisit once a larger, more varied sample is available.
    Note this measures pixel-level textural busyness (it will read heavy
    impasto/dabbed brushwork, e.g. Monet, as 'busy' even when the large-scale
    composition reads as calm) more than large-scale compositional clutter;
    that conflation is a known limitation, not a bug."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    v = float(np.var(lap))
    return float(np.clip(v / 6000.0, 0, 1))


def symmetry(img_bgr: np.ndarray) -> float:
    """Best of horizontal/vertical mirror similarity (0-1), via normalized
    cross-correlation of grayscale halves."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def half_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
        return float(np.clip(np.sum(a * b) / denom, 0, 1))

    h, w = gray.shape
    left, right = gray[:, : w // 2], np.fliplr(gray[:, w - w // 2 :])
    top, bottom = gray[: h // 2, :], np.flipud(gray[h - h // 2 :, :])
    return float(max(half_similarity(left, right), half_similarity(top, bottom)))


def negative_space_ratio(hsv: np.ndarray, tol: int = 18) -> float:
    """Fraction of pixels close (in hue/sat/val) to the single most common
    color — a proxy for uniform background / empty space."""
    small = hsv[::4, ::4].reshape(-1, 3)
    median = np.median(small, axis=0)
    dist = np.linalg.norm(hsv.reshape(-1, 3) - median, axis=1)
    return float(np.mean(dist < tol))


def focal_contrast_position(img_bgr: np.ndarray) -> tuple[float, float]:
    """(x, y) in 0-1 of the centroid of the highest-contrast region (top 5%
    of |Laplacian| energy), i.e. where the eye is most likely pulled first."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    thresh = np.percentile(lap, 95)
    ys, xs = np.where(lap >= thresh)
    if len(xs) == 0:
        return 0.5, 0.5
    h, w = gray.shape
    return float(np.mean(xs) / w), float(np.mean(ys) / h)


def movement_direction(img_bgr: np.ndarray) -> dict:
    """Dominant gradient-orientation bucket (diagonal / horizontal / vertical)
    via the structure tensor, plus a 0-1 'dynamism' score = how dominant that
    bucket is over a uniform spread (i.e. how directional vs chaotic/static)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mask = mag > np.percentile(mag, 70)
    angles = (np.degrees(np.arctan2(gy[mask], gx[mask])) + 180) % 180  # 0-180
    hist, edges = np.histogram(angles, bins=18, range=(0, 180))
    bucket_centers = (edges[:-1] + edges[1:]) / 2
    dom_idx = int(np.argmax(hist))
    dom_angle = bucket_centers[dom_idx]
    dynamism = float(hist[dom_idx] / (hist.sum() + 1e-9)) * 18  # >1 = concentrated, ~1 = flat/chaotic
    dynamism = float(np.clip((dynamism - 1) / 3, 0, 1))
    # dom_angle is the dominant GRADIENT direction, which is perpendicular to
    # the underlying line/edge direction — so a gradient near 0/180 means the
    # edges themselves run vertically, and a gradient near 90 means the edges
    # run horizontally. (This was inverted in an earlier version.)
    if dom_angle < 20 or dom_angle > 160:
        label = "אנכי"
    elif 70 <= dom_angle <= 110:
        label = "אופקי"
    else:
        label = "אלכסוני"
    return {"label": label, "dominant_angle_deg": float(dom_angle), "dynamism": dynamism}


def style_bucket(feats: dict) -> dict:
    """Rule-based placeholder for a real trained style classifier. Buckets
    into the seven idioms used in the spec's style table (section ז),
    from features already extracted. This is intentionally crude — its job
    right now is to be replaced, not to be right."""
    density = feats["composition_density"]
    angularity = feats["line_angularity"]
    sym = feats["symmetry"]
    hues = feats["hue_variety"]
    edges = feats["line_density"]
    sat = feats["saturation"]

    scores = {
        "מינימליזם": (1 - density) * 1.2 + sym * 0.8 + (1 - hues) * 0.6,
        "קוביזם / אבסטרקט-גאומטרי": angularity * 1.1 + sym * 0.7 + (1 - hues) * 0.3,
        "אקספרסיוניזם": angularity * 0.9 + feats["color_clash"] * 1.0 + edges * 0.6,
        "אימפרסיוניזם": (1 - angularity) * 0.8 + hues * 0.7 + (1 - edges) * 0.6,
        "אבסטרקט-גסטורלי": edges * 1.0 + (1 - sym) * 0.8 + hues * 0.5,
        "ריאליזם": (1 - feats["color_clash"]) * 0.6 + (1 - angularity) * 0.4 + density * 0.4,
    }
    best = max(scores, key=scores.get)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {"bucket": best, "runner_up": ranked[1][0], "scores": {k: round(v, 2) for k, v in scores.items()}}


def extract_features(path: str) -> dict:
    img = load_image(path)
    hsv = _hsv(img)
    edges = _edge_map(img)

    feats = {
        "brightness": brightness_value(hsv),
        "saturation": saturation(hsv),
        "color_temperature": color_temperature(hsv),
        "hue_variety": hue_variety(hsv),
        "color_clash": color_clash(hsv),
        "value_contrast": value_contrast(hsv),
        "line_density": line_density(edges),
        "line_thickness": line_thickness(edges),
        "line_angularity": line_angularity(edges),
        "composition_density": composition_density(img),
        "symmetry": symmetry(img),
        "negative_space_ratio": negative_space_ratio(hsv),
    }
    fx, fy = focal_contrast_position(img)
    feats["focal_point"] = {"x": fx, "y": fy}
    feats["movement"] = movement_direction(img)
    feats["style"] = style_bucket(feats)
    return feats

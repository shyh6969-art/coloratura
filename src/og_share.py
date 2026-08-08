"""
Coloratura — gallery share-card compositor (idea #4 of the second "5
ideas" round the user asked for). Renders a real 1200x630 Open Graph image
per published gallery entry (painting + title + style badge, a waveform
strip too for image->music entries) so pasting a gallery link into
WhatsApp/Twitter/etc. shows a rich preview instead of a bare link — the
whole point being to turn a share into a small advertisement for the site,
not just a URL.

Server-side text rendering, not the browser: OG crawlers don't run JS and
don't screenshot pages, they just fetch whatever og:image points at, so
this has to be a real static image file generated with PIL.

Text rendering needs two things stock PIL doesn't give you for free:
  1. Actual font files with Hebrew AND Latin glyph coverage —
     python:3.12-slim ships none at all. See Dockerfile's font installs.
     Found in production, not assumed: Debian's fonts-noto-core installs a
     Hebrew-SUBSET file that renders Hebrew correctly but tofu-boxes plain
     Latin titles (a real uploaded file's own name, e.g. "painting.jpg") —
     PIL does zero font-fallback chaining the way a browser would, so one
     font file has to cover whatever text it's asked to draw, and this one
     doesn't cover both. Fixed by keeping fonts-dejavu-core installed too
     (solid, stable-path Latin coverage) and picking per-STRING between
     the Hebrew-capable font and DejaVu based on whether that particular
     string contains any Hebrew characters — not a single font gamble.
     Both are found via a glob search rather than a hardcoded exact
     filename (the precise name inside each package isn't worth pinning
     against), degrading to PIL's own default bitmap font if a whole
     category genuinely isn't found rather than crashing.
  2. Bidi reordering — PIL draws codepoints in the order given with no
     awareness that Hebrew is RTL, so a Hebrew string handed to it
     directly renders backwards. python-bidi's get_display() does the
     same logical-to-visual reordering a real text-shaping engine would,
     which is enough for these short, single-direction label strings
     (not full paragraph-level mixed-direction typesetting, which this
     never needs here).

Deliberately no emoji glyphs (unlike the '🎨 → 🎵' labels used elsewhere in
the UI) — Noto Sans has no color emoji coverage, so an emoji here would
likely render as a blank tofu box. Plain Unicode arrows instead, which are
ordinary punctuation-block characters any text font covers.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

CANVAS_SIZE = (1200, 630)
_BAND_HEIGHT = 220

_DIRECTION_LABEL = {
    "image2music": "ציור שהפך למוזיקה",
    "music2image": "מוזיקה שהפכה לציור",
}

_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

_font_cache: dict[tuple[bool, int], "ImageFont.FreeTypeFont"] = {}


def _find_font_path(hebrew: bool) -> str | None:
    """See module docstring — one font file per script, not one font
    gambled to cover everything."""
    patterns = (
        [
            "/usr/share/fonts/**/*Hebrew*.[to]tf",
            "/usr/share/fonts/**/*hebrew*.[to]tf",
            "/usr/share/fonts/**/NotoSans-*.[to]tf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        if hebrew
        else [
            "/usr/share/fonts/**/DejaVuSans-Bold.[to]tf",
            "/usr/share/fonts/**/DejaVuSans.[to]tf",
            "/usr/share/fonts/**/*.[to]tf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def _load_font(size: int, text: str) -> "ImageFont.ImageFont":
    """Font choice depends on the actual text being drawn — see module
    docstring for why a single font can't safely be assumed to cover both
    the Hebrew labels and a real uploaded file's (often Latin) name."""
    hebrew = bool(_HEBREW_RE.search(text))
    key = (hebrew, size)
    if key in _font_cache:
        return _font_cache[key]
    path = _find_font_path(hebrew)
    font = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    _font_cache[key] = font
    return font


def _rtl(text: str) -> str:
    """Reorder into visual order for PIL, which has no bidi awareness of
    its own — see module docstring."""
    return get_display(text)


def _cover_fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _clean_title(raw: str) -> str:
    """entry titles come straight from source_image/source_audio, which
    for a real upload is the original filename — strip a trailing
    extension so 'munch_scream.jpg' reads as 'munch_scream' rather than
    looking like a broken file link."""
    p = Path(raw)
    return p.stem if p.suffix and len(p.suffix) <= 5 else raw


def _draw_waveform(draw: ImageDraw.ImageDraw, wav_path: Path, box: tuple[int, int, int, int]) -> None:
    """A simple amplitude-bar waveform (not a real spectrogram — this is
    decorative-but-informative context for a share card, not an analysis
    tool) from the actual generated audio, drawn as translucent white bars
    over the darkened band."""
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    try:
        y, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    except Exception:
        return
    mono = y.mean(axis=1)
    n_bars = 80
    if len(mono) < n_bars:
        return
    chunk = len(mono) // n_bars
    peaks = [float(np.abs(mono[i * chunk:(i + 1) * chunk]).max()) for i in range(n_bars)]
    peak_max = max(peaks) or 1.0
    bar_w = width / n_bars
    for i, p in enumerate(peaks):
        bar_h = max(3, (p / peak_max) * height)
        bx = x0 + i * bar_w
        draw.rectangle(
            [bx, y0 + (height - bar_h) / 2, bx + bar_w * 0.6, y0 + (height + bar_h) / 2],
            fill=(255, 255, 255, 130),
        )


def compose_share_image(entry: dict, source_image_path: Path, wav_path: Path | None, out_path: str) -> None:
    """entry: a gallery.py entry dict (id, direction, title, style_idiom,
    vat, ...). source_image_path: the real image to build the card around
    (the input painting for image2music entries, the generated painting
    for music2image ones — either way, a real painting, never a stand-in).
    wav_path: the generated audio to draw a waveform strip from, only for
    image2music entries (None otherwise)."""
    base = Image.open(source_image_path).convert("RGB")
    canvas = _cover_fit(base, CANVAS_SIZE).convert("RGBA")

    band = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    band_draw = ImageDraw.Draw(band)
    band_top = CANVAS_SIZE[1] - _BAND_HEIGHT
    for i in range(_BAND_HEIGHT):
        alpha = int(200 * (i / _BAND_HEIGHT))
        band_draw.line([(0, band_top + i), (CANVAS_SIZE[0], band_top + i)], fill=(10, 7, 16, alpha))
    canvas = Image.alpha_composite(canvas, band)
    draw = ImageDraw.Draw(canvas)

    pad = 48
    if wav_path is not None and wav_path.exists():
        _draw_waveform(draw, wav_path, (pad, band_top + 18, CANVAS_SIZE[0] - pad, band_top + 68))
        text_top = band_top + 84
    else:
        text_top = band_top + 30

    title_text = _clean_title(entry["title"])
    title_font = _load_font(46, title_text)
    draw.text((CANVAS_SIZE[0] - pad, text_top), _rtl(title_text),
              font=title_font, fill=(255, 255, 255, 255), anchor="ra")

    meta = f'{entry.get("style_idiom", "")}, {_DIRECTION_LABEL.get(entry["direction"], "")}'
    meta_font = _load_font(26, meta)
    draw.text((CANVAS_SIZE[0] - pad, text_top + 62), _rtl(meta),
              font=meta_font, fill=(230, 200, 235, 255), anchor="ra")

    brand_font = _load_font(22, "קולורטורה")
    draw.text((pad, CANVAS_SIZE[1] - 36), "קולורטורה", font=brand_font, fill=(255, 255, 255, 200), anchor="la")

    canvas.convert("RGB").save(out_path, "JPEG", quality=87)

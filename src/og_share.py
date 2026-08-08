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
  1. Actual font files with Hebrew AND Latin/punctuation glyph coverage —
     python:3.12-slim ships none at all. See Dockerfile's font installs.
     Found in production through repeated real testing, not assumed —
     each fix here came from actually looking at the rendered image, not
     from reasoning about font internals in the abstract:
       a) Debian's fonts-noto-core installs a Hebrew-SUBSET file that
          renders Hebrew letters correctly but tofu-boxes plain Latin
          text (a real uploaded file's own name, e.g. "painting.jpg").
       b) That same subset ALSO doesn't cover plain ASCII punctuation
          used inside otherwise-Hebrew strings — an arrow, a middle dot,
          even a bare hyphen-minus inside "אבסטרקט-גסטורלי" all tofu-boxed
          in turn, one at a time, across three separate production checks.
     PIL does zero font-fallback chaining the way a browser would, so
     whatever font draws a run of text has to itself cover every
     character in that run. Rewording strings to dodge individual
     unsupported characters (tried first, see git history) turned into
     whack-a-mole — any new string could hit the same gap again. The
     actual fix: segment each string into runs by Unicode block (Hebrew
     vs. everything else) via _segment_runs(), and render EACH RUN with
     whichever font is proven to cover it — fonts-dejavu-core (a separate,
     stable-path, broad-coverage package) for every non-Hebrew run
     (Latin, digits, punctuation, symbols), the Noto Hebrew subset only
     for the Hebrew letters themselves. Neither font has to cover
     everything; each only has to cover what it's actually asked to draw.
  2. Bidi reordering — PIL draws codepoints in the order given with no
     awareness that Hebrew is RTL, so a Hebrew string handed to it
     directly renders backwards. python-bidi's get_display() does the
     same logical-to-visual reordering a real text-shaping engine would;
     _draw_mixed_text() reorders BEFORE segmenting into font runs, so a
     mixed Hebrew/Latin/punctuation string still lands in correct reading
     order with each piece in the right font.

Deliberately no emoji glyphs (unlike the '🎨 → 🎵' labels used elsewhere in
the UI) — Noto Sans has no color emoji coverage, so an emoji here would
likely render as a blank tofu box.
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


def _load_font(size: int, hebrew: bool) -> "ImageFont.ImageFont":
    key = (hebrew, size)
    if key in _font_cache:
        return _font_cache[key]
    path = _find_font_path(hebrew)
    font = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
    _font_cache[key] = font
    return font


def _segment_runs(text: str) -> list[tuple[bool, str]]:
    """Split into consecutive (is_hebrew, substring) runs — see module
    docstring for why a whole string can't safely be drawn with one font.
    Non-Hebrew runs cover Latin, digits, punctuation and symbols alike;
    fonts-dejavu-core (see Dockerfile) has broad enough coverage that
    those don't need their own further split."""
    runs: list[tuple[bool, str]] = []
    current_hebrew: bool | None = None
    current: list[str] = []
    for ch in text:
        is_hebrew = bool(_HEBREW_RE.match(ch))
        if current_hebrew is None:
            current_hebrew = is_hebrew
        if is_hebrew != current_hebrew:
            runs.append((current_hebrew, "".join(current)))
            current = [ch]
            current_hebrew = is_hebrew
        else:
            current.append(ch)
    if current:
        runs.append((current_hebrew, "".join(current)))
    return runs


def _draw_mixed_text(draw: ImageDraw.ImageDraw, text: str, right_x: int, top_y: int,
                      size: int, fill: tuple[int, int, int, int]) -> None:
    """Right-aligned text draw that's safe for a string mixing Hebrew
    words with Latin/punctuation/digits — see module docstring. Reorders
    into visual order first (python-bidi), THEN segments into per-script
    runs, so a string like 'אבסטרקט-גסטורלי' or a Latin filename both come
    out in correct reading order with each run in a font that actually
    covers it."""
    visual = get_display(text)
    runs = [(hebrew, run, _load_font(size, hebrew)) for hebrew, run in _segment_runs(visual)]
    total_width = sum(draw.textlength(run, font=font) for _, run, font in runs)
    x = right_x - total_width
    for _, run, font in runs:
        draw.text((x, top_y), run, font=font, fill=fill, anchor="la")
        x += draw.textlength(run, font=font)


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
    _draw_mixed_text(draw, title_text, CANVAS_SIZE[0] - pad, text_top, 46, (255, 255, 255, 255))

    meta = f'{entry.get("style_idiom", "")}, {_DIRECTION_LABEL.get(entry["direction"], "")}'
    _draw_mixed_text(draw, meta, CANVAS_SIZE[0] - pad, text_top + 62, 26, (230, 200, 235, 255))

    brand_font = _load_font(22, hebrew=True)
    draw.text((pad, CANVAS_SIZE[1] - 36), "קולורטורה", font=brand_font, fill=(255, 255, 255, 200), anchor="la")

    canvas.convert("RGB").save(out_path, "JPEG", quality=87)

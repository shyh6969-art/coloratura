"""
Coloratura — Stage B for the music -> painting direction: AI-quality image
rendering via OpenAI's gpt-image-2, the visual counterpart to stage_b.py's
Suno integration.

Takes image_stage_a.py's procedural PNG as the reference image and a text
prompt built from visual_mapping_engine.py's brief, and asks the
images.edit endpoint (POST /v1/images/edits) to produce an AI-quality
reinterpretation — not a literal upscale. Same role Suno's cover-
generation endpoint plays for audio: a creative reinterpretation guided by
a reference, not literal reproduction.

Simpler than stage_b.py's Suno flow in one real way: OpenAI's edit call is
a single synchronous HTTP request, not a job you poll a task_id for. But
that also means it can legitimately take 10-60+ seconds depending on
quality/size, which is exactly the shape of problem that already broke
the audio pipeline once in production (see audio_features.load_mono's
docstring) — webapp.py runs this in a background thread and polls an
in-memory status dict instead of blocking the HTTP request on it, rather
than repeat that mistake.

Requires OPENAI_API_KEY in .env (see env_config.py). Costs real money per
call (~$0.04-$0.35/image depending on quality/size) — gated behind the
same explicit-confirmation UI pattern as the audio Stage B.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from env_config import get_env

EDIT_URL = "https://api.openai.com/v1/images/edits"
MODEL = "gpt-image-2"


class OpenAIConfigError(Exception):
    pass


class OpenAIAPIError(Exception):
    pass


# style bucket -> English art-movement phrase, mirroring stage_b.py's
# STYLE_EN pattern (Hebrew brief fields translated to English prompt text
# Suno/OpenAI can actually work with).
STYLE_EN = {
    "אימפרסיוניזם": "impressionist painting style, soft blended brushstrokes, diffused light",
    "אקספרסיוניזם": "expressionist painting style, raw emotional distorted forms, bold non-naturalistic color",
    "קוביזם / אבסטרקט-גאומטרי": "cubist geometric abstract painting, fragmented angular facets",
    "מינימליזם": "minimalist painting, flat fields of color, generous negative space",
    "ריאליזם": "realist painting style, naturalistic detail, balanced composition",
    "סוריאליזם": "surrealist painting, dreamlike uncanny imagery",
    "אבסטרקט-גסטורלי": "abstract expressionist gestural painting, spontaneous sweeping brushwork",
}

_VALENCE_WORDS = [(0.66, "joyful, warm"), (0.45, "calm, gentle"), (0.30, "melancholic"), (0.0, "somber, dark")]
_AROUSAL_WORDS = [(0.66, "intense, dynamic"), (0.40, "moderately energetic"), (0.0, "quiet, still")]
_TENSION_WORDS = [(0.6, "ominous, unsettling"), (0.35, "somewhat tense"), (0.0, "harmonious, resolved")]


def _pick(value: float, table: list[tuple[float, str]]) -> str:
    for threshold, word in table:
        if value >= threshold:
            return word
    return table[-1][1]


def build_prompt(brief: dict) -> str:
    vat = brief["vat"]
    style_phrase = STYLE_EN.get(brief["style_idiom"], "abstract painting")
    mood = ", ".join([
        _pick(vat["valence"], _VALENCE_WORDS),
        _pick(vat["arousal"], _AROUSAL_WORDS),
        _pick(vat["tension_emotional"], _TENSION_WORDS),
    ])
    return (
        f"Repaint this as a museum-quality {style_phrase} artwork. "
        f"Mood: {mood}. Preserve the overall composition, color palette, "
        f"and focal point of the reference image, but render it with real "
        f"painterly technique, texture, and artistic finish — not a flat "
        f"digital illustration."
    )


def request_edit(image_path: str, prompt: str, size: str = "1024x1024", quality: str = "medium") -> bytes:
    """quality="medium" by default, found empirically rather than assumed:
    "high" didn't even complete within a 120s timeout on a real test call;
    "medium" produced a genuinely strong result (visible painterly texture,
    faithfully preserved composition/palette/symmetry from the Stage A
    reference) in 68.6s. Neither quality level is fast — webapp.py runs
    this in a background thread and polls rather than blocking a request
    on it either way (see start_image_stage_b's docstring)."""
    api_key = get_env("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIConfigError("OPENAI_API_KEY חסר ב-.env")

    boundary = "----coloratura" + os.urandom(8).hex()
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    def _field(name: str, value: str) -> bytes:
        return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()

    body = b"".join([
        _field("model", MODEL),
        _field("prompt", prompt),
        _field("size", size),
        _field("quality", quality),
        f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\n'
        f'Content-Type: image/png\r\n\r\n'.encode(),
        image_bytes,
        f'\r\n--{boundary}--\r\n'.encode(),
    ])

    req = urllib.request.Request(
        EDIT_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:500]
        raise OpenAIAPIError(f"שגיאת OpenAI ({e.code}): {detail}") from e
    except Exception as e:
        raise OpenAIAPIError(f"שגיאת בקשה ל-OpenAI: {e}") from e

    b64 = result["data"][0]["b64_json"]
    return base64.b64decode(b64)

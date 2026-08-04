"""
Coloratura — Stage B: brief + Stage A audio -> realistic rendering via Suno.

Honest architecture note, found during research before writing a line of
this: Suno has no public first-party API, and none of the available
third-party proxies (this module targets sunoapi.org) accept MIDI or exact
note/voice-leading data as conditioning. The closest available capability is
"upload-and-cover-audio" — you hand it a URL to a reference AUDIO file (our
own crude synth.py render of the Stage A MIDI) plus a text prompt, and it
generates *new* audio inspired by that reference, not a faithful performance
of our exact SATB voice-leading. That's a real downgrade from what the spec
doc's section ח originally described ("מבצע את השלד" — performs the
skeleton) — the honest framing is "AI reinterpretation conditioned on our
sketch," not "AI orchestration of our exact notes." The spec doc should be
updated to reflect this once this stage is actually exercised end-to-end.

UNTESTED against the live API: written without a SUNO_API_KEY in hand. Set
the SUNO_API_KEY environment variable, or put it in a .env file at the repo
root (see .env.example — that file is gitignored, so the key never gets
committed) before calling anything here — see README for the full setup.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request

BASE_URL = "https://api.sunoapi.org"
# these paths came from docs.sunoapi.org's per-endpoint pages, fetched
# directly — the docs *site's* sidebar/index page shows page slugs
# (like "/suno-api/upload-and-cover-audio") that look like API paths but
# aren't; the real paths only appear on each endpoint's own page.
COVER_PATH = "/api/v1/generate/upload-cover"
STATUS_PATH = "/api/v1/generate/record-info"

TERMINAL_OK = "SUCCESS"
TERMINAL_FAIL = {"CREATE_TASK_FAILED", "GENERATE_AUDIO_FAILED", "SENSITIVE_WORD_ERROR"}

_ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


class SunoConfigError(RuntimeError):
    """Raised when SUNO_API_KEY isn't set, or a required brief field is missing."""


class SunoAPIError(RuntimeError):
    """Raised on a non-2xx response, a FAILED task status, or a malformed response body."""


def _read_env_file(key_name: str) -> str | None:
    """Minimal .env parser (no python-dotenv dependency) — just enough to
    read KEY=value lines, since the harness running this doesn't reliably
    keep exported shell variables across separate command invocations, but
    a file on disk is always there."""
    if not os.path.exists(_ENV_FILE):
        return None
    with open(_ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key_name:
                return v.strip().strip('"').strip("'") or None
    return None


def _api_key() -> str:
    key = os.environ.get("SUNO_API_KEY") or _read_env_file("SUNO_API_KEY")
    if not key:
        raise SunoConfigError(
            "SUNO_API_KEY is not set. Get a key at https://sunoapi.org, then either "
            "set it as an environment variable, or put SUNO_API_KEY=... in a .env "
            "file at the repo root (copy .env.example)."
        )
    return key


STYLE_EN = {
    "אימפרסיוניזם": "Impressionist",
    "אקספרסיוניזם": "Expressionist",
    "קוביזם / אבסטרקט-גאומטרי": "Cubist / geometric-abstract",
    "מינימליזם": "Minimalist",
    "ריאליזם": "Realist",
    "סוריאליזם": "Surrealist",
    "אבסטרקט-גסטורלי": "Abstract-expressionist / gestural",
}

INSTRUMENT_EN = {
    "מיתרים": "strings", "קוורטט כלי-קשת": "string quartet", "פאד מיתרים גבוה": "high synth string pad",
    "קרן יער": "French horn", "פסנתר": "piano", "פסנתר פרפטואום מוביל": "perpetual-motion piano",
    "פעמוני צינור": "tubular bells", "קלרינט": "clarinet", "נבל": "harp", "חלילית": "piccolo",
    "נחושת מעוותת (con sordino)": "muted brass (con sordino)", "פרקאשן לא-מכוון": "unpitched percussion",
    "מרימבה פעימתית": "pulsing marimba", "כלי נשיפה-עץ": "woodwinds", "קונטרבס פיציקטו": "pizzicato double bass",
    "תופים חופשיים": "free drums", "סקסופון אלט": "alto saxophone",
}

MODE_EN = {
    "מאז'ורי (יוני) / לידי": "major (Ionian/Lydian)",
    "מיקסולידי": "Mixolydian",
    "אאולי (מינורי טבעי)": "natural minor (Aeolian)",
    "פריגי": "Phrygian",
    "לוקרי (הכהה והלא-יציב ביותר)": "Locrian",
}

HARMONIC_COMPLEXITY_EN = {
    "פשוט / דיאטוני": "harmonically simple and diatonic",
    "בינוני / כרומטיות מקומית": "moderately complex, with local chromaticism",
    "מורכב / כרומטי-מודולטיבי": "harmonically complex and chromatic, with modulations",
}

CADENCE_EN = {
    "authentic": "ends with a clear, fully resolved cadence",
    "half": "ends suspended on a half-cadence, not fully resolved",
    "none": "ends unresolved, hanging on a dissonant chord with no cadence",
}


def _articulation_phrase(staccato: float) -> str:
    if staccato >= 0.6:
        return "sharp, detached staccato articulation with strong accents"
    if staccato <= 0.3:
        return "smooth, connected legato phrasing throughout"
    return "mostly legato phrasing, with staccato accents at climactic moments"


def _mood_phrase(vat: dict) -> str:
    v, a, tf, te = vat["valence"], vat["arousal"], vat["tension_formal"], vat["tension_emotional"]
    valence_word = "joyful and warm" if v >= 0.6 else ("melancholic and somber" if v < 0.4 else "bittersweet")
    arousal_word = "energetic and driving" if a >= 0.6 else ("calm and unhurried" if a < 0.4 else "moderately animated")
    tension_word = "harmonically restless, dense with color" if tf >= 0.55 else "harmonically settled, clean"
    mood_word = ("with an ominous, unresolved undertone that never fully relaxes" if te >= 0.6
                 else "with a safe, settled emotional feeling")
    return f"{valence_word}, {arousal_word}, {tension_word}, {mood_word}"


def build_style_tags(brief: dict) -> str:
    """Comma-separated genre/style tags for Suno's `style` field."""
    style = STYLE_EN.get(brief["style_idiom"], brief["style_idiom"])
    temp_word = "warm-toned" if brief["raw_features"]["color_temperature"] >= 0.5 else "cool-toned"
    return f"{style}, orchestral, cinematic, instrumental, {temp_word}"


def build_prompt(brief: dict) -> str:
    """Descriptive transformation instruction for Suno's `prompt` field —
    this is what stands in for 'perform this composition,' since Suno has
    no literal note-conditioning. Built in English from engine_params and
    translation tables rather than the brief's Hebrew display strings, so
    the prompt sent to the API isn't a mid-sentence language mix."""
    ep = brief["engine_params"]
    style = STYLE_EN.get(brief["style_idiom"], brief["style_idiom"])
    mode_en = MODE_EN.get(ep["mode_id"], ep["mode_id"])
    complexity_en = HARMONIC_COMPLEXITY_EN.get(brief["harmonic_complexity"], brief["harmonic_complexity"])
    cadence_key = "authentic" if ep["resolved"] else "none"
    # _is_resolved collapses "half" into True — recover it from the label directly when possible
    if brief["harmonic_resolution"].startswith("פתור חלקית"):
        cadence_key = "half"
    cadence_en = CADENCE_EN[cadence_key]
    mood = _mood_phrase(brief["vat"])
    instruments = ", ".join(INSTRUMENT_EN.get(i, i) for i in brief["instrumentation"])
    articulation_en = _articulation_phrase(ep["staccato"])

    return (
        f"Re-orchestrate this exact instrumental sketch with a full, realistic "
        f"{style} ensemble, preserving its harmony, tempo and structure. "
        f"Key/mode: {ep['tonic_name']} {mode_en}. "
        f"Tempo: {brief['tempo_bpm']} BPM, {ep['meter_numerator']}/{ep['meter_denominator']} time. "
        f"Instrumentation to feature: {instruments}. Articulation: {articulation_en}. "
        f"Harmonic character: {complexity_en}; the piece {cadence_en}. "
        f"Overall mood: {mood}. No vocals."
    )


def _http_json(method: str, path: str, body: dict | None = None) -> dict:
    import json as _json

    req = urllib.request.Request(
        BASE_URL + path,
        data=_json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # sunoapi.org sits behind Cloudflare bot-protection that 403s
            # (Cloudflare error 1010) urllib's default "Python-urllib/x.y"
            # User-Agent outright, before the request ever reaches the API
            # layer — a browser-like UA is required just to get through.
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SunoAPIError(f"{method} {path} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e
    # a 2xx HTTP status here doesn't mean success — sunoapi.org wraps
    # business-logic errors (bad params, insufficient credits, etc.) in a
    # 200 response with a non-200 `code` field and `data: null`
    if payload.get("code") not in (200, None):
        raise SunoAPIError(f"{method} {path} -> API code {payload.get('code')}: {payload.get('msg')}")
    return payload


def build_title(brief: dict) -> str:
    stem = brief["source_image"].rsplit(".", 1)[0]
    return stem.replace("_", " ").title()


def request_cover(reference_audio_url: str, brief: dict, model: str = "V5_5",
                   callback_url: str = "https://example.invalid/callback") -> str:
    """Kicks off a cover generation task, returns the taskId. `callback_url`
    is required by the API schema even though this module polls for the
    result instead of standing up a webhook receiver — pass your own if you
    have one.

    customMode=True on purpose: non-custom mode caps `prompt` at 500
    characters (learned from a live 400 response — build_prompt()'s output
    routinely runs longer), and custom mode is also what actually lets
    `style` function as a distinct tag field rather than being ignored."""
    body = {
        "uploadUrl": reference_audio_url,
        "customMode": True,
        "instrumental": True,
        "model": model,
        "prompt": build_prompt(brief),
        "style": build_style_tags(brief),
        "title": build_title(brief),
        "callBackUrl": callback_url,
    }
    data = _http_json("POST", COVER_PATH, body)
    task_id = data.get("data", {}).get("taskId")
    if not task_id:
        raise SunoAPIError(f"No taskId in response: {data}")
    return task_id


def poll_until_done(task_id: str, timeout_s: int = 300, interval_s: int = 8) -> dict:
    """Blocks, polling get-music-generation-details, until SUCCESS or a
    terminal failure status, or timeout_s elapses."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        data = _http_json("GET", f"{STATUS_PATH}?taskId={task_id}")
        status = data.get("data", {}).get("status")
        if status == TERMINAL_OK:
            return data["data"]
        if status in TERMINAL_FAIL:
            raise SunoAPIError(f"Suno task {task_id} failed with status {status}: {data}")
        time.sleep(interval_s)
    raise SunoAPIError(f"Suno task {task_id} did not finish within {timeout_s}s")


def download_audio(result: dict, out_path: str) -> str:
    tracks = result.get("response", {}).get("sunoData", [])
    if not tracks:
        raise SunoAPIError(f"No audio tracks in result: {result}")
    audio_url = tracks[0]["audioUrl"]
    urllib.request.urlretrieve(audio_url, out_path)
    return audio_url


def compose_stage_b(brief: dict, reference_audio_url: str, out_path: str,
                     model: str = "V5_5", timeout_s: int = 300) -> dict:
    """End-to-end: request a cover of `reference_audio_url` conditioned on
    the brief, poll for completion, download the result to `out_path`.
    Returns the raw result dict from the status endpoint for inspection."""
    task_id = request_cover(reference_audio_url, brief, model=model)
    result = poll_until_done(task_id, timeout_s=timeout_s)
    audio_url = download_audio(result, out_path)
    return {"task_id": task_id, "audio_url": audio_url, **result}

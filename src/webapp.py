"""
Coloratura — web app: Stage A (free, always on) + Stage B (Suno, paid per call).

FastAPI wrapper around the existing pipeline. Stage A is local compute (CV +
CLIP + our own harmony engine + our own synthesizer) — no per-call cost, but
CLIP inference isn't free CPU-wise either, so every UI/API route sits behind
HTTP Basic Auth (see verify_credentials below).

Stage B is exposed here too, but deliberately gated behind the same auth
wall PLUS an explicit confirm step in the UI — unlike run_stage_b.py (the
CLI path, still the simpler option for batch/local use), this can be
triggered by anyone who has the site's login, and each click spends real
Suno credit. One consequence of adding it: Suno's servers need to fetch the
Stage A reference audio over a public URL they can reach, which our own
Basic Auth would otherwise block — see the deliberately-unauthenticated
/public/audio/{job_id}.wav route and its docstring for how that's scoped.

Run with: uvicorn webapp:app --reload --app-dir src
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from audio_features import extract_features as extract_audio_features
from audio_semantic import semantic_scores as audio_semantic_scores
from env_config import get_env
from feature_extraction import extract_features
import gallery
from image_stage_a import compose_image_stage_a
import image_stage_b
import itunes_source
from mapping_engine import build_brief
from semantic_features import semantic_scores
from stage_a import compose_stage_a
import stage_b
from synth import STAGE_A_PANS, render_midi_to_wav
from visual_mapping_engine import build_visual_brief

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
# PERSISTENT_DATA_DIR points at a mounted Render Disk in production (set as
# an env var once the disk is attached in Render's dashboard) so job output
# survives redeploys/restarts -- without it, every job lived on the
# container's ephemeral filesystem and was silently wiped on any restart
# (observed directly while testing image Stage B: a job triggered right
# before an env-var-triggered restart just vanished). Falls back to a
# plain local folder for local dev, where that persistence doesn't matter.
PERSISTENT_DIR = Path(get_env("PERSISTENT_DATA_DIR") or (APP_DIR.parent / "output" / "webapp"))
OUTPUT_DIR = PERSISTENT_DIR / "jobs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_INDEX_PATH = PERSISTENT_DIR / "gallery.json"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous for a photographed/scanned painting

ALLOWED_AUDIO_EXT = {".wav", ".mp3"}
MAX_AUDIO_UPLOAD_BYTES = 30 * 1024 * 1024  # 30MB — a few minutes of WAV/MP3
# analyze only a leading excerpt regardless of upload length — a real
# production failure (see audio_features.load_mono's docstring) traced to
# a 12-minute upload driving ~72 sequential CLAP forward passes, easily
# enough to exceed a request timeout on Render's proxy. 90s is plenty for
# mood/style detection and keeps worst-case request time bounded no matter
# how long the uploaded track actually is.
MAX_AUDIO_ANALYSIS_SECONDS = 90

# Known, not yet addressed: CLIP (~600MB) and CLAP (~1.2GB) are both lazy-
# loaded singletons that persist for the life of this process once first
# used (see semantic_features._load_clip / audio_semantic._load_clap). A
# single Render Standard instance (2GB RAM) that serves both an image
# analysis and an audio analysis in the same process lifetime will hold
# both models in memory at once — untested against the actual container
# limit, flagged here rather than assumed fine.

# job_id -> Suno taskId, so /status can be polled without re-requesting.
# In-memory and single-process is fine for this MVP (Render Standard runs
# one instance); it resets on redeploy, which just means an in-flight Stage
# B job started right before a redeploy needs to be re-triggered.
_stage_b_tasks: dict[str, str] = {}
# job_id -> the taskId whose result is currently sitting in stage_b.mp3 on
# disk, so a regeneration (new taskId, same job_id) knows to re-download.
_stage_b_downloaded: dict[str, str] = {}

# job_id -> "pending" | "done" | "failed:<message>" for the image Stage B
# (OpenAI gpt-image-2). No external task_id to track here (see
# image_stage_b.py's docstring — it's one synchronous call, not a job
# queue), so this dict IS the job state, updated by the background thread
# started in start_image_stage_b().
_image_stage_b_status: dict[str, str] = {}

security = HTTPBasic()
_AUTH_USER = get_env("WEBAPP_USER") or "admin"
_AUTH_PASSWORD = get_env("WEBAPP_PASSWORD")
if not _AUTH_PASSWORD:
    # no hardcoded default credential ships in the repo — generate one per
    # run instead, the way tools like Jenkins print an initial admin
    # password on first boot. Set WEBAPP_USER/WEBAPP_PASSWORD in .env (or
    # as host env vars, e.g. Render's Environment tab) for a password that
    # survives a restart.
    _AUTH_PASSWORD = secrets.token_urlsafe(12)
    print(
        "\n  No WEBAPP_PASSWORD set — generated one for this run:\n"
        f"    user:     {_AUTH_USER}\n"
        f"    password: {_AUTH_PASSWORD}\n"
        "  Set WEBAPP_USER / WEBAPP_PASSWORD for a stable login.\n",
        flush=True,  # stdout is block-buffered when redirected to a file/log
        # collector; without an explicit flush this can sit unread in the
        # buffer indefinitely on a low-traffic server, which defeats the
        # entire point of printing a one-time password to the log.
    )


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    # compare_digest on both, even though only the password is secret —
    # a plain == on the username would let an attacker use response-timing
    # to fish out the correct username before even trying passwords.
    user_ok = secrets.compare_digest(credentials.username, _AUTH_USER)
    pass_ok = secrets.compare_digest(credentials.password, _AUTH_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(401, "פרטי התחברות שגויים", headers={"WWW-Authenticate": "Basic"})
    return credentials.username


app = FastAPI(title="Coloratura")

# every route on this router requires login — the UI, the (free) Stage A
# pipeline, and the (paid) Stage B trigger all live here. Routes registered
# directly on `app` instead (see /public/audio below) are NOT covered by
# this, on purpose.
protected = APIRouter(dependencies=[Depends(verify_credentials)])


def _job_dir(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(400, "job id לא תקין")
    return OUTPUT_DIR / job_id


@protected.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@protected.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"פורמט קובץ לא נתמך: {ext or 'לא ידוע'}. נתמכים: JPG, PNG, WEBP")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / f"input{ext}"

    size = 0
    with open(image_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "הקובץ גדול מדי (מקסימום 15MB)")
            out.write(chunk)

    try:
        feats = extract_features(str(image_path))
        sem = semantic_scores(str(image_path))
        brief = build_brief(file.filename or "upload", feats, sem)

        midi_path = job_dir / "stage_a.mid"
        stats = compose_stage_a(brief, str(midi_path))

        wav_path = job_dir / "stage_a.wav"
        render_midi_to_wav(str(midi_path), str(wav_path), pans=STAGE_A_PANS)

        with open(job_dir / "brief.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "stage_a_stats": stats,
        "audio_url": f"/api/audio/{job_id}",
    })


@protected.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    wav_path = _job_dir(job_id) / "stage_a.wav"
    if not wav_path.exists():
        raise HTTPException(404, "לא נמצא — ייתכן שהג'וב פג תוקף")
    return FileResponse(wav_path, media_type="audio/wav")


@protected.post("/api/stage_b/{job_id}")
def start_stage_b(job_id: str, request: Request):
    """Kicks off a Suno cover generation for an already-analyzed painting.
    Costs real Suno credit — the frontend confirms with the user before
    ever calling this. Returns immediately; poll /status for the result
    rather than blocking here for the 2-3 minutes generation actually takes."""
    job_dir = _job_dir(job_id)
    brief_path = job_dir / "brief.json"
    if not (job_dir / "stage_a.wav").exists() or not brief_path.exists():
        raise HTTPException(404, "לא נמצא — צריך קודם להריץ ניתוח (דרגה A) על הציור הזה")

    with open(brief_path, encoding="utf-8") as f:
        brief = json.load(f)

    reference_url = str(request.base_url).rstrip("/") + f"/public/audio/{job_id}.wav"
    try:
        task_id = stage_b.request_cover(reference_url, brief)
    except stage_b.SunoConfigError as e:
        raise HTTPException(500, str(e)) from e
    except stage_b.SunoAPIError as e:
        raise HTTPException(502, f"שגיאת Suno: {e}") from e

    _stage_b_tasks[job_id] = task_id
    return JSONResponse({"task_id": task_id})


@protected.get("/api/stage_b/{job_id}/status")
def stage_b_status(job_id: str):
    """Single non-blocking status check — the frontend calls this
    repeatedly on its own timer instead of one request hanging open for
    minutes (which Render's proxy would likely kill anyway)."""
    task_id = _stage_b_tasks.get(job_id)
    if not task_id:
        raise HTTPException(404, "לא נמצאה בקשת דרגה B לג'וב הזה")

    try:
        result = stage_b.check_once(task_id)
    except stage_b.SunoAPIError as e:
        raise HTTPException(502, f"שגיאת Suno: {e}") from e

    status = result.get("status")
    if status == stage_b.TERMINAL_OK:
        job_dir = _job_dir(job_id)
        mp3_path = job_dir / "stage_b.mp3"
        # keyed on task_id, not just file existence — a regenerated ("צור
        # שוב") job reuses the same job_id but gets a new task_id, and its
        # result must overwrite the previous file, not be skipped because
        # *a* stage_b.mp3 happens to already be sitting there
        if _stage_b_downloaded.get(job_id) != task_id:
            try:
                stage_b.download_audio(result, str(mp3_path))
            except stage_b.SunoAPIError as e:
                raise HTTPException(502, f"שגיאת הורדה מ-Suno: {e}") from e
            _stage_b_downloaded[job_id] = task_id
        return JSONResponse({"status": "done", "audio_url": f"/api/stage_b_audio/{job_id}"})
    if status in stage_b.TERMINAL_FAIL:
        return JSONResponse({"status": "failed", "detail": status})
    return JSONResponse({"status": "pending", "detail": status})


@protected.get("/api/stage_b_audio/{job_id}")
def get_stage_b_audio(job_id: str):
    mp3_path = _job_dir(job_id) / "stage_b.mp3"
    if not mp3_path.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(mp3_path, media_type="audio/mpeg")


@protected.post("/api/analyze_audio")
async def analyze_audio(file: UploadFile = File(...)):
    """The reverse direction's Stage A: audio in, procedurally-painted PNG
    out. Mirrors /api/analyze's structure (same job_dir/upload/error-
    cleanup pattern) with a separate job_id namespace-in-practice — both
    directions write into OUTPUT_DIR/{job_id}, but since job_id is fresh
    per request either way, there's no real collision risk."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise HTTPException(400, f"פורמט קובץ לא נתמך: {ext or 'לא ידוע'}. נתמכים: WAV, MP3")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / f"input{ext}"

    size = 0
    with open(audio_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_AUDIO_UPLOAD_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "הקובץ גדול מדי (מקסימום 30MB)")
            out.write(chunk)

    try:
        audio_feats = extract_audio_features(str(audio_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        sem = audio_semantic_scores(str(audio_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        brief = build_visual_brief(file.filename or "upload", audio_feats, sem)

        png_path = job_dir / "painting.png"
        stats = compose_image_stage_a(brief, str(png_path))

        with open(job_dir / "visual_brief.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False)
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "image_stats": stats,
        "image_url": f"/api/painting/{job_id}",
    })


@protected.get("/api/painting/{job_id}")
def get_painting(job_id: str):
    png_path = _job_dir(job_id) / "painting.png"
    if not png_path.exists():
        raise HTTPException(404, "לא נמצא — ייתכן שהג'וב פג תוקף")
    return FileResponse(png_path, media_type="image/png")


@protected.get("/api/search_track")
def search_track(q: str):
    """The Spotify substitute — see itunes_source.py's docstring for why
    this is Apple's public iTunes Search API rather than Spotify's."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(400, "יש להזין שם שיר/אמן")
    try:
        results = itunes_source.search_tracks(q.strip())
    except itunes_source.ITunesError as e:
        raise HTTPException(502, str(e)) from e
    return JSONResponse({"results": results})


@protected.post("/api/analyze_itunes")
async def analyze_itunes(payload: dict = Body(...)):
    """Same pipeline as /api/analyze_audio, fed by a downloaded-and-
    converted iTunes preview clip instead of a direct upload. 30s previews
    are already well under MAX_AUDIO_ANALYSIS_SECONDS, but the cap is
    applied anyway for consistency rather than as a special case."""
    preview_url = payload.get("preview_url")
    label = payload.get("track_name") or "itunes_track"
    if not preview_url:
        raise HTTPException(400, "preview_url חסר")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    mp3_path = job_dir / "input.mp3"

    try:
        itunes_source.download_preview_as_mp3(preview_url, str(mp3_path))
        audio_feats = extract_audio_features(str(mp3_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        sem = audio_semantic_scores(str(mp3_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        brief = build_visual_brief(label, audio_feats, sem)

        png_path = job_dir / "painting.png"
        stats = compose_image_stage_a(brief, str(png_path))

        with open(job_dir / "visual_brief.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False)
    except itunes_source.ITunesError as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(502, str(e)) from e
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "image_stats": stats,
        "image_url": f"/api/painting/{job_id}",
    })


def _run_image_stage_b(job_id: str, image_path: str, prompt: str) -> None:
    """Runs in a background thread (see start_image_stage_b) — the whole
    point is that the HTTP request that kicks this off returns immediately
    rather than blocking on a call that can legitimately take 10-60+
    seconds, the same class of problem that already broke the audio
    pipeline once (see audio_features.load_mono's docstring)."""
    try:
        png_bytes = image_stage_b.request_edit(image_path, prompt)
        out_path = Path(image_path).parent / "stage_b_painting.png"
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        _image_stage_b_status[job_id] = "done"
    except (image_stage_b.OpenAIConfigError, image_stage_b.OpenAIAPIError) as e:
        _image_stage_b_status[job_id] = f"failed:{e}"
    except Exception as e:
        _image_stage_b_status[job_id] = f"failed:שגיאה: {e}"


@protected.post("/api/image_stage_b/{job_id}")
def start_image_stage_b(job_id: str):
    """Kicks off an OpenAI gpt-image-2 edit for an already-analyzed audio
    job. Costs real money per call — the frontend confirms with the user
    first, same as the audio Stage B does for Suno."""
    job_dir = _job_dir(job_id)
    brief_path = job_dir / "visual_brief.json"
    png_path = job_dir / "painting.png"
    if not png_path.exists() or not brief_path.exists():
        raise HTTPException(404, "לא נמצא — צריך קודם להריץ ניתוח (דרגה A) על השמע הזה")

    with open(brief_path, encoding="utf-8") as f:
        brief = json.load(f)
    prompt = image_stage_b.build_prompt(brief)

    _image_stage_b_status[job_id] = "pending"
    threading.Thread(target=_run_image_stage_b, args=(job_id, str(png_path), prompt), daemon=True).start()
    return JSONResponse({"status": "pending"})


@protected.get("/api/image_stage_b/{job_id}/status")
def image_stage_b_status(job_id: str):
    status = _image_stage_b_status.get(job_id)
    if status is None:
        raise HTTPException(404, "לא נמצאה בקשת דרגה B לג'וב הזה")
    if status == "done":
        return JSONResponse({"status": "done", "image_url": f"/api/image_stage_b_painting/{job_id}"})
    if status.startswith("failed:"):
        return JSONResponse({"status": "failed", "detail": status[len("failed:"):]})
    return JSONResponse({"status": "pending"})


@protected.get("/api/image_stage_b_painting/{job_id}")
def get_image_stage_b_painting(job_id: str):
    path = _job_dir(job_id) / "stage_b_painting.png"
    if not path.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(path, media_type="image/png")


@protected.post("/api/reincarnate/{job_id}")
def reincarnate(job_id: str):
    """Closes the loop: takes an already-generated Stage A WAV from an
    image->music job and runs it back through the music->image pipeline,
    producing a second painting from the music that came out of the
    first one. No external API call (both legs are free local compute),
    so this runs synchronously like /api/analyze rather than needing a
    background-thread/poll pattern. Reuses the same job_dir rather than
    minting a new job_id -- this result belongs to that job, not a new one."""
    job_dir = _job_dir(job_id)
    wav_path = job_dir / "stage_a.wav"
    if not wav_path.exists():
        raise HTTPException(404, "לא נמצא — צריך קודם להריץ ניתוח ציור על הג'וב הזה")

    try:
        audio_feats = extract_audio_features(str(wav_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        sem = audio_semantic_scores(str(wav_path), max_duration_s=MAX_AUDIO_ANALYSIS_SECONDS)
        brief = build_visual_brief(f"reincarnation-of-{job_id}", audio_feats, sem)

        png_path = job_dir / "reincarnation.png"
        stats = compose_image_stage_a(brief, str(png_path))

        with open(job_dir / "reincarnation_brief.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "image_stats": stats,
        "image_url": f"/api/reincarnation/{job_id}",
    })


@protected.get("/api/reincarnation/{job_id}")
def get_reincarnation(job_id: str):
    png_path = _job_dir(job_id) / "reincarnation.png"
    if not png_path.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(png_path, media_type="image/png")


@protected.post("/api/reincarnate_music/{job_id}")
def reincarnate_music(job_id: str):
    """The other direction's version of reincarnate() above: takes an
    already-generated painting from a music->image job and runs it back
    through the image->music pipeline, producing a second piece of music
    from the painting that came out of the first one."""
    job_dir = _job_dir(job_id)
    png_path = job_dir / "painting.png"
    if not png_path.exists():
        raise HTTPException(404, "לא נמצא — צריך קודם להריץ ניתוח שמע על הג'וב הזה")

    try:
        feats = extract_features(str(png_path))
        sem = semantic_scores(str(png_path))
        brief = build_brief(f"reincarnation-of-{job_id}", feats, sem)

        midi_path = job_dir / "reincarnation.mid"
        stats = compose_stage_a(brief, str(midi_path))

        wav_path = job_dir / "reincarnation.wav"
        render_midi_to_wav(str(midi_path), str(wav_path), pans=STAGE_A_PANS)

        with open(job_dir / "reincarnation_brief.json", "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "stage_a_stats": stats,
        "audio_url": f"/api/reincarnation_audio/{job_id}",
    })


@protected.get("/api/reincarnation_audio/{job_id}")
def get_reincarnation_audio(job_id: str):
    wav_path = _job_dir(job_id) / "reincarnation.wav"
    if not wav_path.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(wav_path, media_type="audio/wav")


@protected.get("/api/original_audio/{job_id}")
def get_original_audio(job_id: str):
    """The music->image direction's original input was never served back
    before now -- analyze_audio/analyze_itunes save it as input.wav or
    input.mp3 but nothing read it again. Needed for reincarnate_music's
    original-vs-new comparison in the UI."""
    job_dir = _job_dir(job_id)
    for ext, media_type in ((".wav", "audio/wav"), (".mp3", "audio/mpeg")):
        p = job_dir / f"input{ext}"
        if p.exists():
            return FileResponse(p, media_type=media_type)
    raise HTTPException(404, "לא נמצא")


def _job_direction(job_dir: Path) -> str | None:
    if (job_dir / "painting.png").exists():
        return "music2image"
    if (job_dir / "stage_a.wav").exists():
        return "image2music"
    return None


@protected.post("/api/publish/{job_id}")
def publish_to_gallery(job_id: str):
    """Opt-in only, triggered by a button in the results UI -- running an
    analysis is not the same as wanting it shown to strangers, so this is
    a separate explicit action from /api/analyze* itself."""
    job_dir = _job_dir(job_id)
    direction = _job_direction(job_dir)
    if direction is None:
        raise HTTPException(404, "לא נמצא — צריך קודם להריץ ניתוח על הג'וב הזה")

    brief_file = "brief.json" if direction == "image2music" else "visual_brief.json"
    with open(job_dir / brief_file, encoding="utf-8") as f:
        brief = json.load(f)

    if direction == "image2music":
        title = brief.get("source_image", "ציור")
        has_stage_b = (job_dir / "stage_b.mp3").exists()
    else:
        title = brief.get("source_audio", "שיר")
        has_stage_b = (job_dir / "stage_b_painting.png").exists()

    entry = gallery.publish(
        GALLERY_INDEX_PATH, job_id, direction, title,
        brief.get("style_idiom", ""), brief.get("vat", {}), has_stage_b,
    )
    return JSONResponse(entry)


app.include_router(protected)


@app.get("/public/audio/{job_id}.wav")
def public_audio(job_id: str):
    """Deliberately NOT behind verify_credentials — Suno's servers fetch
    this URL directly as the cover-generation reference and can't present
    our Basic Auth credentials. The access control here is job_id's own
    randomness (uuid4().hex[:12], 48 bits) rather than a login: acceptable
    for what this is (a few seconds of AI-generated instrumental music,
    not sensitive data), not something to reuse for anything higher-stakes."""
    wav_path = _job_dir(job_id) / "stage_a.wav"
    if not wav_path.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(wav_path, media_type="audio/wav")


# The gallery is a read-only showcase of already-generated, explicitly-
# published artifacts — deliberately unauthenticated (same access-control
# reasoning as /public/audio above: nothing here triggers new compute or
# spends anything, it just serves files that already exist and that the
# site owner chose to publish). The compute-triggering routes above this
# point (everything under `protected`) stay behind auth exactly as before.
_GALLERY_MEDIA_TYPES = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
}


def _serve_by_prefix(job_dir: Path, prefix: str):
    for ext, media_type in _GALLERY_MEDIA_TYPES.items():
        p = job_dir / f"{prefix}{ext}"
        if p.exists():
            return FileResponse(p, media_type=media_type)
    return None


@app.get("/api/gallery")
def get_gallery():
    return JSONResponse({"entries": gallery.list_entries(GALLERY_INDEX_PATH)})


@app.get("/public/gallery/{entry_id}/input")
def gallery_input(entry_id: str):
    entry = gallery.get_entry(GALLERY_INDEX_PATH, entry_id)
    if not entry:
        raise HTTPException(404, "לא נמצא")
    resp = _serve_by_prefix(_job_dir(entry["job_id"]), "input")
    if not resp:
        raise HTTPException(404, "לא נמצא")
    return resp


@app.get("/public/gallery/{entry_id}/output")
def gallery_output(entry_id: str):
    entry = gallery.get_entry(GALLERY_INDEX_PATH, entry_id)
    if not entry:
        raise HTTPException(404, "לא נמצא")
    job_dir = _job_dir(entry["job_id"])
    name = "stage_a.wav" if entry["direction"] == "image2music" else "painting.png"
    p = job_dir / name
    if not p.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(p, media_type=_GALLERY_MEDIA_TYPES[p.suffix])


@app.get("/public/gallery/{entry_id}/output_b")
def gallery_output_b(entry_id: str):
    entry = gallery.get_entry(GALLERY_INDEX_PATH, entry_id)
    if not entry:
        raise HTTPException(404, "לא נמצא")
    job_dir = _job_dir(entry["job_id"])
    name = "stage_b.mp3" if entry["direction"] == "image2music" else "stage_b_painting.png"
    p = job_dir / name
    if not p.exists():
        raise HTTPException(404, "לא נמצא")
    return FileResponse(p, media_type=_GALLERY_MEDIA_TYPES[p.suffix])


@app.get("/gallery")
def gallery_page():
    return FileResponse(STATIC_DIR / "gallery.html")


# mounted last so it doesn't shadow the routes above
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

"""
Coloratura — Stage A web MVP.

FastAPI wrapper around the existing pipeline (feature_extraction +
semantic_features + mapping_engine + stage_a + synth). No per-call cost
(everything is local CV + CLIP + our own harmony engine + our own
synthesizer) — but CLIP inference isn't free CPU-wise either, so every
route sits behind HTTP Basic Auth (see verify_credentials below) to keep
this from being an open door if it's ever reachable beyond localhost.
Stage B (Suno) stays a separate, deliberately-manual CLI step
(run_stage_b.py) specifically because it also costs real money per call.

Run with: uvicorn webapp:app --reload --app-dir src
"""

from __future__ import annotations

import secrets
import shutil
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles

from env_config import get_env
from feature_extraction import extract_features
from mapping_engine import build_brief
from semantic_features import semantic_scores
from stage_a import compose_stage_a
from synth import STAGE_A_PANS, render_midi_to_wav

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
OUTPUT_DIR = APP_DIR.parent / "output" / "webapp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB — generous for a photographed/scanned painting

security = HTTPBasic()
_AUTH_USER = get_env("WEBAPP_USER") or "admin"
_AUTH_PASSWORD = get_env("WEBAPP_PASSWORD")
if not _AUTH_PASSWORD:
    # no hardcoded default credential ships in the repo — generate one per
    # run instead, the way tools like Jenkins print an initial admin
    # password on first boot. Set WEBAPP_USER/WEBAPP_PASSWORD in .env for a
    # password that survives a restart.
    _AUTH_PASSWORD = secrets.token_urlsafe(12)
    print(
        "\n  No WEBAPP_PASSWORD set in .env — generated one for this run:\n"
        f"    user:     {_AUTH_USER}\n"
        f"    password: {_AUTH_PASSWORD}\n"
        "  Set WEBAPP_USER / WEBAPP_PASSWORD in .env for a stable login.\n"
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


# dependency on the app itself, not per-route, so every current AND future
# route (including anything later added under /static) requires auth by
# default rather than by remembering to annotate each one.
app = FastAPI(title="Coloratura", dependencies=[Depends(verify_credentials)])


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/analyze")
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
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"שגיאה בעיבוד: {e}") from e

    return JSONResponse({
        "job_id": job_id,
        "brief": brief,
        "stage_a_stats": stats,
        "audio_url": f"/api/audio/{job_id}",
    })


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str):
    # job_id comes straight from uuid4().hex[:12] server-side and is never
    # echoed back into a path from user input beyond this lookup, but keep
    # the check anyway — cheap insurance against path traversal.
    if not job_id.isalnum():
        raise HTTPException(400, "job id לא תקין")
    wav_path = OUTPUT_DIR / job_id / "stage_a.wav"
    if not wav_path.exists():
        raise HTTPException(404, "לא נמצא — ייתכן שהג'וב פג תוקף")
    return FileResponse(wav_path, media_type="audio/wav")


# mounted last so it doesn't shadow the /api/* routes above
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

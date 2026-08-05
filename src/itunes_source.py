"""
Coloratura — iTunes Search API integration (the Spotify substitute).

Spotify's Web API killed both preview_url (30s clips) and the audio-
features/audio-analysis endpoints for any app registered after
2024-11-27 — confirmed live against Spotify's own developer community
threads, not assumed from memory. Apple's iTunes Search API is the
legitimate replacement found in its place: public, unauthenticated, no
developer registration or API key, and still returns a real previewUrl
for the vast majority of commercially released tracks — verified live
here too (a real search + a real preview download, both actually
succeeded, before this module was written to depend on it).

The 30-second preview length is not an Apple-specific limitation — it's
the industry-standard preview-rights convention baked into essentially
every major platform's licensing deal with record labels (Spotify's now-
dead previews were the same 30s; so are Deezer's, Amazon Music's). There
is no free, legal way to get more than that for copyrighted commercial
audio. Longer analysis (up to webapp.py's own MAX_AUDIO_ANALYSIS_SECONDS)
is only meaningful for content the user actually holds rights to — that's
what the plain file-upload mode is for.

Preview clips come back as AAC in an M4A container, which soundfile
(libsndfile) cannot decode (see audio_features.py) — download_preview_as_mp3
shells out to ffmpeg to convert, the same tool used for the equivalent
one-off local conversion done earlier in this project. Requires ffmpeg to
be installed wherever this runs (see Dockerfile).
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.parse
import urllib.request

SEARCH_URL = "https://itunes.apple.com/search"
# iTunes' API 403s a default urllib UA the same way Suno's CDN did (see
# stage_b.py) — reusing a realistic browser UA rather than rediscovering
# the same fix twice.
_BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


class ITunesError(Exception):
    pass


def search_tracks(query: str, limit: int = 8) -> list[dict]:
    params = urllib.parse.urlencode({"term": query, "media": "music", "entity": "song", "limit": limit})
    req = urllib.request.Request(f"{SEARCH_URL}?{params}", headers={"User-Agent": _BROWSER_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise ITunesError(f"שגיאת חיפוש ב-iTunes: {e}") from e

    results = []
    for r in data.get("results", []):
        preview_url = r.get("previewUrl")
        if not preview_url:
            continue  # a real minority of tracks (mostly classical/catalog gaps) have none
        results.append({
            "track_id": r.get("trackId"),
            "track_name": r.get("trackName", ""),
            "artist_name": r.get("artistName", ""),
            "collection_name": r.get("collectionName", ""),
            "artwork_url": r.get("artworkUrl100", ""),
            "preview_url": preview_url,
            "duration_ms": r.get("trackTimeMillis"),
        })
    return results


def download_preview_as_mp3(preview_url: str, out_mp3_path: str) -> None:
    if not preview_url.startswith("https://"):
        raise ITunesError("preview_url לא תקין")

    req = urllib.request.Request(preview_url, headers={"User-Agent": _BROWSER_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            m4a_bytes = resp.read()
    except Exception as e:
        raise ITunesError(f"שגיאת הורדה מ-iTunes: {e}") from e

    tmp_m4a = out_mp3_path + ".tmp.m4a"
    with open(tmp_m4a, "wb") as f:
        f.write(m4a_bytes)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_m4a, "-ar", "44100", "-q:a", "2", out_mp3_path],
            check=True, capture_output=True, timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise ITunesError(f"שגיאת המרה ל-MP3: {e.stderr.decode(errors='replace')[:300]}") from e
    except FileNotFoundError as e:
        raise ITunesError("ffmpeg לא מותקן על השרת") from e
    finally:
        if os.path.exists(tmp_m4a):
            os.remove(tmp_m4a)

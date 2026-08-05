"""
Coloratura — persistent gallery index (idea #4 of the 5 substantial
improvements the user asked for). A JSON-file-backed list of published
creations, living on the same persistent disk as job output (see
webapp.py's PERSISTENT_DIR / GALLERY_INDEX_PATH) so it survives redeploys.

Deliberately opt-in per job (a "פרסמו בגלריה" button in the results UI,
not automatic publishing of every analysis) — running an analysis is not
the same as wanting it shown to strangers.

No database, no locking beyond a single read-modify-write per call. A
personal portfolio project's actual publish traffic (the site owner
clicking a button occasionally) will never produce real concurrent writes;
building for load this will never see would be the kind of complexity this
project has deliberately avoided elsewhere (see e.g. _stage_b_tasks' own
plain-dict-is-enough reasoning in webapp.py).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _read_index(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_index(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def publish(index_path: Path, job_id: str, direction: str, title: str, style_idiom: str, vat: dict, has_stage_b: bool) -> dict:
    """direction: 'image2music' | 'music2image'."""
    entries = _read_index(index_path)
    entry = {
        "id": uuid.uuid4().hex[:10],
        "job_id": job_id,
        "direction": direction,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "style_idiom": style_idiom,
        "vat": vat,
        "has_stage_b": has_stage_b,
    }
    entries.append(entry)
    _write_index(index_path, entries)
    return entry


def list_entries(index_path: Path) -> list[dict]:
    return sorted(_read_index(index_path), key=lambda e: e["created_at"], reverse=True)


def get_entry(index_path: Path, entry_id: str) -> dict | None:
    for e in _read_index(index_path):
        if e["id"] == entry_id:
            return e
    return None

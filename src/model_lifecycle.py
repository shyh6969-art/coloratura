"""
Coloratura — shared idle-eviction registry for the two big lazy-loaded
models (CLIP in semantic_features.py, CLAP in audio_semantic.py).

Why this exists: measured directly (not assumed) that a real forward pass
through both models in the same process pushes RSS to ~1.36GB (878MB after
CLIP alone, +480MB more once CLAP also runs), against a 2GB Render Standard
instance — only ~640MB left for FastAPI, request buffers, and Stage A image
rendering. Real risk under any traffic that touches both directions (the
round-trip "reincarnation" feature does exactly this on purpose), not a
hypothetical one, once both models have been used at least once in this
process's lifetime, since neither was ever evicted before.

The fix is deliberately NOT "evict the other model every time" — reincarnate()
in webapp.py chains an image-side and audio-side model call within the same
user flow (upload image -> Stage A music -> click reincarnate -> Stage A
painting from that music), so an aggressive per-request evict would force a
multi-second reload delay into the middle of the app's flagship feature. A
15-minute idle grace period comfortably covers any single session (including
waiting on Stage B, which alone can take up to ~160s) while still reclaiming
memory once a process has genuinely stopped using one direction — e.g. a
different visitor later hitting only the other endpoint.

This lives in its own module (rather than each of semantic_features.py and
audio_semantic.py importing the other) to avoid a two-way import between
them for what both already treat as a fully independent, swappable model.
"""

from __future__ import annotations

import gc
import time
from typing import Callable

_IDLE_EVICT_SECONDS = 15 * 60

_registry: dict[str, dict] = {}


def register(name: str, evict_fn: Callable[[], None]) -> None:
    """Call once, right after a model is first loaded."""
    _registry[name] = {"last_used": time.time(), "evict_fn": evict_fn}


def touch(name: str) -> None:
    """Call on every use (including the one right after register())."""
    if name in _registry:
        _registry[name]["last_used"] = time.time()


def evict_idle_others(except_name: str) -> None:
    """Call right before loading/using `except_name`'s model — evicts any
    OTHER registered model that's sat idle past the grace period, freeing
    its memory before the potentially-second model gets warmed up."""
    now = time.time()
    for name in list(_registry):
        if name == except_name:
            continue
        entry = _registry[name]
        if now - entry["last_used"] > _IDLE_EVICT_SECONDS:
            entry["evict_fn"]()
            del _registry[name]
            gc.collect()

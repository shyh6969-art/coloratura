"""
Coloratura — shared .env reading.

Minimal parser (no python-dotenv dependency) — just enough to read
KEY=value lines from a .env file at the repo root. Used instead of relying
solely on os.environ because the harness these scripts often run under
doesn't reliably keep exported shell variables across separate command
invocations, but a file on disk is always there. See .env.example.
"""

from __future__ import annotations

import os

ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def read_env_file(key_name: str) -> str | None:
    if not os.path.exists(ENV_FILE):
        return None
    with open(ENV_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key_name:
                return v.strip().strip('"').strip("'") or None
    return None


def get_env(key_name: str) -> str | None:
    """Environment variable first, then .env file fallback."""
    return os.environ.get(key_name) or read_env_file(key_name)

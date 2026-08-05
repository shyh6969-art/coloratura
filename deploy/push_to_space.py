"""
Coloratura — push the Stage A web MVP to a Hugging Face Space.

Builds a clean, minimal copy (Dockerfile, requirements.txt, src/, and the
Space-flavored README with its required YAML frontmatter) in a temp
directory, then pushes it as the Space's entire git history — kept
completely separate from this repo's own git history and README, since
the Space's README needs HF-specific frontmatter this repo's own
Hebrew-language README has no use for.

Requires HF_USERNAME and HF_TOKEN in .env (create a token with the
"write" role at https://huggingface.co/settings/tokens) and an existing
Space named "coloratura" (Docker SDK) already created under that account.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from env_config import get_env  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPACE_NAME = "coloratura"


def main():
    username = get_env("HF_USERNAME")
    token = get_env("HF_TOKEN")
    if not username or not token:
        print("Set HF_USERNAME and HF_TOKEN in .env first (see .env.example).")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        shutil.copy(ROOT / "Dockerfile", tmp / "Dockerfile")
        shutil.copy(ROOT / "requirements.txt", tmp / "requirements.txt")
        shutil.copytree(ROOT / "src", tmp / "src")
        shutil.copy(ROOT / "deploy" / "space_readme.md", tmp / "README.md")

        def run(*args: str) -> None:
            subprocess.run(args, cwd=tmp, check=True)

        run("git", "init", "-q")
        run("git", "checkout", "-q", "-b", "main")
        run("git", "add", ".")
        run("git", "-c", "user.email=deploy@coloratura.local", "-c", "user.name=coloratura-deploy",
            "commit", "-q", "-m", "Deploy Stage A web MVP")
        remote = f"https://{username}:{token}@huggingface.co/spaces/{username}/{SPACE_NAME}"
        run("git", "push", "-f", remote, "main")

    print(f"Pushed. Space should build at: https://huggingface.co/spaces/{username}/{SPACE_NAME}")


if __name__ == "__main__":
    main()

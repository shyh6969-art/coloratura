"""
Coloratura — Stage B runner.

Requires SUNO_API_KEY to be set (get one at sunoapi.org). Not part of
run_test.py on purpose: Stage B costs money and takes 2-3 minutes per
painting per the sunoapi.org docs, so it shouldn't fire on every pipeline
run the way the free, instant lite/Stage A renders do.

Usage:
    SUNO_API_KEY=sk-... python src/run_stage_b.py                 # all 5
    SUNO_API_KEY=sk-... python src/run_stage_b.py munch_the_scream  # one
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from stage_b import request_cover, poll_until_done, download_audio, SunoConfigError, SunoAPIError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "output")
STAGE_B_DIR = os.path.join(OUT_DIR, "wav", "stage_b")

# raw.githubusercontent.com URLs for the Stage A reference renders already
# committed under assets/ — see assets/stage_a_wav/
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/shyh6969-art/coloratura/master/assets/stage_a_wav"

STEMS = ["van_gogh_starry_night", "kandinsky_composition_8", "mondrian_composition_ii",
         "monet_water_lilies", "munch_the_scream"]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    stems = [only] if only else STEMS

    os.makedirs(STAGE_B_DIR, exist_ok=True)
    for stem in stems:
        brief_path = os.path.join(OUT_DIR, stem + ".json")
        if not os.path.exists(brief_path):
            print(f"skip {stem}: run src/run_test.py first to generate {brief_path}")
            continue
        with open(brief_path, encoding="utf-8") as f:
            brief = json.load(f)

        reference_url = f"{GITHUB_RAW_BASE}/{stem}.wav"
        out_path = os.path.join(STAGE_B_DIR, stem + ".mp3")
        print(f"{stem}: requesting cover of {reference_url} ...")
        try:
            # done as three explicit steps (not compose_stage_b) so the
            # task_id is printed the moment it exists — a paid generation
            # that later fails only at the download step (has happened:
            # the CDN 403'd urllib's default UA) shouldn't leave the task_id
            # unrecoverable, since record-info can be re-polled for free.
            task_id = request_cover(reference_url, brief)
            print(f"  task_id={task_id} (save this — polling/download can be retried without re-paying)")
            result = poll_until_done(task_id)
            audio_url = download_audio(result, out_path)
        except SunoConfigError as e:
            print(f"  config error: {e}")
            return
        except SunoAPIError as e:
            print(f"  API error: {e}")
            continue
        print(f"  done -> {out_path} (audio_url={audio_url})")


if __name__ == "__main__":
    main()

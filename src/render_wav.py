"""Render every generated MIDI file (lite + Stage A) to WAV via synth.py."""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from synth import render_midi_to_wav, LITE_PANS, STAGE_A_PANS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIDI_DIR = os.path.join(ROOT, "output", "midi")
WAV_DIR = os.path.join(ROOT, "output", "wav")


def main():
    for tier, pans in [("lite", LITE_PANS), ("stage_a", STAGE_A_PANS)]:
        src_dir = os.path.join(MIDI_DIR, tier)
        dst_dir = os.path.join(WAV_DIR, tier)
        os.makedirs(dst_dir, exist_ok=True)
        for mid_path in sorted(glob.glob(os.path.join(src_dir, "*.mid"))):
            stem = os.path.splitext(os.path.basename(mid_path))[0]
            wav_path = os.path.join(dst_dir, stem + ".wav")
            render_midi_to_wav(mid_path, wav_path, pans=pans)
            print(f"{tier}/{stem}.wav")


if __name__ == "__main__":
    main()

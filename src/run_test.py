"""
Coloratura — prototype sanity-check runner.

Runs feature_extraction + mapping_engine on the five test paintings and
prints a compact comparison table, plus writes the full brief JSON for each
into ../output/. This is step 1-3 of the pipeline only (see spec doc,
section ד) — no generation layer yet. The point is to eyeball whether the
mapping "feels right" before investing in audio synthesis.
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from feature_extraction import extract_features
from mapping_engine import build_brief
from semantic_features import semantic_scores
from composer import compose_midi
from stage_a import compose_stage_a

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "test_images")
OUT_DIR = os.path.join(ROOT, "output")
MIDI_DIR = os.path.join(OUT_DIR, "midi", "lite")
STAGE_A_DIR = os.path.join(OUT_DIR, "midi", "stage_a")

PAINTINGS = [
    ("van_gogh_starry_night.jpg", "ואן גוך — ליל כוכבים"),
    ("kandinsky_composition_8.jpg", "קנדינסקי — קומפוזיציה 8"),
    ("mondrian_composition_ii.jpg", "מונדריאן — קומפוזיציה II"),
    ("monet_water_lilies.jpg", "מונה — נימפיאות"),
    ("munch_the_scream.jpg", "מונק — הצעקה"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MIDI_DIR, exist_ok=True)
    os.makedirs(STAGE_A_DIR, exist_ok=True)
    rows = []
    for fname, label in PAINTINGS:
        path = os.path.join(IMG_DIR, fname)
        feats = extract_features(path)
        sem = semantic_scores(path)
        brief = build_brief(fname, feats, sem)

        stem = fname.rsplit(".", 1)[0]
        out_path = os.path.join(OUT_DIR, stem + ".json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(brief, f, ensure_ascii=False, indent=2)

        compose_midi(brief, os.path.join(MIDI_DIR, stem + ".mid"))
        stage_a_stats = compose_stage_a(brief, os.path.join(STAGE_A_DIR, stem + ".mid"))

        rows.append((label, feats, brief, stage_a_stats))

    col = "{:<26}{:<7}{:<7}{:<9}{:<9}{:<22}{:<20}{:<7}{:<10}{:<10}"
    print(col.format("ציור", "Val", "Aro", "Ten-form", "Ten-emo", "מודוס/סגנון", "מקצב/מטר", "טמפו",
                      "StageA-N", "||-viol"))
    print("-" * 160)
    for label, feats, brief, sa in rows:
        v = brief["vat"]
        print(col.format(
            label,
            v["valence"], v["arousal"], v["tension_formal"], v["tension_emotional"],
            brief["style_idiom"][:20],
            brief["meter"],
            brief["tempo_bpm"],
            sa["n_chords"],
            sa["parallel_violations"],
        ))

    print("\n--- פירוט מלא ---\n")
    for label, feats, brief, sa in rows:
        print(f"### {label}")
        print(f"  VAT (blended):  {brief['vat']}")
        print(f"  VAT (pixel):    {brief['engine_notes']['vat_pixel_only']}")
        print(f"  VAT (semantic): {brief['engine_notes']['vat_semantic_only']}")
        print(f"  style: pixel-heuristic={brief['engine_notes']['style_pixel_heuristic']} "
              f"-> semantic (used)={brief['style_idiom']}")
        print(f"  semantic style distribution: {brief['engine_notes']['style_semantic_scores']}")
        print(f"  key: {brief['key']}   tempo: {brief['tempo_bpm']}bpm   meter: {brief['meter']}")
        print(f"  form: {brief['form']}")
        print(f"  harmonic_complexity: {brief['harmonic_complexity']}")
        print(f"  harmonic_resolution: {brief['harmonic_resolution']}")
        print(f"  engine_params: {brief['engine_params']}")
        stem = brief["source_image"].rsplit(".", 1)[0]
        print(f"  midi (lite):    output/midi/lite/{stem}.mid")
        print(f"  midi (stage A): output/midi/stage_a/{stem}.mid  "
              f"[{sa['n_chords']} chords, {sa['duration_sec']}s, cadence={sa['cadence']}, "
              f"parallel violations={sa['parallel_violations']}]")
        print(f"  instrumentation: {brief['instrumentation']}")
        print(f"  register: {brief['register_range']}   articulation: {brief['articulation']}")
        print(f"  climax_position: {brief['climax_position']}   dynamic_curve: {brief['dynamic_curve']}")
        print(f"  style_idiom: {brief['style_idiom']} (pixel runner-up: {brief['engine_notes']['style_pixel_runner_up']})")
        rf = brief["raw_features"]
        print(f"  raw: brightness={rf['brightness']:.2f} saturation={rf['saturation']:.2f} "
              f"temp={rf['color_temperature']:.2f} hue_var={rf['hue_variety']:.2f} "
              f"clash={rf['color_clash']:.2f} line_density={rf['line_density']:.3f} "
              f"thickness={rf['line_thickness']:.2f} angularity={rf['line_angularity']:.2f} "
              f"comp_density={rf['composition_density']:.2f} symmetry={rf['symmetry']:.2f} "
              f"neg_space={rf['negative_space_ratio']:.2f} movement={rf['movement']['label']}")
        print()


if __name__ == "__main__":
    main()

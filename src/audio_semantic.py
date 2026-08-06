"""
Coloratura — semantic feature layer for audio (CLAP), music -> painting direction.

The mirror of semantic_features.py: audio_features.py's tempo/key/spectral
numbers are useful but have the same kind of blind spot pixel features had
before CLIP was added — a fast, bright-spectrum track and a fast, harsh one
can share every signal-processing statistic while sounding completely
different in mood. This module adds zero-shot semantic understanding via
CLAP (the audio-text analog of CLIP, trained the same contrastive way),
scored against the same valence/arousal/tension prompt pairs used on the
painting side, so both directions of the pipeline share one emotional
coordinate system.

Style classification works differently here than on the image side. CLIP
was trained on real image-caption pairs, so asking it "does this look like
an Impressionist painting" is squarely in its native domain. CLAP was
trained on audio-caption pairs (sound effects, music genre/mood/instrument
descriptions) — it has never seen the word "Impressionism" paired with
anything, so scoring "this sounds like an Impressionist painting" directly
would be an out-of-domain guess dressed up as a measurement. Instead this
module classifies the music's own idiom (soft/blended vs. dissonant/atonal
vs. repetitive-minimalist, etc — the musical correlates
mapping_engine.py already reasons FROM art style TO, per the spec doc's
section ז) using prompts CLAP can actually judge. visual_mapping_engine.py
reuses that same style table in reverse to land on an art movement — same
lookup table, opposite direction, and each half of the pipeline only asks
its model questions it can actually answer.

CLAP's own checkpoint (laion/clap-htsat-unfused) caps a single forward pass
at 10 seconds of audio and expects 48kHz mono, so long tracks are chunked
into consecutive 10s windows, embedded independently, and mean-pooled (then
renormalized) rather than fed just one random crop — a full Stage B render
can run 60-160s, and a single 10s window would be a coin-flip sample of the
piece's mood rather than a summary of the whole thing.

UPDATE (real-data calibration pass, superseding the finding below rather
than deleting it — the original diagnostic and reasoning were sound given
what it tested, it just tested too narrow a sample): re-ran the same
pairwise-similarity check against 34 real, diverse, commercially produced
tracks (classical through metal through hip-hop; gathered via
itunes_source.py into output/audio_reference_large) instead of 4 of this
project's own Stage A renders. Pairwise cosine similarity spread
0.115-0.969 (median 0.503) — genuine discrimination, nothing like the
0.89-0.97 near-collapse below. style_bucket also stopped degenerating to
one answer: 5 of the 7 buckets won as the top pick across the sample, with
sensible correlations (all three metal tracks landed on
אקספרסיוניזם/אבסטרקט-גסטורלי, the two most intense/dissonant buckets in
the set — not a coincidence). Conclusion: compounding cause (2) below was
the dominant one, not (1) — CLAP itself reads real music's mood/style
reasonably well; the original collapse was a property of this project's
own narrow-timbre synthesized audio, not evidence that CLAP is generally
weak here. visual_mapping_engine.py's weights were raised back
accordingly. The live sequencer (also synth.py-style oscillators) likely
still shares Stage A's narrow-timbre weakness and wasn't specifically
retested — flagged as a residual gap, not silently assumed fixed by
association.

Original finding, kept for the record: found the same way the tempo
octave-error was — by testing against this project's own audio rather than
assuming a model built for a painting-side counterpart transfers cleanly:
run against 4 distinct Stage A renders (Munch/Mondrian/Kandinsky/Monet —
deliberately different moods), the four *audio* embeddings' pairwise cosine
similarity came back at 0.89-0.97. That's embedding collapse — CLAP was
barely telling these pieces apart before a single text prompt was even
scored, which was also why style_bucket kept landing on "ריאליזם"
regardless of input. Two candidate compounding causes were proposed at the
time: (1) CLAP's audio tower, trained mostly on general sound-event/caption
data (LAION-Audio-630k), being weaker at fine-grained instrumental-music
mood/style than CLIP is at painting style; (2) synth.py rendering every
piece with the same small set of soft, sustained instrument timbres,
narrowing the acoustic diversity CLAP had to key off in the first place.
The follow-up above resolved which one actually dominated.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import ClapModel, ClapProcessor

from audio_features import load_mono

_MODEL_NAME = "laion/clap-htsat-unfused"
_CLAP_SR = 48000
_WINDOW_S = 10

_model = None
_processor = None


def _load_clap():
    global _model, _processor
    if _model is None:
        _model = ClapModel.from_pretrained(_MODEL_NAME)
        _processor = ClapProcessor.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _model, _processor


def _resample(y: np.ndarray, sr: int, target_sr: int = _CLAP_SR) -> np.ndarray:
    if sr == target_sr:
        return y
    g = gcd(sr, target_sr)
    return resample_poly(y, target_sr // g, sr // g)


VALENCE_POS = [
    "joyful, warm, uplifting music",
    "music that sounds happy and hopeful",
    "bright, cheerful music",
]
VALENCE_NEG = [
    "somber, melancholic, sorrowful music",
    "music that sounds sad and hopeless",
    "dark, gloomy music",
]
AROUSAL_POS = [
    "turbulent, dramatic, high-energy music",
    "music full of chaotic motion and intensity",
    "frantic, agitated music",
]
AROUSAL_NEG = [
    "calm, peaceful, still music",
    "music that sounds quiet and restful",
    "serene, tranquil music",
]
TENSION_POS = [
    "ominous, anxious, unresolved music",
    "music that sounds threatening and unsettling",
    "tense, foreboding music",
]
TENSION_NEG = [
    "harmonious, balanced, resolved music",
    "music that sounds safe and settled",
    "peaceful, reconciled music",
]

# Musical correlates of the same seven idioms as semantic_features.py's
# STYLE_PROMPTS / the spec doc's section ז table — phrased as things about
# the SOUND, which is what CLAP can actually judge (see module docstring).
STYLE_PROMPTS = {
    "אימפרסיוניזם": ["impressionist orchestral music with soft blended harmony, floating ambiguous tonality and diffused timbre, like Debussy or Ravel"],
    "אקספרסיוניזם": ["atonal expressionist music with harsh dissonance, extreme wide melodic leaps and raw emotional intensity, like Schoenberg or Berg"],
    "קוביזם / אבסטרקט-גאומטרי": ["rhythmically fragmented music with irregular changing meters and abrupt angular shifts, like Stravinsky"],
    "מינימליזם": ["minimalist music built from repetitive looping patterns and slow, gradual phasing, like Steve Reich or Philip Glass"],
    "ריאליזם": ["classical tonal music with clear traditional harmonic structure and balanced, natural phrasing"],
    "סוריאליזם": ["music with unexpected dreamlike harmonic shifts, free rubato and an uncanny, disorienting quality"],
    "אבסטרקט-גסטורלי": ["free improvised music, spontaneous and chaotic with no fixed structure or steady pulse"],
}

# Per-bucket class-imbalance correction, found the same way every other
# calibration constant in this project was: by measuring, not assuming.
# The 34-track calibration sample (output/audio_reference_large) showed the
# raw softmax scores above are NOT evenly distributed across the seven
# prompts even before a winner is picked — some prompts describe qualities
# (rubato/dynamic freedom, spontaneity, dissonance) that loosely apply to
# most expressively-performed real music regardless of genre, so they
# structurally out-scored the more specific prompts (a genuinely repetitive/
# phasing piece vs. a genuinely floating/ambiguous-tonality one are rarer
# to hit exactly). Measured mean per-bucket score across all 34 songs
# (an unbiased classifier would average ~1/7=0.143 for every bucket):
# סוריאליזם 0.231, אבסטרקט-גסטורלי 0.218, אקספרסיוניזם 0.213, ריאליזם 0.179,
# קוביזם/אבסטרקט-גאומטרי 0.075, מינימליזם 0.054, אימפרסיוניזם 0.031. Two
# buckets (מינימליזם, אימפרסיוניזם) never won a single one of the 34 songs
# as a result — directly the cause of a user-reported symptom ("all songs
# turn into the same type of painting"), since visual_mapping_engine.py's
# brush_type_id/composition renderer is selected by this argmax. Dividing
# each raw score by its own measured mean (inverse-frequency reweighting,
# the same idea as class-balanced loss weighting) and picking the argmax of
# THAT is a real-data correction for the structural bias, not a random
# tie-break: re-tested against the same 34 tracks, argmax counts flattened
# from {סוריאליזם:11, אקספרסיוניזם:9, אבסטרקט-גסטורלי:6, ריאליזם:6, קוביזם:2,
# מינימליזם:0, אימפרסיוניזם:0} to {אקספרסיוניזם:7, סוריאליזם:6, ריאליזם:6,
# אבסטרקט-גסטורלי:5, מינימליזם:5, קוביזם:3, אימפרסיוניזם:2} — every bucket
# now reachable, top bucket down from 32% to 21% of the sample, and spot-
# checked reassignments stayed musically defensible (e.g. Vivaldi's "Spring"
# ritornello and "Billie Jean"'s famously repetitive bassline both moved to
# מינימליזם; metal tracks stayed on the two most intense buckets throughout).
STYLE_BUCKET_BIAS = {
    "סוריאליזם": 0.231,
    "אבסטרקט-גסטורלי": 0.218,
    "אקספרסיוניזם": 0.213,
    "ריאליזם": 0.179,
    "קוביזם / אבסטרקט-גאומטרי": 0.075,
    "מינימליזם": 0.054,
    "אימפרסיוניזם": 0.031,
}


def _embed_texts(model, processor, prompts: list[str]) -> torch.Tensor:
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_text_features(**inputs).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    mean = feats.mean(dim=0, keepdim=True)
    return mean / mean.norm(dim=-1, keepdim=True)


def _embed_audio(model, processor, y: np.ndarray, sr: int) -> torch.Tensor:
    """Mean-pooled embedding across consecutive 10s windows — see module
    docstring for why a single crop isn't used for anything but the
    shortest clips."""
    window_len = _WINDOW_S * sr
    if len(y) <= window_len:
        chunks = [y]
    else:
        n_chunks = int(np.ceil(len(y) / window_len))
        chunks = [y[i * window_len:(i + 1) * window_len] for i in range(n_chunks)]
        if len(chunks) > 1 and len(chunks[-1]) < sr:
            chunks.pop()  # trailing sliver under ~1s adds noise more than signal

    inputs = processor(audio=chunks, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        feats = model.get_audio_features(**inputs).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    mean = feats.mean(dim=0, keepdim=True)
    return mean / mean.norm(dim=-1, keepdim=True)


def _pole_score(audio_features, logit_scale, model, processor, pos: list[str], neg: list[str]) -> float:
    pos_emb = _embed_texts(model, processor, pos)
    neg_emb = _embed_texts(model, processor, neg)
    sim_pos = (audio_features @ pos_emb.T) * logit_scale
    sim_neg = (audio_features @ neg_emb.T) * logit_scale
    probs = torch.softmax(torch.cat([sim_pos, sim_neg], dim=-1), dim=-1)
    return float(probs[0, 0])


def semantic_scores(path: str, max_duration_s: float | None = None) -> dict:
    model, processor = _load_clap()
    y, sr = load_mono(path, max_duration_s)
    y = _resample(y, sr).astype(np.float32)

    with torch.no_grad():
        audio_features = _embed_audio(model, processor, y, _CLAP_SR)
        logit_scale = model.logit_scale_a.exp()

        valence = _pole_score(audio_features, logit_scale, model, processor, VALENCE_POS, VALENCE_NEG)
        arousal = _pole_score(audio_features, logit_scale, model, processor, AROUSAL_POS, AROUSAL_NEG)
        tension = _pole_score(audio_features, logit_scale, model, processor, TENSION_POS, TENSION_NEG)

        style_names = list(STYLE_PROMPTS.keys())
        style_embs = torch.cat([_embed_texts(model, processor, STYLE_PROMPTS[s]) for s in style_names], dim=0)
        sims = (audio_features @ style_embs.T) * logit_scale
        probs = torch.softmax(sims, dim=-1)[0]
        style_scores = {name: float(p) for name, p in zip(style_names, probs)}
        # best_style is picked from the bias-corrected scores (see
        # STYLE_BUCKET_BIAS above), not the raw ones — style_scores itself
        # stays raw/uncorrected in the returned dict since it's also used
        # as human-readable diagnostic context (engine_notes).
        corrected_scores = {k: v / STYLE_BUCKET_BIAS[k] for k, v in style_scores.items()}
        best_style = max(corrected_scores, key=corrected_scores.get)

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "tension": round(tension, 3),
        "style_bucket": best_style,
        "style_scores": {k: round(v, 3) for k, v in sorted(style_scores.items(), key=lambda kv: -kv[1])},
    }

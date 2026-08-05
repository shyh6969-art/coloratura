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

Known limitation, found the same way the tempo octave-error was — by
testing against this project's own audio rather than assuming a model
built for a painting-side counterpart transfers cleanly: run against 4
distinct Stage A renders (Munch/Mondrian/Kandinsky/Monet — deliberately
different moods), the four *audio* embeddings' pairwise cosine similarity
came back at 0.89-0.97. That's embedding collapse — CLAP is barely
telling these pieces apart before a single text prompt is even scored,
which is also why style_bucket kept landing on "ריאליזם" regardless of
input (its prompt text just has the highest baseline similarity to
generic music-like audio; better prompt wording can't fix a discrimination
problem that already exists on the audio side). Two likely compounding
causes, not mutually exclusive: (1) CLAP's audio tower was trained mostly
on general sound-event/caption data (LAION-Audio-630k), and is
demonstrably weaker at fine-grained instrumental-music mood/style than
CLIP is at painting style; (2) synth.py renders every piece with the same
small set of soft, sustained instrument timbres, which narrows the
acoustic diversity CLAP has to key off in the first place — real Suno
(Stage B) renders discriminate somewhat better (e.g. arousal spread
0.125-0.352 vs. Stage A's tight 0.668-0.879 cluster) but still far less
cleanly than CLIP scored paintings. Not chasing a fix here (prompt
ensembling was considered and would not address the root cause, which is
on the audio embedding side, not the text side) — instead,
visual_mapping_engine.py deliberately weights this module's output well
below audio_features.py's numeric signal-processing features (which DID
discriminate meaningfully between these same four pieces), the mirror
image of how mapping_engine.py leans on semantic_features.py, not an
equal blend.
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


def semantic_scores(path: str) -> dict:
    model, processor = _load_clap()
    y, sr = load_mono(path)
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
        best_style = max(style_scores, key=style_scores.get)

    return {
        "valence": round(valence, 3),
        "arousal": round(arousal, 3),
        "tension": round(tension, 3),
        "style_bucket": best_style,
        "style_scores": {k: round(v, 3) for k, v in sorted(style_scores.items(), key=lambda kv: -kv[1])},
    }

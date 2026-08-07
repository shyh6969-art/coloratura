"""
Coloratura — semantic feature layer (CLIP).

The hand-crafted pixel statistics in feature_extraction.py have two
structural blind spots, found during the 5-painting sanity check
(output/report.txt):

  1. Achromatic blindness — near-black/near-white pixels carry almost no
     reliable hue signal, so color_temperature / color_clash / hue_variety
     under-read paintings whose emotional charge lives in dark or void
     regions. Munch's "The Scream" was the clearest case: its near-black
     fjord against a fiery sky read as high-valence and low-tension, which
     is backwards.
  2. Micro/macro conflation — the composition-density proxy (Laplacian
     variance) measures pixel-level texture, not large-scale gestural
     energy, so broad dramatic brushwork reads as "calm" and "minimal".

Neither is fixable by tuning constants — they're structural limits of
statistics computed only from local pixel neighborhoods. This module adds a
second, independent signal with actual semantic understanding: zero-shot
CLIP scoring against emotion-word prompt pairs (mapped to the same
valence/arousal/tension space from the spec doc, section ג) and against the
seven style idioms from section ז. mapping_engine.py blends this with the
pixel features rather than replacing them — line/curve/symmetry geometry is
still something the pixel layer measures more precisely than CLIP does.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import model_lifecycle

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None


def _evict_clip():
    global _model, _processor
    _model = None
    _processor = None


def _load_clip():
    """Lazy singleton, evicted after a long idle stretch if CLAP (the audio
    side's own equally-large model) is what's actually being used right
    now — see model_lifecycle.py's docstring for the measured RSS numbers
    that motivated this."""
    global _model, _processor
    model_lifecycle.evict_idle_others(except_name="clip")
    if _model is None:
        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
        _model.eval()
        model_lifecycle.register("clip", _evict_clip)
    model_lifecycle.touch("clip")
    return _model, _processor


VALENCE_POS = [
    "a joyful, warm, uplifting painting",
    "a painting that feels happy and hopeful",
    "a bright and cheerful artwork",
]
VALENCE_NEG = [
    "a somber, melancholic, sorrowful painting",
    "a painting that feels sad and hopeless",
    "a dark and gloomy artwork",
]
AROUSAL_POS = [
    "a turbulent, dramatic, high-energy painting",
    "a painting full of chaotic motion and intensity",
    "a frantic, agitated artwork",
]
AROUSAL_NEG = [
    "a calm, peaceful, still painting",
    "a painting that feels quiet and restful",
    "a serene, tranquil artwork",
]
TENSION_POS = [
    "an ominous, anxious, unresolved painting",
    "a painting that feels threatening and unsettling",
    "a tense, foreboding artwork",
]
TENSION_NEG = [
    "a harmonious, balanced, resolved painting",
    "a painting that feels safe and settled",
    "a peaceful, reconciled artwork",
]

# Same seven idioms as the spec doc's style table (section ז).
STYLE_PROMPTS = {
    "אימפרסיוניזם": ["an impressionist painting with soft blended brushstrokes and diffused light, in the style of Monet or Renoir"],
    "אקספרסיוניזם": ["an expressionist painting with distorted forms and raw emotional intensity, in the style of Munch or Kirchner"],
    "קוביזם / אבסטרקט-גאומטרי": ["a cubist or geometric-abstract painting with fragmented angular shapes, in the style of Picasso, Braque, or Mondrian"],
    "מינימליזם": ["a minimalist painting with flat fields of color, very few elements, and a great deal of empty space"],
    "ריאליזם": ["a realist painting with accurate proportions and naturalistic detail"],
    "סוריאליזם": ["a surrealist painting with dreamlike, uncanny, impossible imagery, in the style of Dalí"],
    "אבסטרקט-גסטורלי": ["an abstract expressionist painting with spontaneous, all-over gestural brushwork and no single focal point, in the style of Pollock"],
}


def _embed_texts(model, processor, prompts: list[str]) -> torch.Tensor:
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        # this transformers version returns a BaseModelOutputWithPooling whose
        # .pooler_output has already been overwritten with the *projected*
        # embedding (see CLIPModel.get_text_features source) — not a plain
        # tensor, despite what the method's own docstring example implies.
        feats = model.get_text_features(**inputs).pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    mean = feats.mean(dim=0, keepdim=True)
    return mean / mean.norm(dim=-1, keepdim=True)


def _pole_score(image_features, logit_scale, model, processor, pos: list[str], neg: list[str]) -> float:
    pos_emb = _embed_texts(model, processor, pos)
    neg_emb = _embed_texts(model, processor, neg)
    sim_pos = (image_features @ pos_emb.T) * logit_scale
    sim_neg = (image_features @ neg_emb.T) * logit_scale
    probs = torch.softmax(torch.cat([sim_pos, sim_neg], dim=-1), dim=-1)
    return float(probs[0, 0])


def semantic_scores(path: str) -> dict:
    model, processor = _load_clip()
    image = Image.open(path).convert("RGB")
    img_inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**img_inputs).pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp()

        valence = _pole_score(image_features, logit_scale, model, processor, VALENCE_POS, VALENCE_NEG)
        arousal = _pole_score(image_features, logit_scale, model, processor, AROUSAL_POS, AROUSAL_NEG)
        tension = _pole_score(image_features, logit_scale, model, processor, TENSION_POS, TENSION_NEG)

        style_names = list(STYLE_PROMPTS.keys())
        style_embs = torch.cat([_embed_texts(model, processor, STYLE_PROMPTS[s]) for s in style_names], dim=0)
        sims = (image_features @ style_embs.T) * logit_scale
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

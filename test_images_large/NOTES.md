# Reference dataset for the calibration pass — collection notes

25 images across 7 style buckets, all downloaded from Wikimedia Commons via
its public API, each individually verified against Commons' own license
metadata before inclusion (not assumed from "the artist is old").

## Per-bucket counts (target was ~5 each)

- אימפרסיוניזם (Impressionism): 5
- ריאליזם (Realism): 5
- אקספרסיוניזם (Expressionism): 4 — Emil Nolde search returned no PD match; his
  work is mostly still in copyright (d. 1956) or not yet digitized as PD on
  Commons under a findable title. Not substituted with a mismatch just to
  hit 5 — see the two real mismatches caught and removed, below.
- אבסטרקט-גסטורלי (Abstract Gestural): 4 — entirely Kandinsky + one Hilma af
  Klint. This bucket is honestly hard to fill with clean public-domain
  examples: Pollock (d. 1956) and most other canonical abstract-expressionists
  are not public domain yet.
- קוביזם / אבסטרקט-גאומטרי (Cubism): 3 (Mondrian, Malevich, Léger) — **Picasso
  and Braque, the two artists most people mean by "cubism," are NOT public
  domain** (Picasso d. 1973, Braque d. 1963 — neither clears life+70 until
  the 2030s-40s). A Commons file titled '"Guernica" by Picasso at MOMA, NYC'
  did come back tagged "Public domain," but its actual license was
  PD-Gotfryd — the *photograph's* copyright (released by photographer Bernard
  Gotfryd via the Library of Congress), not a PD-Art tag on the painting
  itself. That does not clear Guernica's own copyright, so it was downloaded,
  checked, and deleted rather than kept. This bucket is real cubism/geometric-
  abstract-adjacent work, not cubism's two most famous names.
- מינימליזם (Minimalism): 2 (Malevich's Black Square, Mondrian's Composition
  with Red, Yellow and Blue) — flagged as the hardest bucket from the start,
  confirmed. Minimalism as a movement is mostly mid-late-20th-century and
  the field's actual minimalists (Agnes Martin d. 2004, Ellsworth Kelly
  d. 2015, Ad Reinhardt d. 1967, Barnett Newman d. 1970) are all still under
  copyright. These two are the closest genuinely-public-domain approximation,
  not a clean match to the movement itself.
- סוריאליזם (Surrealism): 2 (Odilon Redon, Henri Rousseau's "The Dream") —
  Dalí is not public domain (d. 1989). A de Chirico search for "Song of Love"
  returned a completely different, unrelated painting by Edward Burne-Jones
  that happens to share the French title "Le Chant d'Amour" — caught by
  checking the artist field (it said Burne-Jones, not de Chirico) before
  including it, and it was downloaded, checked, and deleted. A Max Ernst
  search similarly returned a 15th-century Hieronymus Bosch drawing (also
  deleted). Real de Chirico/Ernst work exists on Commons but a more targeted
  search than what this pass used would be needed to find it — left as a
  gap rather than force a wrong substitute.

## Two real PD-status mistakes, caught and corrected (not just avoided)

Both of the following were downloaded by the automated search, then removed
after manual verification of their actual Commons license category (not just
the generic "LicenseShortName: Public domain" field, which can describe a
*photograph's* copyright rather than the depicted 2D artwork's):

1. `"Guernica" by Picasso at MOMA, NYC.jpg` — PD-Gotfryd (photographer's
   release), not PD-Art. Picasso is not public domain.
2. `Edward Burne-Jones Le Chant d'Amour (Song of Love).jpg` — wrong artist
   entirely, matched on title-string overlap with a de Chirico search.

The validation logic was tightened after finding these: a file is now only
trusted as PD if its Commons categories contain an artwork-level signal
(`PD-Art`, `PD-old`, `PD-Russia`, `PD-US`, or "died more than ... years ago")
— not just a generic "Public domain" license field, which the Guernica case
proved can be true for a photo while the depicted work stays in copyright.

## One cosmetic (non-legal) attribution fix

Two files' Commons "Artist" metadata field showed the *museum photographer's*
credit (Didier Descouens) rather than the painter's name, even though the
underlying paintings are genuinely public domain (Degas d. 1917, Courbet
d. 1877 — both long past any copyright term). Manifest `artist` field
corrected to the actual painter for `impressionism__edgar_degas_ballet_dancers_painting.jpg`
and `realism__gustave_courbet_painting.jpg`, with an `artist_note` explaining
the correction.

## Not done here (scope boundary)

This pass only gathers and verifies the dataset. It does not run
feature_extraction.py / semantic_features.py against these images or
re-tune any constants — that's a separate step against this manifest.

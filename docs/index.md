# SentimentSphere

Multimodal emotion recognition over text, speech, and facial expression, with
calibrated late fusion — rebuilt from a coursework project after an audit found
that most of its reported numbers could not be reproduced.

## Current state

| Modality | State | Measured |
|---|---|---|
| **Text** | trained, calibrated, served | macro F1 **0.599**, accuracy **0.639**, ECE **0.020** |
| Speech | not built — corpora not acquired | — |
| Face | not built — FER-2013 not acquired | — |
| Fusion | not built — needs two heads first | — |

Unbuilt modalities are reported as unbuilt everywhere: the CLI, the API (a
declared `501`), and the demo. None of them returns a placeholder prediction.

## Why the project exists in this form

The [v1 audit](audit.md) found four things worth fixing properly rather than
patching:

1. **Nothing was reproducible.** For each modality the shipped artifact, the
   training notebook, and the written report disagreed, with no way to tell
   which produced which number.
2. **A reported 100% accuracy was a leak** — a two-speaker corpus split at
   random, so the model memorised word identity.
3. **A deployed app ran on randomly initialised weights.** It printed an error
   about the missing file and then predicted anyway.
4. **Eight accuracy points were lost to a two-line preprocessing mismatch**
   between training and serving.

Each has a structural answer in v2, not a one-off fix:

| v1 failure | v2 mechanism |
|---|---|
| Unreproducible numbers | Frozen, fingerprinted [split manifests](reproducing.md); provenance stamped into every report |
| Split leakage | Actor- and content-disjointness asserted in CI; TESS is leave-one-speaker-out |
| Predicting from absent weights | Checksummed artifact bundles; a missing model is a `503`, never a guess |
| Train/serve skew | Preprocessing lives inside the fitted pipeline, so a caller cannot skip it |

## Start here

- [Reproducing the numbers](reproducing.md) — one command
- [Model card for the text head](model-card-text.md) — including what not to use it for
- [Design decisions](decisions.md) — why seven labels, why the gender head was removed
- [The v1 audit](audit.md) — the original findings

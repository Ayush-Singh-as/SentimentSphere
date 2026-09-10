# SentimentSphere

Multimodal emotion recognition over **text**, **speech**, and **facial
expression**, with calibrated late fusion across the three.

> **Status: v2 rebuild in progress.** The text head is trained, calibrated,
> served, and measured. Speech, face, and fusion are **not built** — the CLI,
> the API, and the demo all say so explicitly rather than returning something
> that looks like an answer. Every number below came from a frozen split and is
> reproducible with one command.
>
> - **[UPGRADE_PLAN.md](UPGRADE_PLAN.md)** — the roadmap, phase by phase, with
>   exit criteria
> - **[docs/audit.md](docs/audit.md)** — what the v1 code actually did, measured
>   rather than claimed
> - **[legacy/v1/](legacy/v1/README.md)** — v1 preserved verbatim as evidence

---

## What this is

v1 was coursework: three modalities, three notebooks, four Streamlit apps, no
shared label space, and no way to tell which numbers were real. Auditing it
found that the shipped text model was eight accuracy points below its own
capability because the apps skipped the preprocessing it was trained with, that
the speech app loaded random weights and predicted anyway, and that a reported
100% came from randomly splitting a two-speaker corpus.

v2 keeps the goal and rebuilds the substrate: one canonical label space, frozen
speaker-independent splits, provenance on every run, and per-modality
probabilities that are calibrated before they are fused. The audit is published
alongside it rather than quietly fixed — the diagnosis is the interesting part.

---

## Where it stands

| Modality | State | Measured |
|---|---|---|
| **Text** | trained, calibrated, served | macro F1 **0.599**, accuracy **0.639**, ECE **0.020** |
| Speech | not built — corpora not acquired | — |
| Face | not built — FER-2013 not acquired | — |
| Fusion | not built — needs two heads first | — |

### The text head, in full

TF-IDF (word 1–2 grams + character 3–5 grams) → class-weighted logistic
regression, temperature-scaled on dev, evaluated once on a frozen test split of
**6,169** examples.

| Metric | Value |
|---|---:|
| Accuracy | 0.6387 |
| Macro F1 (all 7 classes) | 0.5986 |
| Weighted F1 | 0.6382 |
| ECE, before calibration | 0.0236 |
| ECE, after calibration | 0.0200 |
| Fitted temperature | 0.9871 |
| p50 / p95 latency | 5.0 ms / 8.6 ms |

Per class, on the same split:

| Class | F1 | Support |
|---|---:|---:|
| joy | 0.708 | 2,087 |
| neutral | 0.669 | 275 |
| fear | 0.667 | 849 |
| sadness | 0.598 | 1,231 |
| anger | 0.596 | 753 |
| surprise | 0.572 | 803 |
| disgust | 0.381 | 171 |

Every run also sweeps the test split under seeded typo noise, reported as
degradation from the clean baseline:

| Typo rate | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 |
|---|---:|---:|---:|---:|---:|
| Macro F1 | 0.599 | 0.583 | 0.565 | 0.540 | 0.474 |
| Δ | — | −0.016 | −0.034 | −0.059 | −0.125 |

Gradual, not cliff-edged — the character n-grams are what make that true, and
calibration holds across the sweep.

**Why this is not directly comparable to v1's 71.55%.** That figure was eight
classes, scored on a split that retained exact duplicates, using a model whose
training membership is unknown. This one is seven classes on a deduplicated,
conflict-free corpus. Reading the drop as a regression would be wrong; the
populations differ. What the harness fixes is that the difference is now
*stated* instead of hidden.

**The label-noise ceiling is real and measured.** Preparing the corpus reports
it rather than quietly dropping rows:

| Finding | Rows |
|---|---:|
| Raw rows | 34,792 |
| `shame` excluded (no other modality has it) | 146 |
| Exact duplicate rows | 3,630 |
| Duplicate rows collapsed after normalization | 3,387 |
| Groups whose identical text carries conflicting labels | 51 (416 rows) |
| **Retained** | **30,843** |

Fifty-one distinct texts appear with two different gold labels. No architecture
gets those right, which is most of the gap between this number and a good one.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/Ayush-Singh-as/SentimentSphere
cd SentimentSphere && make install
```

Reproduce the reported text metrics from scratch — freezes the split, trains,
calibrates, and writes `reports/`:

```bash
make reproduce-text
```

Then predict, serve, or open the demo:

```bash
uv run sphere predict --text "I cannot believe we pulled this off"
make serve    # FastAPI on :8000, OpenAPI at /docs
make demo     # Gradio on :7860
```

`sphere labels` prints the canonical 7-class space; `--source tess` (or
`fer2013`, `meld`, `ravdess`, `crema_d`, `savee`, `text_aggregate`,
`v1_speech_conv1d`) shows how one dataset's raw vocabulary maps onto it,
including which labels are dropped and why. `sphere info` prints the provenance
stamp a run on this machine would carry.

`make help` lists every target. `make check` runs exactly what CI enforces.

---

## The API

| Route | Behaviour |
|---|---|
| `POST /v1/predict/text` | Calibrated 7-class distribution, with the model id that produced it |
| `POST /v1/predict/{audio,image,video}` | **501** — declared in the OpenAPI document, not silently absent |
| `GET /v1/models` | Which heads are installed; unavailable ones say so |
| `GET /healthz` | Liveness |

Every response carries `x-request-id`, the originating `model_id`, and whether
the confidence was calibrated. A modality with no verified artifact returns
**503** — never a guess. That rule is the direct reaction to v1's speech app,
which printed an error about missing weights and then predicted with randomly
initialised ones.

---

## What makes the numbers trustworthy

- **Frozen split manifests**, committed under [`manifests/`](manifests/). Every
  metric names the manifest fingerprint it was computed against, and re-freezing
  with a different seed is refused rather than silently overwriting.
- **Leakage is a test, not a promise.** Actor and content-hash disjointness
  across partitions is asserted in CI. TESS is split leave-one-speaker-out, with
  calibration drawn only from the training speaker.
- **Macro F1 counts zero-support classes**, so a head covering two classes
  cannot look complete over seven.
- **Calibration is reported, not assumed** — ECE before and after, plus a
  reliability diagram in every report directory.
- **Artifacts are checksummed.** A tampered or mislabelled bundle refuses to
  load instead of quietly answering.
- **Provenance on every run**: git SHA, dirty flag, config hash, dataset
  fingerprint, split fingerprint, seed report, and package versions.

---

## Layout

```
src/sentimentsphere/
  core/        labels, config, types, io, seeding, provenance
  data/        loaders, frozen splits, dataset catalog
  models/  training/     the text baseline and its trainer
  eval/        metrics, calibration, reports, robustness, v1 audit
  inference/   artifact verification, registry, predictors
  serving/     FastAPI app and schemas
apps/space/    the Gradio app + Dockerfile for the HF Space
configs/       one YAML per model variant
tests/         unit + integration + golden
legacy/v1/     v1, preserved verbatim, excluded from every quality gate
docs/          audit, model card, ADRs
```

`core/labels.py` is the load-bearing module: the only place in the codebase
where an emotion label exists as a string. Everything else imports from it.

---

## Limitations

Read [`docs/model-card-text.md`](docs/model-card-text.md) before using any of
this for anything. In short: the output is a **predicted expression category**
for a piece of text, not a read on how anyone feels; the training corpus has
measured label noise including outright contradictions; and it is
English-only, social-media-shaped text. Facial and vocal expression do not map
reliably onto internal emotional states
([Barrett et al., 2019](https://doi.org/10.1177/1529100619832930)).

---

## License

MIT — see [LICENSE](LICENSE), which also lists the per-dataset terms. Several
corpora used here are research/non-commercial only, and the v1 speech model was
adapted from third-party work; both are noted there.

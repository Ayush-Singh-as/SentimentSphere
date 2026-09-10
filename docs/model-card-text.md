# Model card — SentimentSphere text head

**Model id prefix:** `text_tfidf_lr`
**Version:** 2.0.0.dev0
**Task:** single-label classification of a text into one of seven expression categories
**License:** MIT (source). The training corpus is an aggregate whose per-source
terms are not fully documented — see [Open questions](#open-questions).

## What it is

TF-IDF features (word 1–2 grams, plus character 3–5 grams inside word
boundaries) feeding a class-weighted multinomial logistic regression, with a
single temperature parameter fitted on the development split.

Character n-grams are included so that small misspellings degrade the
prediction rather than erasing the evidence for it. Class weighting is on
because the corpus is severely imbalanced (`joy` outnumbers `disgust` by more
than 12:1).

Preprocessing — Unicode NFKC, whitespace collapse, casefolding — lives **inside
the fitted pipeline**. This is deliberate. v1's largest measured defect was an
8.2-point accuracy loss caused by training with `neattext` cleaning and then
serving raw text; a transform that ships inside the artifact cannot be skipped
by a caller.

## Intended use

Exploratory analysis of English text, and as the first calibrated component of
a planned multimodal fusion system. It is a **baseline**: it exists to be a
number that later models must beat, honestly measured.

## Out-of-scope use

Do not use this for any of the following:

- inferring what a person actually feels, their mental state, or their intent
- any decision affecting a person — hiring, moderation with consequences,
  credit, education, health, law enforcement, insurance
- clinical or safety-critical settings of any kind
- non-English text, or text materially unlike short social-media English
- claims about groups of people derived from aggregated predictions

The output names a **predicted expression category for a string**. It is not a
measurement of an internal state. Emotion categories do not map reliably onto
facial, vocal, or written expression across people and contexts
([Barrett et al., 2019](https://doi.org/10.1177/1529100619832930)).

## Label space

Seven canonical classes, fixed order:
`anger, disgust, fear, joy, neutral, sadness, surprise`.

`shame` (146 rows, 0.42%) is excluded: no other modality in this project has
it, and retaining it would mean reporting F1 over a couple of dozen test
examples. The exclusion is recorded in `core/labels.py`, not applied ad hoc.

The v1 speech model's gender-prefixed labels (`female_angry`, …) are mapped to
their emotion and the gender component is **dropped**. Shipping a gender
classifier as a side effect of emotion detection is a liability with no upside
for the stated task.

## Training data

`data/raw/text_emotion_aggregate.csv`, an aggregate of 34,792 labelled texts
with fields `Emotion` and `Text`. It carries **no source-corpus field**, so the
provenance of individual rows cannot be recovered from the file.

Preparation is fixed, documented, and reported rather than silent:

| Step | Rows |
|---|---:|
| Raw rows | 34,792 |
| `shame` excluded | 146 |
| Text empty after normalization | 1 |
| Exact duplicate rows in the source | 3,630 |
| Duplicate rows collapsed after normalization | 3,387 |
| Groups whose identical text carries conflicting labels — **dropped entirely** | 51 groups / 416 rows |
| **Retained** | **30,843** |

Deduplication is by normalized, casefolded text, and the surviving group is the
unit that gets split — so no sentence appears on both sides of the split in a
different casing.

Conflicting groups are dropped rather than resolved by majority. Choosing a
label for a text the corpus labels two ways would manufacture a certainty the
data does not contain.

## Evaluation

Split: stratified 64/16/20 train/dev/test, frozen in
`manifests/text_aggregate-v1.json` and validated by fingerprint before every
run. The model is fit on train, the temperature on dev, and the test split is
scored once.

Test set: 6,169 examples.

| Metric | Value |
|---|---:|
| Accuracy | 0.6387 |
| Macro F1 (all 7 classes) | 0.5986 |
| Weighted F1 | 0.6382 |
| Brier score | 0.4885 |
| ECE before calibration | 0.0236 |
| ECE after calibration | 0.0200 |
| Temperature | 0.9871 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| joy | 0.719 | 0.697 | 0.708 | 2,087 |
| neutral | 0.607 | 0.745 | 0.669 | 275 |
| fear | 0.656 | 0.677 | 0.667 | 849 |
| sadness | 0.611 | 0.587 | 0.598 | 1,231 |
| anger | 0.583 | 0.610 | 0.596 | 753 |
| surprise | 0.564 | 0.579 | 0.572 | 803 |
| disgust | 0.424 | 0.345 | 0.381 | 171 |

The confusion matrix and reliability diagram are written to
`reports/<run_id>/` by `make reproduce-text`.

**Calibration was already close.** The fitted temperature of 0.987 is nearly 1,
so this model was not badly overconfident to begin with — an expected property
of a class-weighted linear model, and worth stating rather than presenting the
0.0036 ECE improvement as a large win.

## Comparison to v1 — read this carefully

v1's shipped `CountVectorizer` + `LogisticRegression` was measured at 71.55%
accuracy / 68.40% macro F1. **Those numbers and these are not comparable**:

| | v1 audit | v2 baseline |
|---|---|---|
| Classes | 8 (incl. `shame`) | 7 |
| Duplicates | retained | collapsed |
| Conflicting labels | retained | dropped |
| Training membership of the test rows | unknown | disjoint by construction |
| Split | regenerated at scoring time | frozen manifest, fingerprinted |

A rescore of an old model on a newly formed holdout does not prove it never
trained on those examples. The v1 figures remain in
[`docs/audit.md`](audit.md) as historical evidence, not as a baseline this
model beat or lost to.

## Known weaknesses

- **`disgust` is the weakest class** (F1 0.381, 171 test examples). It is the
  rarest label and the most confusable with anger. Do not rely on it.
- **The label-noise ceiling binds.** 51 texts in the corpus carry contradictory
  gold labels; the true attainable accuracy on data of this quality is well
  under 1.0 and this model is not near the ceiling either.
- **Sarcasm, negation scope, and context** are not handled beyond what bigrams
  capture.
- **No demographic evaluation.** The corpus has no author metadata, so
  performance across demographic groups is **unmeasured** — which is not the
  same as fair. Treat unmeasured as unknown.
- **English only.**

## Reproducing

```bash
make install
make reproduce-text
```

Writes `reports/<run_id>/metrics.json` with the git SHA, dirty flag, config
hash, dataset fingerprint, split fingerprint, seed report, and package
versions. A dirty tree is recorded as dirty; published numbers should come from
a clean one.

## Open questions

- The aggregate's per-row source corpus and licence terms are not recorded in
  the file. `LICENSE` lists per-dataset terms for the corpora this project
  names, but the mapping from row to corpus is not recoverable here.
- Whether a cleaned-subset evaluation (excluding near-duplicates as well as
  exact ones) materially changes the ranking of models has not been measured.

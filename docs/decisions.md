# Design decisions

Short records of choices that are expensive to reverse, and what would make
each one wrong.

---

## 1. Seven canonical labels, alphabetically ordered, defined in one module

**Decision.** `anger, disgust, fear, joy, neutral, sadness, surprise`, declared
once in `core/labels.py`. Every dataset gets an explicit raw→canonical map in
that same module. No other file may contain an emotion label as a literal.

**Why.** v1 had four incompatible label spaces — text 8, TESS 7, the speech app
10 (gender × emotion), FER-2013 7 — which made fusion impossible and made every
cross-modality comparison meaningless. Alphabetical order is arbitrary but must
be *fixed*: the index is baked into every trained artifact, manifest, and
report.

**Consequence.** Reordering the labels silently invalidates every artifact. A
[golden test](reproducing.md) pins the order and a digest of every source
vocabulary, so a change has to be deliberate.

**What would change this.** Adding an eighth class shared by all modalities, or
adopting a dimensional (valence/arousal) output instead of a categorical one.

---

## 2. `shame` is excluded, and exclusion is distinct from "unknown"

**Decision.** `shame` maps to `None` with a recorded reason. An *unrecognised*
label raises `UnknownLabelError`.

**Why.** `shame` is 146 rows (0.42%) in one corpus and absent from every other
modality; keeping it means reporting F1 over a couple of dozen test examples.
But dropping unknown labels silently is how data quietly disappears — so the
two cases are deliberately different code paths.

**What would change this.** A corpus with meaningful `shame` support across
more than one modality.

---

## 3. The gender head is removed

**Decision.** The v1 speech model's `female_angry` / `male_angry` labels map to
`anger`; the gender component is dropped, not predicted.

**Why.** Shipping a gender classifier as a side effect of emotion detection is
a liability with no upside for the stated task. It invites inference about
people that the project never intended to support and cannot validate.

**What would change this.** Nothing foreseeable. If gender were ever needed it
would be an explicit, separately justified, separately consented feature.

---

## 4. Texts with conflicting labels are dropped, not resolved

**Decision.** 51 groups (416 rows) whose identical text carries two different
gold labels are removed from the dataset entirely, and the count is reported.

**Why.** The alternatives are worse. Majority vote manufactures a certainty the
data does not contain; keeping both copies guarantees the model is wrong on one
of them and pollutes any split; keeping one at random is unreproducible. And it
matters that the number is *published* — it is the measured label-noise floor
that bounds every accuracy figure in this project.

**Consequence.** v2 numbers are not comparable to v1 numbers, which retained
these rows. Every published comparison must state this.

**What would change this.** Re-annotation, or a corpus with per-row source
provenance allowing a principled precedence rule.

---

## 5. Splits are frozen manifests, committed and fingerprinted

**Decision.** Splits live in `manifests/` as JSON, tracked in git, keyed by a
content hash of each sample. Every training run validates the manifest against
the dataset fingerprint before it starts. Re-freezing with a different seed is
**refused**, not silently applied.

**Why.** v1 regenerated splits at scoring time, so no two numbers were computed
on the same data. Committing the manifest under `manifests/` rather than
`data/` matters because `data/` is broadly gitignored.

**What would change this.** Nothing; a corpus change gets a new versioned
filename (`-v2`), never an edit in place.

---

## 6. Grouped splits, and leave-one-speaker-out for TESS

**Decision.** Audio splits group by verified actor id. TESS, which has exactly
two speakers, gets two leave-one-speaker-out folds; the development set is a
content-disjoint subset of the *training* speaker only.

**Why.** This is the direct fix for v1's headline 100%: a random split over two
speakers reciting the same 200 carrier words puts the same speaker saying the
same word on both sides. Calibrating on the held-out speaker would leak the
same way, one step later.

**Consequence.** The TESS number will be low. That is the truth about a
two-speaker corpus, and it gets published as such.

---

## 7. A missing model is a refusal, never a fallback

**Decision.** Artifact bundles carry SHA-256 checksums for every file plus the
label order they were trained against. Verification failure raises
`ModelUnavailableError`. The API answers `503`; the CLI exits nonzero; the demo
says no model is installed.

**Why.** v1's speech app loaded weights that did not exist, displayed an error,
and then predicted from a randomly initialised network — producing confident
output with no relationship to its input. Failing open is worse than failing.

---

## 8. Preprocessing lives inside the fitted artifact

**Decision.** Normalization and casefolding are a `preprocessor` on the
vectorizer inside the pipeline, not a step callers are asked to remember.

**Why.** v1's single most expensive measured defect was an 8.2-point accuracy
loss from training with `neattext` cleaning and serving raw text. A transform
that ships inside the artifact cannot be skipped.

---

## 9. Macro F1 is averaged over all seven declared classes

**Decision.** `macro_f1` includes zero-support classes; `supported_macro_f1` is
reported separately alongside an explicit `unsupported_classes` list.

**Why.** A head evaluated on two of seven classes would otherwise report a
macro F1 that looks like full coverage. The definition is written into every
`metrics.json` so a reader never has to guess which convention was used.

---

## 10. Calibration is fitted on dev and reported before and after

**Decision.** One temperature per head, fitted on the development split only,
with ECE and a reliability diagram reported both ways.

**Why.** Fusing uncalibrated probabilities is the classic way to make a fusion
model worse than its best component. Reporting only the post-calibration number
would hide how much of the result came from the calibration step — for the text
head that turned out to be very little (temperature 0.987), which is worth
saying plainly rather than dressing up.

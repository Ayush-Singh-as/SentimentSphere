# v1 audit — findings of record

**Audited:** 2026-09-04, against commit `79dbd51` (the last v1 commit).
**Method:** every claim below was verified by loading the artifact, executing the
code, or reading the notebook's own saved cell outputs. Nothing was inferred
from the README. Where a number is quoted, the command that produces it is
given.

This document is the evidence base for [`UPGRADE_PLAN.md`](../UPGRADE_PLAN.md).
The plan says what v2 does; this says why. It is written to stay true after v2
ships — it describes a frozen commit, not the current tree. The code it
describes is preserved verbatim under [`legacy/v1/`](../legacy/v1/README.md).

Three findings mattered enough to become executable regression tests rather
than prose. Those are marked **[locked]** and named inline.

---

## 1. Reproducing the audit

v1 cannot run on a modern interpreter (TensorFlow publishes no wheels for
Python 3.14, which is what the machine had). The `legacy` extra exists purely
to re-score v1:

```bash
uv sync --extra legacy
```

`tf-keras` is in that extra because the shipped `.h5` files carry
`keras_version` `2.0.6` and `2.4.0`; TensorFlow ≥ 2.16 needs the Keras 2 compat
layer to open them at all. The four v1 weight files are no longer in git (see
§6) — they are published as an archive on the HF Hub, and `sphere data` fetches
them.

---

## 2. The README described a project that did not exist

| README claim | What was actually there |
|---|---|
| Text: "NLP + **LSTM**" | `CountVectorizer` → `LogisticRegression`. The BiLSTMs were trained but never shipped; checkpoints went to a Google Drive path. |
| Speech: "LSTM" | Two rival efforts. The one with an app was a **third-party Conv1D** model; the LSTM's weights file was a 7,608-byte empty shell. |
| Vision: "CNN", real-time webcam | The shipped `model.h5` was produced by **none of the notebooks in the repo**. |
| "hosted together on a unified Streamlit interface" | Four separate `app.py` files, no launcher, no shared label space. |
| `git clone .../tashir0605/SentimentSphere` | `origin` was `Ayush-Singh-as/SentimentSphere` — the documented clone command pointed at a different account. |

The README also terminated mid-code-fence: the file ended at `cd SentimentSphere`
with no closing delimiter and no install or run instructions.

**Severity: high.** Not because the claims were dishonest — they describe work
that genuinely happened — but because a reader cannot tell which sentences map
to a runnable artifact. That is the specific failure v2's reporting rules
(`UPGRADE_PLAN.md` §II.4) exist to prevent.

---

## 3. Text

### 3.1 What shipped

Loaded from `Emotion Through Text/Text Emotion/model/text_emotion.pkl`:

```
Pipeline
  cv: CountVectorizer     1-grams, lowercase, 26,163-term vocab, no stop-word filter
  lr: LogisticRegression  C=1.0, lbfgs, l2, max_iter=100
  classes_: anger disgust fear joy neutral sadness shame surprise
```

Scored on the notebook's own stratified 20 % split (n = 6,959):

| Metric | Value |
|---|---|
| Accuracy | 71.55 % |
| Macro F1 | 68.40 % |
| Weighted F1 | 71.22 % |
| Majority-class baseline | 31.74 % |

Per-class recall is skewed by imbalance: `fear` 0.82, `joy` 0.81, but `disgust`
0.47, `neutral` 0.53, `shame` 0.59.

This 71.55 % is the only number in v1 that survives scrutiny. It is the bar v2
has to clear, and Phase 1 re-derives it from a frozen split before anything is
retrained.

### 3.2 Train/serve preprocessing skew — the most expensive bug in the repo

Training applied `neattext` cleaning (`remove_userhandles`, then
`remove_stopwords`) before fitting the vectorizer. **Neither app applied it at
inference** — both passed raw text straight into the pipeline. Same model, same
test set, only the inference-time transform differs:

| Inference preprocessing | Accuracy |
|---|---|
| What the apps did (raw text) | 71.55 % |
| What the model was trained for (cleaned) | **79.75 %** |
| Cost of the skew | **−8.21 points** |

18.2 % of predictions change. This is pure loss: eight points of accuracy
discarded by a missing two-line transform. It is also the cleanest possible
argument for v2's rule that preprocessing lives in the model artifact, not in
the caller — a `Predictor` owns its own transform, so there is no second place
for it to drift out of sync.

### 3.3 The deep models all lost to the baseline

| Notebook (v1 path → preserved as) | Embedding | Classes | Best val acc |
|---|---|---|---|
| `Human_Emotion_via_text.ipynb` → [`bilstm_glove200d_8class.ipynb`](../legacy/v1/text/notebooks/bilstm_glove200d_8class.ipynb) | GloVe 200d | 8 | ~0.59 |
| `Human_Emotion_via_text_.ipynb` → [`bilstm_glove300d_7class.ipynb`](../legacy/v1/text/notebooks/bilstm_glove300d_7class.ipynb) | GloVe 300d, `mask_zero` | 7 | 0.531 @ ep 23 |
| `C_of_Human_Emotion_via_text.ipynb` → [`bilstm_best_val0593.ipynb`](../legacy/v1/text/notebooks/bilstm_best_val0593.ipynb) | GloVe | 7 | **0.593** |

Every one is worse than the 71.55 % logistic regression that actually shipped,
at roughly 500× the training cost (~30 min/epoch). All three checkpointed to
`/content/drive/MyDrive/colab_checkpoints/`, so **no deep text weights exist in
this repository** and none of these runs can be re-scored — only their saved
cell outputs remain.

Incidental: `maxlen = max([len(t) for t in df_train['Text']])` measures
**characters** (368), and the code then hard-codes padding to **631** anyway.
Sequences were padded to a magic number with no stated origin.

### 3.4 The dataset caps what any architecture can reach

[`data/raw/text_emotion_aggregate.csv`](../data/raw/text_emotion_aggregate.csv)
(34,792 rows) is a merged aggregate:

- Severe imbalance: `joy` 11,045 → `shame` 146 (0.42 %).
- 1,662 exact duplicate rows.
- **Identical texts carrying conflicting labels.** The notebook finds this
  itself: `Yes .` appears as both `neutral` and `joy`; `Why not ?` as both
  `anger` and `neutral`.

Irreducible label noise puts a ceiling on accuracy that no model breaks
through. v2 quantifies the ceiling rather than pretending it isn't there, and
`shame` is dropped — reporting F1 over 29 test examples is not reporting.

---

## 4. Speech — two incompatible efforts, neither working

### 4.1 The inherited Conv1D model

Preserved as [`app_streamlit_conv1d.py`](../legacy/v1/speech/app_streamlit_conv1d.py)
and [`conv1d_ravdess_savee_thirdparty.ipynb`](../legacy/v1/speech/notebooks/conv1d_ravdess_savee_thirdparty.ipynb).

- Conv1D over MFCCs, **10 classes = gender × 5 emotions** (`female_angry` …
  `male_sad`), reported 72.73 %.
- Notebook output paths read `C:\Users\mites\Documents\Cognitive\Final Exam\` —
  adapted from a public RAVDESS/SAVEE repo, not original work. The training
  cell is deleted, with the note *"Removed the whole training part for avoiding
  unnecessary long epochs list."*
- The app hand-rebuilds the architecture and carries the comment
  `# Change this from 256 to 128 to match the saved weights`. The notebook's
  first layer is `Conv1D(256)`; the saved `.h5` is `Conv1D(128)`. **Script and
  artifact disagree.**
- **The load path is dead.** It loads
  `saved_models/Emotion_Voice_Detection_Model.h5`; there was no `saved_models/`
  directory anywhere in the repo. The app printed an error and then **ran
  inference with randomly initialised weights** — it never stopped, and never
  stopped producing confident-looking output.
- Audio capture is `pyaudio` reading the **server's** microphone. Cannot work on
  any hosted deployment.
- Features are `np.mean(mfcc(n_mfcc=13), axis=0)` — averaging across
  *coefficients*, not across time.

**Severity: critical.** A model that fails open is worse than one that fails
closed, because nothing downstream can detect it. This is why every `sphere`
subcommand that lacks its artifact exits non-zero
([`test_unimplemented_commands_exit_nonzero`](../tests/unit/test_cli.py))
instead of degrading silently.

The gender head is dropped in v2 on top of that: shipping a gender classifier
as a side effect of emotion detection is a liability with no upside for the
stated task. **[locked]** — `test_v1_speech_gender_is_stripped` in
[`tests/unit/test_labels.py`](../tests/unit/test_labels.py).

### 4.2 The TESS LSTM reported 100 % — that is a leak, not a result

Preserved as [`tess_lstm_100pct_LEAKED.ipynb`](../legacy/v1/speech/notebooks/tess_lstm_100pct_LEAKED.ipynb).

100 % training accuracy **and** 100 % validation accuracy. The cause:

> TESS contains exactly **two speakers** — `OAF` (1,399 files) and `YAF`
> (1,400) — each reciting the **same 200 carrier words** in 7 emotions; 2,800
> files total. A random 80/20 split puts the same speaker saying the same word
> on both sides of the split. The model memorises word identity, not emotion.

Verify the speaker counts against the corpus (`data/raw/tess/` is untracked;
`sphere data` fetches it):

```bash
find data/raw/tess -name 'OAF_*.wav' | wc -l && find data/raw/tess -name 'YAF_*.wav' | wc -l
```

That prints `1399` and `1400`. The per-file label is the filename's last
underscore field, and the whole raw vocabulary across 2,800 files is
`{angry, disgust, fear, happy, neutral, ps, sad}` — seven emotions, two
speakers, 200 shared carrier words.

Consequences, all of which v2 fixes structurally rather than by convention:

- Splits are grouped by speaker (`GroupKFold`), never random, so this class of
  leak cannot be expressed.
- The saved artifact `emotion_through_speech_model.h5` is **7,608 bytes** —
  `"layers": []`, zero weight datasets. An empty shell; there is nothing to
  re-score.
- `emotion_recognition_app_final.py` (preserved as
  [`nbconvert_not_an_app.py`](../legacy/v1/speech/nbconvert_not_an_app.py)) is
  **not an app**. It is an `nbconvert` dump of the training notebook, and it
  walks `tess_data/` — a directory that never existed under that name (the real
  one is `TESS Toronto emotional speech set data/`). It would have trained on an
  empty dataframe.
- Label parsing `filename.split('_')[-1]` yields `ps` for *pleasant surprise*,
  never mapped to a canonical name. **[locked]** —
  `test_tess_ps_means_pleasant_surprise`.

---

## 5. Vision — three mutually inconsistent stories

| Source | Architecture | Reported | Artifact |
|---|---|---|---|
| Shipped `model.h5` | double-conv 32/64/128/256, **ELU** + BatchNorm, Dense 64→64→7, Keras **2.4.0** | unknown | present, 16 MB |
| [Notebook in repo](../legacy/v1/vision/notebooks/cnn_fer2013_died_epoch12.ipynb) | 64(3×3)/128(5×5)/512/512, Dense 256→512→7, ReLU, 4,478,727 params | val ≈ 0.573 | **never saved** |
| [PDF report](../legacy/v1/vision/Visual_Emotion_Recognition_report.pdf) | 3 blocks 64/128/256, Flatten 9216, Dense 512 | 66.32 train / **67.20 val**, 50 epochs | `best_model.keras` — **absent** |

Three networks, three numbers, and the file that shipped matches none of them.

- The notebook ran **12 of 48 epochs** before dying on a `steps_per_epoch`
  bug — `Your input ran out of data; interrupting training`, with epoch times
  alternating 143 s / 9 s as the generator exhausted.
- Its `ModelCheckpoint(monitor='val_acc')` **never fired**:
  `Can save best model only with val_acc available, skipping`. The Keras 3 key
  is `val_accuracy`. **The notebook was structurally incapable of saving a
  model.**
- The PDF documents deployment via `main.py` loading `best_model.keras`. The
  repo had `app.py` loading `model.h5`. The report describes code that is not
  there.

### 5.1 Label-order scramble

`flow_from_directory` sorts FER-2013's class directories alphabetically:

```
angry  disgust  fear  happy  neutral  sad  surprise
   0        1      2      3        4    5         6
```

`app.py` decoded predictions with
`['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']` — **indices
4/5/6 rotated**. Every neutral, sad, and surprise prediction came out
mislabelled; the first four were correct by luck of the alphabet.

This is called out as *likely* rather than *confirmed* for one reason:
`model.h5`'s provenance is unverifiable, so its true output order cannot be
proven from the repo alone. Phase 1 settles it in one run — if the confusion
matrix shows a 4→5→6→4 rotation, the scramble is real.

**[locked], both directions:** `test_fer2013_directory_order_is_canonical`
asserts the alphabetical order *is* canonical, and
`test_v1_visual_hardcoded_order_was_wrong` asserts v1's list disagrees at
exactly indices 4, 5, and 6. The second test is the interesting one — it fails
if someone ever "fixes" the canonical order to match v1's.

### 5.2 The output was rendered invisible

Two CSS bugs in `app.py`: `.emotion-label` set `color:#111111` on
`background:#000000`, and the instructions block set `color:#111111` on the
black `.stApp`. **The detected emotion — the single most important output —
was black on black.**

Worth recording because it is the failure no test catches and no reviewer
mentions. v2's demo gets a screenshot in CI for this reason.

### 5.3 Smaller findings

- The vendored
  [`haarcascade_frontalface_default.xml`](../legacy/v1/vision/haarcascade_frontalface_default.xml)
  was **never used** — the app read `cv2.data.haarcascades` from the OpenCV
  install instead. It is preserved anyway, since Haar-vs-RetinaFace is a
  Phase 4 comparison worth running.
- `load_model('model.h5')` used a bare relative path: broken unless `cwd`
  happened to be that directory.
- FER-2013 itself was not in the repo (the notebook expects 28,821 train /
  7,066 validation images under `images/`).

---

## 6. Repository hygiene

- **Five** top-level module directories for three modalities:
  `Emotion Through Text/`, `Emotion Through Speech/`,
  `Emotion through Visual Data/`, `emotion_through_speech/` (a *separate*
  lowercase effort), and `SentimentSphere /` — with a **literal trailing space
  in the directory name** — holding one duplicate notebook.
- Byte-identical duplicates: `text_emotion.pkl` in two places
  (`md5 1d7f58e5…`), `text_emotion_dataset_raw_.csv` in two
  (`md5 c175e122…`).
- `C_of_Human_Emotion_via_text.ipynb` also existed twice. Those two copies are
  *not* byte-identical, but the only difference across 136 cells is the
  Colab badge URL in cell 0 — and both badges point at
  `github.com/tashir0605/SentimentSphere`, corroborating §2: the notebooks were
  round-tripped through a different account's fork, which is where the wrong
  clone command came from. The `SentimentSphere /` copy was dropped; the other
  is preserved as
  [`bilstm_best_val0593.ipynb`](../legacy/v1/text/notebooks/bilstm_best_val0593.ipynb).
- **2,800 TESS `.wav` files (274 MB) committed to git.** Packed `.git` was
  **247 MB**; a deleted 7.5 MB `fer2013_emotion_model.h5` was still reachable
  in history. A fresh clone paid for all of it.
- Absent: `.gitignore`, `.gitattributes`/LFS, `LICENSE`, tests, CI,
  `pyproject.toml`, pinned dependencies.
- Root `requirements.txt` was 8 **unpinned** packages, and omitted `librosa`,
  `matplotlib`, `altair`, `seaborn`, `pyaudio`, and `h5py` — all imported by the
  code.
- `.devcontainer/devcontainer.json` ran
  `streamlit run Emotion Through Text/Text Emotion/app.py` — an **unquoted path
  containing spaces**, so the container's auto-launch could not work.
- The environment had numpy and pandas only, on **Python 3.14, which
  TensorFlow does not support**. The project could not be run as-is on the
  machine that held it.

### 6.1 What Phase 0 did about it

History was rewritten with `git filter-repo` to remove the TESS corpus, all
`*.h5`, all `*.pkl`, and `.DS_Store`: **`.git` went 248 MB → 11 MB (22×)**. The
datasets and weights now live on the HF Hub and are fetched on demand.

A full pre-rewrite bare mirror was taken first and verified with `git fsck`
before anything destructive ran. It is kept outside this repository — the
rewrite is reversible, and every artifact cited above is recoverable from it.

The guardrails that keep this from recurring are in CI, not in a contributing
guide: [`ci.yml`](../.github/workflows/ci.yml)'s `hygiene` job rejects any
tracked file over 2 MB, any tracked weight extension, and any committed
notebook output, and `check-added-large-files` blocks the same thing at commit
time.

---

## 7. The four findings that drove the plan

1. **Nothing was reproducible.** For all three modalities the shipped artifact,
   the training notebook, and the written report disagree. There was no single
   source of truth for any model. → Phase 0's provenance stamping and Phase 1's
   frozen splits exist for this.
2. **Two of three deployed models were structurally broken.** Speech loaded
   random weights and failed open; vision very likely emitted scrambled labels.
   → Fail-closed CLI, and the label space as one typed module.
3. **The one honest number (71.55 %) was beaten by a two-line fix** the apps
   didn't apply. → Preprocessing belongs inside the artifact.
4. **The one impressive number (100 %) was a leak** from randomly splitting a
   2-speaker corpus. → Speaker-grouped splits, enforced in code.

None of this is unusual for coursework, and none of it was hidden — the notebooks
record their own failures honestly in their saved outputs. What was missing was
the layer that would have caught any of it automatically. That layer is what v2
builds first.

# SentimentSphere v2 — Audit & Upgrade Plan

**Status:** Phase 0 complete; implementation of Phases 1–8 started 2026-09-10.
See the execution ledger below for current evidence. The original audit narrative
is historical and is qualified by `PROJECT_BIBLE.md` §20.
**Author:** Ayush Singh (plan drafted 2026-09-04)
**Goal:** take a coursework-era multimodal emotion project from "three notebooks in folders" to something an engineer or recruiter can click, trust, and read.

## Execution ledger (authoritative current status)

| Phase | Status | Evidence / remaining gate |
|---|---|---|
| 0 | Complete, with discovered edge cases being repaired | Existing core/CLI and 56 passing tests |
| 1 | In progress | Implement validated data inventory, frozen manifests, calibration and baseline evaluation before model claims |
| 2–5 | Pending experiments | Implement/train/evaluate each model, then measure fusion on paired data |
| 6–8 | Pending | Inference contracts, API, demo, containers, documentation, and CI |
| 9 | Optional stretch, after 0–8 | Not a prerequisite for the public demo |

### Corrections adopted before implementation

- Accuracy thresholds are experiment targets, not results that engineering can
  guarantee. A missed threshold must produce an honest report, not test-set tuning
  or a fabricated completion claim.
- Keep the frozen archive unchanged. The best text notebook actually contains an
  eight-class single LSTM; TESS outputs contain 5,600 rows; the precise leakage
  mechanism remains unverified. See the Bible for the full evidence corrections.
- Frozen manifests live under tracked `manifests/`, outside ignored `data/`.
  Text splits group normalized duplicate texts; audio splits group verified actors.
  TESS uses two leave-one-speaker-out folds, with calibration drawn only from the
  training speaker, never the held-out speaker.
- Published v1 and v2 numbers must state class set, cleaning, split, and whether
  historical training overlap is unknown. Rescoring an old model on a newly formed
  holdout does not prove it never trained on those examples.
- Fusion training needs out-of-fold or otherwise disjoint base-model predictions;
  calibration and meta-learning must not consume the final test labels.
- ONNX parity and latency are measured per architecture. Do not force unsupported
  quantization operators or promise int8 parity within 1e-3 before measuring it.
- Remote GPU access and public Space identity/credentials must be established.
  Infrastructure code and local validation can proceed while these details are
  pending; deployment is complete only after an externally verified working URL.
- Hosting assumption corrected: HF's documentation checked on 2026-09-10 says new
  Docker Spaces require a paid account plan, although CPU Basic has no hourly fee.
  Eligible free personal accounts have a separate Gradio/ZeroGPU allowance. No
  subscription or paid hardware is authorized by this plan; see
  [current Spaces documentation](https://huggingface.co/docs/hub/spaces-overview).

**Decisions already locked in:**

| Decision | Choice |
|---|---|
| Demo host | Hugging Face Spaces (Gradio + Docker SDK), weights on HF Hub |
| Training compute | Self-hosted GPU server, 48 GB VRAM |
| Scope | All three modalities rebuilt + late fusion |

---

# Part I — Audit: what is actually in the repo today

Everything below was verified against the files, not inferred from the README. This section exists because the plan's priorities fall straight out of it, and because the audit itself is publishable content (see Phase 7).

## I.1 Headline: the README describes a project that does not exist here

| README claim | Reality |
|---|---|
| "Text-Based Emotion Recognition (NLP + **LSTM**)" | Deployed artifact is `CountVectorizer` + `LogisticRegression`. The LSTM was trained but **never shipped** — its checkpoints were written to a Google Drive folder. |
| "Speech-Based Emotion Recognition (LSTM)" | Two rival efforts. The one with an app is a **third-party Conv1D** model; the LSTM one has a **7.6 KB empty weights file**. |
| "Visual Emotion Recognition (CNN)" — real-time webcam | Shipped `model.h5` was produced by **none of the notebooks in the repo**. Provenance unknown. |
| "hosted together on a unified Streamlit interface" | There is no unified interface. Four separate `app.py` files, no launcher. |
| `git clone https://github.com/tashir0605/SentimentSphere` | `origin` is `Ayush-Singh-as/SentimentSphere`. The documented clone command points at a different account. |

The README also **terminates mid-code-fence** — the file ends at `cd SentimentSphere` with no closing ``` and no install/run instructions.

## I.2 Repository hygiene

- **Five** top-level module directories for three modalities: `Emotion Through Text/`, `Emotion Through Speech/`, `Emotion through Visual Data/`, `emotion_through_speech/` (a *separate* lowercase effort), and `SentimentSphere /` — with a **literal trailing space in the directory name** — holding one duplicate notebook.
- Byte-identical duplicates: `text_emotion.pkl` in two places (`md5 1d7f58e5…`), `text_emotion_dataset_raw_.csv` in two places (`md5 c175e122…`).
- **2,800 TESS `.wav` files (274 MB) committed to git.** `.git` is **247 MB** packed. A deleted `fer2013_emotion_model.h5` (7.5 MB) is still in history.
- Absent: `.gitignore`, `.gitattributes`/LFS, `LICENSE`, tests, CI, `pyproject.toml`, pinned dependencies.
- Root `requirements.txt` is 8 **unpinned** packages and omits `librosa`, `matplotlib`, `altair`, `seaborn`, `pyaudio`, `h5py` — all of which the code imports.
- `.devcontainer/devcontainer.json` runs `streamlit run Emotion Through Text/Text Emotion/app.py` — an **unquoted path containing spaces**, so the container's auto-launch cannot work.
- Nothing is installed in the current environment (numpy + pandas only) and it is on **Python 3.14, which TensorFlow does not support**. The project cannot be run as-is on this machine.

## I.3 Text modality

**Deployed model** (`Emotion Through Text/Text Emotion/model/text_emotion.pkl`), inspected by loading it:

```
Pipeline
  cv: CountVectorizer   (1-grams, lowercase, 26,163 vocab, no stop-word filter)
  lr: LogisticRegression (C=1.0, lbfgs, penalty=l2, max_iter=100)
  classes: anger, disgust, fear, joy, neutral, sadness, shame, surprise
```

**Measured** on the notebook's own stratified 20 % split (n = 6,959):

| Metric | Value |
|---|---|
| Accuracy | **71.55 %** |
| Macro F1 | **68.40 %** |
| Weighted F1 | 71.22 % |
| Majority-class baseline | 31.74 % |

Per-class recall is badly skewed by imbalance: `fear` 0.82 and `joy` 0.81, but `disgust` 0.47, `neutral` 0.53, `shame` 0.59.

### The most expensive bug in the repo: train/serve preprocessing skew

The training code applied `neattext` cleaning — `remove_userhandles` then `remove_stopwords` — before fitting. **Neither app applies it at inference.** Both pass raw text straight into the pipeline. Same model, same test set, only preprocessing differs:

| Inference preprocessing | Accuracy |
|---|---|
| What the apps actually do (raw text) | 71.55 % |
| What the model was trained for (cleaned) | **79.75 %** |
| **Cost of the skew** | **−8.21 points** |

**18.2 % of predictions change** when the correct preprocessing is applied. This is a pure-loss bug: eight points of accuracy are being discarded by a missing two-line transform.

### The deep text models

Three notebook variants (`Human_Emotion_via_text.ipynb`, `Human_Emotion_via_text_.ipynb`, `C_of_Human_Emotion_via_text.ipynb`) train BiLSTM stacks over frozen GloVe:

| Notebook | Embedding | Classes | Best val accuracy |
|---|---|---|---|
| `Human_Emotion_via_text` | GloVe 200d | 8 | ~0.59 (10 epochs, ~30 min/epoch) |
| `Human_Emotion_via_text_` | GloVe 300d, `mask_zero` | 7 (dropped shame) | 0.531 @ epoch 23, early-stopped 28 |
| `C_of_Human_Emotion_via_text` | GloVe | 7 | **0.593** — the best deep run |

**Every one of these is worse than the 71.55 % logistic-regression baseline that actually shipped**, at roughly 500× the training cost. All three checkpointed to `/content/drive/MyDrive/colab_checkpoints/` — **no deep text weights exist in this repository.**

Also: `maxlen = max([len(t) for t in df_train['Text']])` measures **characters** (368), then the code hard-codes padding to **631** anyway — so sequences are padded to a magic number with no stated origin.

**Dataset** (`text_emotion_dataset_raw_.csv`, 34,792 rows) is a merged aggregate with severe imbalance (`joy` 11,045 → `shame` 146) plus 1,662 exact duplicates and **identical texts carrying conflicting labels** — the notebook discovers this itself (`Yes .` labelled both `neutral` and `joy`, `Why not ?` both `anger` and `neutral`). This puts a hard ceiling on achievable accuracy that no architecture will break through.

## I.4 Speech modality — two incompatible efforts, neither working

### (a) `Emotion Through Speech/` — inherited third-party model

- Conv1D stack over MFCCs, **10 classes = gender × 5 emotions** (`female_angry` … `male_sad`), reported 72.73 % accuracy.
- Notebook output paths read `C:\Users\mites\Documents\Cognitive\Final Exam\` — adapted from a public RAVDESS/SAVEE repo, not original work. The training cell is deleted ("*Removed the whole training part for avoiding unnecessary long epochs list*").
- `app.py` hand-rebuilds the architecture with the comment `# Change this from 256 to 128 to match the saved weights` — the notebook's first layer is `Conv1D(256)`, the saved `.h5` is `Conv1D(128)`. The script and the artifact disagree.
- **The load path is dead.** `app.py` loads `saved_models/Emotion_Voice_Detection_Model.h5`; there is **no `saved_models/` directory anywhere in the repo**. The app shows an error and then **runs inference with randomly initialised weights** — it never returns and never stops producing confident-looking predictions.
- Audio capture is `pyaudio` reading the **server's** microphone. This cannot work on any hosted deployment.
- Feature extraction is `np.mean(mfcc(n_mfcc=13), axis=0)` — averaging across *coefficients*, not across time, yielding a 216-step series of "mean MFCC".

### (b) `emotion_through_speech/` — the TESS LSTM

- Reports **100 % training accuracy and 100 % validation accuracy.** That is not a result, it is a leak.
- **Why:** TESS contains exactly **two speakers** — `OAF` (1,399 files) and `YAF` (1,400) — each reciting the **same 200 carrier words** in 7 emotions, 2,800 files total. A random 80/20 split places the same speaker saying the same word in both halves. The model memorises word identity, not emotion.
- The saved artifact `emotion_through_speech_model.h5` is **7,608 bytes**: `"layers": []`, **zero weight datasets**. An empty shell.
- `emotion_recognition_app_final.py` is **not an app** — it is an `nbconvert` dump of the training notebook, and it walks `tess_data/`, a directory that does not exist (the real one is `TESS Toronto emotional speech set data/`). It would train on an empty dataframe.
- Label parsing `filename.split('_')[-1]` yields `ps` for *pleasant surprise* — never mapped to a canonical name.

## I.5 Visual modality — three mutually inconsistent stories

| Source | Architecture | Result | Artifact |
|---|---|---|---|
| **Shipped `model.h5`** | Double-conv 32/64/128/256, **ELU** + BatchNorm, Dense 64→64→7, Keras **2.4.0** | unknown | present (16 MB) |
| **Notebook in repo** | 64(3×3)/128(5×5)/512/512, Dense 256→512→7, ReLU, 4,478,727 params | val acc **≈0.573** | **never saved** |
| **PDF report** | 3 blocks 64/128/256, Flatten 9216, Dense 512 | 66.32 % train / **67.20 % val**, 50 epochs | `best_model.keras` — **absent** |

Three different networks, three different numbers, and the file that ships matches none of them. Additional findings:

- The notebook ran **12 of 48 epochs** before dying on a `steps_per_epoch` bug — the log shows `Your input ran out of data; interrupting training`, with epoch times alternating 143 s / 9 s as the generator exhausted.
- Its `ModelCheckpoint(monitor='val_acc')` **never fired**: `Can save best model only with val_acc available, skipping` (the Keras 3 key is `val_accuracy`). The notebook was structurally incapable of saving a model.
- The PDF describes deployment via `main.py` loading `best_model.keras`. The repo has `app.py` loading `model.h5`. **The report documents code that is not here.**
- **Likely label-order bug:** `flow_from_directory` sorts FER-2013 classes alphabetically → `angry, disgust, fear, happy, neutral, sad, surprise`. `app.py` uses `['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']` — **indices 4/5/6 are scrambled**, so every neutral/sad/surprise prediction is mislabelled. (Flagged as *likely* only because `model.h5`'s provenance is unverifiable; the Phase 1 harness settles it in one run.)
- **Two invisible-text CSS bugs:** `.emotion-label` sets `color:#111111` on `background:#000000`, and the instructions block sets `color:#111111` on the black `.stApp`. The detected emotion — the single most important output — is rendered black-on-black.
- The vendored `haarcascade_frontalface_default.xml` is **never used**; the app reads `cv2.data.haarcascades` from the OpenCV install instead.
- `load_model('model.h5')` is a bare relative path — breaks unless `cwd` happens to be that directory.
- **FER-2013 itself is not in the repo** (the notebook expects 28,821 train / 7,066 validation images under `images/`).

## I.6 Audit summary — the four things that matter

1. **Nothing is reproducible.** For all three modalities, the shipped artifact, the training notebook, and the written report disagree. There is no single source of truth for any model.
2. **Two of three deployed models are structurally broken** — speech loads random weights; visual very likely has scrambled labels.
3. **The one honest number in the project (71.55 %) is beaten by a two-line fix** the apps don't apply.
4. **The one impressive-looking number (100 %) is a leak** caused by splitting a 2-speaker dataset at random.

None of this is unusual for coursework. All of it is fixable, and the fixes are exactly what makes the v2 story worth telling.

---

# Part II — Target architecture

## II.1 Canonical label space (do this first, everything depends on it)

Fusion is impossible without one shared label set. Today: text has 8 classes, TESS 7, the speech app 10 (gender × emotion), FER-2013 7.

**Canonical set — 7 classes:** `anger, disgust, fear, joy, neutral, sadness, surprise`

| Source label | Maps to | Rationale |
|---|---|---|
| `angry` | `anger` | normalise |
| `happy` | `joy` | normalise |
| `sad` | `sadness` | normalise |
| `ps` / `pleasant_surprise` | `surprise` | TESS naming |
| `calm` | `neutral` | RAVDESS |
| `shame` | **excluded** | 146 rows (0.42 %); no other modality has it; retaining it means reporting F1 on 29 test examples |
| `female_*` / `male_*` | strip gender | see below |

**Dropping the gender head is deliberate.** The inherited speech model predicts gender as part of the label. Shipping a gender classifier as a side effect of emotion detection is a liability with no upside for the stated task. It gets removed, and the model card says why — that reads as judgement, not omission.

Implemented once in `src/sentimentsphere/core/labels.py` as the only place label strings exist, with a unit test asserting every dataset's raw vocabulary maps into the canonical 7.

## II.2 Repository layout

```
sentimentsphere/
├── pyproject.toml            # uv-managed, pinned, Python 3.11
├── uv.lock
├── Makefile                  # data / train / eval / serve / demo / test / docker
├── README.md                 # rewritten (Phase 7)
├── LICENSE                   # MIT
├── .gitignore  .gitattributes  .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── text/{tfidf_lr,deberta_v3_base,deberta_v3_large}.yaml
│   ├── audio/{wavlm_base_plus,wavlm_large}.yaml
│   ├── vision/{convnext_tiny,vit_base}.yaml
│   └── fusion/{avg,meta_lr,attention}.yaml
├── src/sentimentsphere/
│   ├── core/        labels.py  types.py  config.py  seeding.py  registry.py
│   ├── data/        text.py  audio.py  vision.py  meld.py  splits.py  download.py
│   ├── models/      base.py  text.py  audio.py  vision.py  fusion.py
│   ├── training/    trainer.py  callbacks.py  sweep.py
│   ├── eval/        metrics.py  calibration.py  robustness.py  report.py
│   ├── inference/   predictors.py  onnx_export.py  pipeline.py
│   └── serving/     api.py  schemas.py
├── apps/
│   ├── space/       app.py  Dockerfile  README.md   # the HF Space
│   └── cli.py                                        # sphere predict --text "..."
├── tests/           unit/  integration/  golden/  fixtures/
├── notebooks/       01_eda_text.ipynb …              # EDA only, outputs stripped
├── artifacts/       # gitignored — pulled from HF Hub
├── reports/         # generated: metrics.json, confusion matrices, cards
└── docs/            # mkdocs-material → GitHub Pages
```

Every model implements one interface, which is what makes fusion and serving uniform:

```python
class EmotionModel(Protocol):
    labels: tuple[str, ...]                        # always the canonical 7
    def predict_proba(self, x: Batch) -> np.ndarray:  # (n, 7), rows sum to 1
    def to_onnx(self, path: Path) -> None: ...
```

## II.3 Model targets

Sized for a 48 GB card — none of these need gradient checkpointing or offload.

| Modality | Baseline (reproduce honestly) | Target model | Datasets |
|---|---|---|---|
| **Text** | TF-IDF + LR, class-weighted | **DeBERTa-v3-base**, then `-large` if base plateaus | existing 34.8k aggregate (for apples-to-apples before/after) + MELD text |
| **Speech** | MFCC + SVM | **WavLM-Large** + attentive statistics pooling | TESS + RAVDESS (24 spk) + CREMA-D (91 spk) + SAVEE (4 spk) |
| **Face** | the existing `model.h5`, scored properly | **ConvNeXt-Tiny** or **ViT-B/16** on aligned crops | FER-2013 (+ RAF-DB if licence obtained) |
| **Fusion** | uniform probability average | **calibrated probs → LightGBM/LR meta-learner**, plus attention-over-embeddings variant | **MELD** |

**Why MELD is the fusion dataset:** it carries text + audio + video for ~13k utterances with **exactly the canonical 7 emotion labels**, and ships official train/dev/test splits. No label remapping, no split invention, and it is freely downloadable. Every alternative (IEMOCAP needs a licence request; CREMA-D has no text) costs more and fits worse.

**Face detection replaces Haar** with RetinaFace or MediaPipe, plus 5-point alignment before the 48×48/224×224 crop. Haar is the single largest source of real-world failure in the current visual app, well ahead of the classifier.

## II.4 Honest-reporting commitments

These are load-bearing for credibility, not decoration:

- **Speaker-independent evaluation for speech, always.** Grouped splits by actor ID. The README will show the random-split number *and* the speaker-independent number side by side, precisely because the gap is the point.
- **TESS is reported as leave-one-speaker-out only.** With 2 speakers that means train on one, test on the other. The number will be low. That is the truth about a 2-speaker corpus.
- **Calibration reported, not just accuracy.** Temperature scaling per modality; ECE and reliability diagrams in `reports/`. Fusion over uncalibrated probabilities is the classic way to make a fusion model worse than its best component.
- **Construct-validity caveat stated up front.** Facial expressions do not map reliably onto emotional states (Barrett et al., 2019, *Emotional Expressions Reconsidered*). FER-2013 human accuracy is ≈65 ± 5 %, so a 67 % model is at human level on a task humans are not reliably good at. The model card frames the output as *"predicted expression category"*, not *"the person's emotion"*.
- **Label-noise ceiling acknowledged** for the text aggregate, with the conflicting-label examples shown as evidence.

---

# Part III — Phased implementation

Each phase has an exit test. Do not advance past a red exit test.

## Phase 0 — Foundation & truth (≈0.5 day)

Make the repo buildable and stop the bleeding.

1. `git tag v1-archive && git push --tags` — **freeze the current state before touching anything.**
2. Scaffold `pyproject.toml` (uv, Python 3.11 — TF/torch have no 3.14 wheels), `Makefile`, `.gitignore`, `LICENSE` (MIT), `.pre-commit-config.yaml` with `ruff`, `ruff-format`, and `nbstripout`.
3. Collapse the five module directories into `src/`. Delete `SentimentSphere /` (trailing space) and the duplicate `.pkl` / `.csv` pairs.
4. Move datasets and weights **out of git**: `make data` downloads with checksum verification; weights live on the HF Hub and are fetched by `hf_hub_download`.
5. Write `core/labels.py` (§II.1) and `core/seeding.py` (seed `random`, `numpy`, `torch`, `PYTHONHASHSEED`; log git SHA + config hash into every artifact).

**Exit test:** `make install && make test` green on a clean clone; `sphere --help` runs.

**⚠️ Requires your explicit go-ahead:** removing the 274 MB of TESS audio from git *history* (`git filter-repo`) rewrites every commit hash and requires a force-push. The alternative — leave history alone, only stop tracking new large files — keeps `.git` at 247 MB forever but is non-destructive. My recommendation: tag `v1-archive`, push it, keep a local mirror, *then* rewrite. I will not run `filter-repo` without you confirming.

## Phase 1 — Evaluation harness, before any modelling (≈1.5 days)

This is the phase that separates v2 from v1. **You cannot improve what you cannot measure, and right now nothing in the repo can be measured.**

1. `data/splits.py`: deterministic, versioned, checksummed splits. Grouped by **actor** for audio, by **utterance/dialogue** for MELD, stratified for text. Splits are written to disk as manifests and committed — not regenerated at runtime.
2. `eval/metrics.py`: accuracy, macro/weighted F1, per-class P/R/F1, confusion matrix, ECE. Everything writes `reports/<model>/metrics.json` including git SHA, config hash, and split manifest hash.
3. `eval/report.py`: renders confusion matrices, per-class bars, and reliability diagrams to `reports/`.
4. **Score every legacy artifact through the new harness** and record the numbers as the official v1 baseline: the text pickle (with *and* without the correct preprocessing), the visual `model.h5` (which resolves the label-order question definitively), and the speech Conv1D model with weights that actually load.
5. `eval/robustness.py`: audio SNR sweep, image brightness/blur/occlusion, text typo/paraphrase perturbation.

**Exit test:** `make eval-baseline` regenerates every v1 number reproducibly, and a leakage regression test asserts **zero speaker overlap** across audio splits and zero text overlap across text splits.

**Deliverable:** `reports/baseline/` — the "before" column of every table in the final README.

## Phase 2 — Text (≈1 day)

1. Reproduce TF-IDF + LR properly (class-weighted, calibrated) as the documented baseline. Expect ≈0.72 macro F1 with correct preprocessing.
2. Fine-tune **DeBERTa-v3-base**: class-weighted cross-entropy or focal loss, layer-wise LR decay, 5 seeds, mean ± std reported. Escalate to `-large` only if base plateaus.
3. Temperature-scale on dev.
4. Export ONNX + dynamic int8; verify parity within 1e-3 and record CPU latency.

**Exit test:** macro F1 **≥ 0.78** on the frozen test split (v1: 0.684), calibrated ECE **< 0.05**, ONNX parity verified.

## Phase 3 — Speech (≈2 days)

1. Fold in RAVDESS + CREMA-D + SAVEE alongside TESS (`data/download.py` with checksums). TESS alone cannot support a credible model.
2. MFCC + SVM baseline under **speaker-independent** splits — this is the number that replaces the fictional 100 %.
3. Fine-tune **WavLM-Large** with attentive statistics pooling; `GroupKFold` by actor; augmentation (SpecAugment, speed perturbation, room impulse response, additive noise).
4. Publish the **leakage demonstration** as a first-class artifact: same model, same data, random split vs. speaker-independent split, side by side.
5. Temperature-scale, export ONNX int8.

**Exit test:** speaker-independent macro F1 **≥ 0.65** across the combined corpora; leave-one-speaker-out TESS number reported however low it lands; `reports/speech/leakage_demo.json` shows both splits.

## Phase 4 — Face (≈1.5 days)

1. `make data-fer` fetches FER-2013 from Kaggle with a checksum (it is not in the repo and never should be).
2. Replace Haar with **RetinaFace/MediaPipe** detection + 5-point alignment; benchmark detection rate against Haar on the existing `test_audios`-equivalent image fixtures.
3. Fine-tune **ConvNeXt-Tiny** (and ViT-B/16 as a comparison) on aligned crops with the augmentation the PDF describes but the code never applied.
4. Add **temporal smoothing** for video/webcam — an exponential moving average over per-frame probabilities. Single-frame FER jitters badly; this is the difference between a demo that looks broken and one that looks solid.
5. Temperature-scale, export ONNX int8.

**Exit test:** FER-2013 test accuracy **≥ 0.71** (v1 notebook: 0.573; v1 PDF claim: 0.672), and the label ordering is asserted by a golden test rather than assumed.

## Phase 5 — Fusion (≈2 days)

1. MELD loader with official splits; extract per-modality calibrated probabilities for train/dev/test.
2. Three fusion strategies, ablated:
   - uniform probability average (the baseline everyone skips)
   - **learned meta-classifier** (LR / LightGBM) over concatenated calibrated probs — the recommended default
   - attention over per-modality embeddings
3. **Full ablation table** — every single modality, every pair, all three. This table is the centrepiece of the README; it is also the honest way to show whether fusion earns its complexity.
4. Missing-modality handling: the demo must degrade gracefully when a clip has no detectable face or no speech. Train the meta-learner with modality dropout so this is a supported path, not an exception.

**Exit test:** fused macro F1 beats the **best single modality** on MELD test by ≥ 2 points, ablation table complete, and inference succeeds with any subset of modalities present.

## Phase 6 — Serving (≈1.5 days)

1. `serving/api.py` — FastAPI, pydantic v2, `POST /v1/predict/{text,audio,image,video}`, `GET /healthz`, `GET /v1/models`. Request IDs, structured logging, upload size/duration caps, rate limiting, and typed error envelopes.
2. ONNX Runtime int8 everywhere — the free Space tier is CPU-only, and this is what makes it feel instant.
3. Model loading from HF Hub at container start, with a warm-up pass so the first user request isn't the one that pays for lazy init.
4. Latency table (p50/p95 per modality, 2 vCPU) into `reports/latency.json`.

**Exit test:** contract tests pass against a live container; every modality p95 **< 2 s** on 2 vCPU; OpenAPI docs render.

## Phase 7 — The demo and the story (≈2 days)

This is where "presentation to outsiders" gets built.

**7a — The HF Space** (`apps/space/`, Docker SDK):

- Gradio Blocks, five tabs: **Text · Voice · Face · Video (fusion) · How it works**
- `gr.Audio(sources=["microphone","upload"])` and `gr.Image(sources=["webcam"])` — **client-side capture**, which is what kills the `pyaudio` / `cv2.VideoCapture(0)` deployment blocker for good.
- The fusion tab is the moment: upload a short clip → transcript, voice read, face read, and the fused verdict, side by side with confidences.
- Curated examples preloaded (`gr.Examples`) so a visitor gets a result in one click without hunting for a `.wav`.
- A **Metrics tab** rendering `reports/` live — confusion matrices, the ablation table, the leakage demo. Very few portfolio demos show their own error analysis.

**7b — The README rewrite:**

- Hero GIF of the fusion tab, live-demo badge, one-line pitch
- Mermaid architecture diagram
- **Honest before/after table** (v1 audit numbers → v2 numbers, every one traceable to `reports/`)
- Three-command quickstart
- A **"What the audit found"** section — the invisible CSS, the dead model path, the 100 % leak, the 8.2-point skew
- A **"Limitations"** section — construct validity, label noise, dataset bias, what this must not be used for

**7c — Model cards** (HF format) per model: intended use, out-of-scope use, training data, per-class metrics, calibration, bias analysis across the demographic axes the corpora expose, and the gender-head removal rationale.

**7d — `docs/`** via mkdocs-material → GitHub Pages: architecture, dataset provenance, reproduction guide, ADRs for the label space / fusion strategy / gender-head decisions.

**7e — The write-up.** *"I audited my own ML project a year later: 100 % accuracy was a lie and 8 points were sitting on the floor."* The Part I findings are genuinely good technical writing material, and a post that demonstrates you can find your own leaks is worth more than one that reports a high number.

**Exit test:** a stranger reaches a working prediction in **< 30 s** from landing on the Space, with no local setup.

## Phase 8 — Engineering rigor (≈1 day, interleaved from Phase 0)

- **pytest**: label-mapping totality; split determinism; **no-leakage assertions**; feature-extraction shape/dtype contracts; golden-file inference (fixed input → fixed output within tolerance, so a silent model swap fails CI); API contract tests.
- **CI** (`.github/workflows/ci.yml`): ruff → mypy (strict on `src/`) → pytest on 3.11 + 3.12 → build Docker → smoke-run `sphere predict` on a fixture. Coverage gate ≥ 80 % on `src/`.
- **W&B** for experiment tracking, with public run dashboards linked from the README — free presentation surface.
- **Reproducibility**: `uv.lock`, seeds, git SHA + config hash in every `metrics.json`, split manifests committed.

**Exit test:** CI green on a fresh clone; `make reproduce-text` regenerates the reported text metrics within seed variance.

## Phase 9 — Stretch (only after 0–8 ship)

Streaming webcam/mic inference over WebRTC · multilingual text (XLM-R) · knowledge distillation to a <10 MB student for browser-side ONNX · valence/arousal regression alongside categorical output · a `sentimentsphere` PyPI package.

---

# Part IV — Sequencing, effort, and risk

## IV.1 Critical path

```
Phase 0 ──→ Phase 1 ──┬──→ Phase 2 (text)   ──┐
   (truth)  (harness)  ├──→ Phase 3 (speech) ──┼──→ Phase 5 ──→ Phase 6 ──→ Phase 7
                       └──→ Phase 4 (face)   ──┘   (fusion)     (serve)     (demo)
                                                Phase 8 runs alongside from Phase 0
```

Phases 2–4 are independent and can interleave with GPU queueing — while WavLM trains, write the text eval.

| Phase | Effort | Blocks |
|---|---|---|
| 0 Foundation | 0.5 d | everything |
| 1 Harness | 1.5 d | everything |
| 2 Text | 1.0 d | fusion |
| 3 Speech | 2.0 d | fusion |
| 4 Face | 1.5 d | fusion |
| 5 Fusion | 2.0 d | demo |
| 6 Serving | 1.5 d | demo |
| 7 Demo & docs | 2.0 d | — |
| 8 Rigor | 1.0 d | interleaved |
| **Total** | **≈13 focused days** | |

**Minimum viable pro — 4 days**, if you want something presentable fast: Phase 0 → Phase 1 → Phase 2 (text only) → a single-tab Space + rewritten README + honest metrics. That alone converts the project from "unreproducible" to "measured, deployed, and honestly documented," which is most of the credibility gain. Speech, face, and fusion then land as visible progress on a live demo rather than as prerequisites.

## IV.2 Risks

| Risk | Mitigation |
|---|---|
| **Fusion doesn't beat the best single modality** | Report it anyway with the ablation — a negative result honestly reported is a stronger signal than a suppressed one. MELD's text channel is known to dominate; the *finding* that audio/visual add little on MELD is publishable. |
| MELD download size / availability | Cache to HF Hub Datasets under your account; checksum in `download.py`. Fall back to CREMA-D (audio+video, no text) if MELD becomes unavailable. |
| FER-2013 needs Kaggle credentials | Document `KAGGLE_USERNAME`/`KAGGLE_KEY`; mirror to a private HF dataset for CI. |
| HF free Space is CPU-only and memory-capped | ONNX int8 for all four heads; lazy-load per tab; if the fusion path exceeds the cap, keep it upload-only (no live webcam) and document the ceiling. |
| Text ceiling from label noise | Report the noise analysis as a finding; consider a cleaned-subset evaluation alongside the full-set number. |
| History rewrite breaks external clones | `v1-archive` tag + local mirror before any `filter-repo`; **your explicit confirmation required**. |
| Scope creep across three modalities | Exit tests are gates. A red exit test blocks the next phase — that is the whole point of ordering Phase 1 before any modelling. |

## IV.3 The before/after table this plan is designed to produce

| | v1 (audited, measured) | v2 (target) |
|---|---|---|
| Text macro F1 | 0.684 *(0.72 with correct preprocessing)* | **≥ 0.78** |
| Speech | 100 % *(leaked)* / random weights in the app | **≥ 0.65 speaker-independent** |
| Face accuracy | 0.573 notebook · 0.672 claimed · labels likely scrambled | **≥ 0.71**, ordering asserted |
| Fusion | none | **beats best single modality by ≥ 2 pts** |
| Reproducible? | no — artifacts, code, and reports disagree | `make reproduce-*`, seeds + SHA in every metric |
| Deployed? | localhost only; server-side mic/webcam | **public HF Space, client-side capture** |
| Tests / CI | none | pytest + ruff + mypy + Docker in CI |
| Calibration | unmeasured | ECE reported per modality |
| Model cards | none | one per model, with limitations |
| `.git` size | 247 MB | < 5 MB |

---

# Part V — What I need from you before Phase 0

1. **Go / no-go on the git history rewrite** (Part III, Phase 0). Non-destructive alternative available.
2. **HF account/org name** for the Space and the model/dataset repos.
3. **Kaggle credentials** available for FER-2013, or should I mirror it to a private HF dataset instead?
4. **W&B**: use it (public dashboards linked from the README) or keep experiment tracking local with MLflow?
5. **Full path or the 4-day minimum-viable-pro cut** to start?

Answer 1 and 5 and I can begin; 2–4 are needed by Phase 2.

# v1 — preserved as audited evidence

This directory is the SentimentSphere code as it stood before the v2 rebuild. It
is **kept deliberately and is not maintained**. It is excluded from linting,
type-checking, formatting, and coverage.

It is here for three reasons:

1. The v2 evaluation harness scores these artifacts to produce the honest
   "before" column of every metrics table in the README.
2. The audit write-up references specific lines in this code.
3. Deleting the evidence would make the audit unverifiable.

The full pre-rewrite git history — including the 274 MB TESS corpus and the
model binaries that were removed from this repo's history — is preserved in a
separate bare mirror outside this repository. See `docs/audit.md`.

## What is in here

| Path | What it is | Why it matters |
|---|---|---|
| `text/app_streamlit.py` | The deployed text app | Feeds **raw** text to a pipeline trained on `neattext`-cleaned text. Costs 8.21 accuracy points and changes 18.2% of predictions. |
| `text/app_streamlit_older.py` | Earlier text app | Emoji dict values are all `"."`; has dead `happy`/`sad` keys absent from `pipe_lr.classes_`. |
| `text/notebooks/bilstm_glove200d_8class.ipynb` | BiLSTM + GloVe 200d, 8 classes | Peaked ~0.59 val accuracy — **worse than the logistic regression that shipped**. `maxlen` counts characters, then pads to a hardcoded 631. |
| `text/notebooks/bilstm_glove300d_7class.ipynb` | BiLSTM + GloVe 300d, 7 classes | Best val_loss 1.322 @ epoch 23. Checkpointed to Google Drive only. |
| `text/notebooks/bilstm_best_val0593.ipynb` | Best of the three deep text runs | val_accuracy ~0.593. Weights never committed. |
| `speech/app_streamlit_conv1d.py` | The deployed speech app | Loads `saved_models/…h5`, **a path that does not exist**, then predicts with randomly initialised weights. Records via server-side `pyaudio`. |
| `speech/notebooks/conv1d_ravdess_savee_thirdparty.ipynb` | Inherited Conv1D notebook | Output paths show `C:\Users\mites\…` — adapted from a third-party repo. Its first layer is `Conv1D(256)`; the shipped weights are `Conv1D(128)`. |
| `speech/notebooks/tess_lstm_100pct_LEAKED.ipynb` | TESS LSTM | Reports **100% train and 100% val accuracy**. TESS has exactly 2 speakers reciting the same 200 words, so a random split puts identical speaker+word pairs on both sides. |
| `speech/nbconvert_not_an_app.py` | Named like an app, isn't | An `nbconvert` dump of the training notebook (`# In[ ]:` markers intact) that walks a non-existent `tess_data/`. |
| `vision/app_streamlit.py` | The deployed visual app | Hardcodes label order `[…,'Sad','Surprise','Neutral']` against alphabetically-sorted FER-2013 dirs, scrambling indices 4/5/6. Renders its own output `#111111` on `#000000` — invisible. |
| `vision/notebooks/cnn_fer2013_died_epoch12.ipynb` | The visual training notebook | Ran 12 of 48 epochs before `Your input ran out of data`. Its `ModelCheckpoint(monitor='val_acc')` never fired, so it **saved nothing**. |
| `vision/Visual_Emotion_Recognition_report.pdf` | The written report | Describes a *third* architecture, 67.20% val accuracy, `best_model.keras`, and deployment via `main.py` — none of which exist in the repo. |
| `speech/test_audios/*.wav` | 8 short clips | Retained as demo fixtures and for golden-file inference tests. |

## Weights

The v1 model binaries are **not** in git — they were removed from history along
with the datasets. They are republished as a `v1-baseline` revision on the
Hugging Face Hub so the baseline stays reproducible:

| File | Size | What it is |
|---|---|---|
| `v1_text_countvec_lr.pkl` | 2.0 MB | `CountVectorizer` + `LogisticRegression`, 8 classes. The model that actually shipped. |
| `v1_vision_cnn.h5` | 16 MB | Keras 2.4.0 CNN. Matches no notebook in the repo; provenance unknown. |
| `v1_speech_conv1d.h5` | 3.5 MB | Keras 2.0.6 Conv1D, 10 gender x emotion classes. |
| `v1_tess_lstm_EMPTY.h5` | 7.6 KB | `"layers": []`, zero weight datasets. An empty shell that was presented as a trained model. |

Loading the two older `.h5` files needs the Keras 2 compatibility layer:
`make install-legacy` (installs `tensorflow-cpu` + `tf-keras`).

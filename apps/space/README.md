---
title: SentimentSphere
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# SentimentSphere

Calibrated emotion recognition over a canonical seven-label space.

Only the **text** head is trained. The Voice, Face, and Video tabs say so
rather than showing output from weights that do not exist — that failure mode
is precisely what this rebuild was written to correct.

Every prediction reports the model id that produced it and whether its
confidence was calibrated. The Metrics tab renders the project's own
measured numbers, including the classes it does badly on.

Source: <https://github.com/Ayush-Singh-as/SentimentSphere>

## Running it locally

```bash
make install
make reproduce-text   # freeze the split, train and calibrate the text head
make demo             # http://127.0.0.1:7860
```

## Deploying

The image ships no weights. Mount or fetch a verified bundle into
`SPHERE_ARTIFACTS_DIR` at deploy time; without one the app loads and reports
that no model is installed.

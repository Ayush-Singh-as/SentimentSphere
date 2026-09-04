# SentimentSphere

Multimodal emotion recognition over **text**, **speech**, and **facial
expression**, with calibrated late fusion across the three.

> **Status: v2 rebuild in progress.** This README is provisional. It describes
> only what runs today and deliberately quotes no headline metric, because the
> v2 models do not exist yet. It gets rewritten in Phase 7, once there are
> numbers that came from a frozen split on a clean tree.
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
100 % came from randomly splitting a two-speaker corpus.

v2 keeps the goal and rebuilds the substrate: one canonical label space, frozen
speaker-independent splits, provenance on every run, and per-modality
probabilities that are calibrated before they are fused. The audit is published
alongside it rather than quietly fixed — the diagnosis is the interesting part.

---

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/Ayush-Singh-as/SentimentSphere
```

```bash
cd SentimentSphere && make install
```

That installs the package plus dev tooling and the pre-commit hooks. What works
right now:

```bash
uv run sphere labels
```

Prints the canonical 7-class label space. `--source tess` (or `fer2013`,
`meld`, `ravdess`, `crema_d`, `savee`, `text_aggregate`, `v1_speech_conv1d`)
shows how one dataset's raw vocabulary maps onto it, including which labels are
dropped and why.

```bash
uv run sphere info
```

Prints the provenance stamp a run on this machine would carry: git SHA, whether
the tree is dirty, and the version of every numeric dependency.

Every other subcommand (`data`, `eval`, `train`, `predict`, `serve`) exits with
code 2 and names the phase it lands in. That is on purpose — see
[docs/audit.md §4.1](docs/audit.md) for the v1 behaviour it is reacting to.

`make help` lists every target. `make check` runs exactly what CI enforces.

---

## Layout

```
src/sentimentsphere/
  core/        labels, seeding, provenance      <- built
  data/  models/  training/  eval/  inference/  serving/   <- Phases 1-6
apps/space/    the Gradio app + Dockerfile for the HF Space   <- Phase 8
configs/       one YAML per model variant
tests/         unit + integration + golden
legacy/v1/     v1, preserved verbatim, excluded from every quality gate
docs/          audit + model cards -> mkdocs site
```

`core/labels.py` is the load-bearing module: the only place in the codebase
where an emotion label exists as a string. Everything else imports from it.

---

## License

MIT — see [LICENSE](LICENSE), which also lists the per-dataset terms. Several
corpora used here are research/non-commercial only, and the v1 speech model was
adapted from third-party work; both are noted there.

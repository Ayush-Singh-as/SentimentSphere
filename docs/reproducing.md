# Reproducing the numbers

Every figure this project publishes comes from a frozen split on a stamped run.
This page is how you regenerate them and how you check that what you got is
what was published.

## One command

```bash
make install
make reproduce-text
```

That freezes the split (or validates the committed one), fits the model on
train, fits the temperature on dev, scores the test split once, and writes
`reports/<run_id>/`.

Expected on the committed corpus and manifest:

| Metric | Value |
|---|---:|
| Test samples | 6,169 |
| Accuracy | 0.6387 |
| Macro F1 | 0.5986 |
| Weighted F1 | 0.6382 |
| ECE (calibrated) | 0.0200 |
| Temperature | 0.9871 |

The run id embeds a timestamp, so it differs every time; the metrics should
not. If they do, something in the chain below changed.

## What each run writes

```
reports/<run_id>/
  metrics.json      metrics, config, and the full provenance stamp
  predictions.json  per-sample test probabilities, for later fusion work
  confusion.svg     confusion matrix
  calibration.svg   per-class F1 and the reliability diagram
artifacts/
  models/<run_id>/  model.joblib + metadata.json (checksummed)
  registry.json     which bundle is active per modality
```

## The provenance stamp

`metrics.json` carries everything needed to invalidate the result:

| Field | Meaning |
|---|---|
| `git_sha` | The commit that produced it |
| `git_dirty` | Whether tracked *or untracked* files differed from HEAD |
| `config_hash` | Hash of the resolved run config, key-order independent |
| `extra.dataset` | Fingerprint over every sample id, label, group, and content hash |
| `extra.split_manifest` | Fingerprint of the frozen manifest |
| `extra.seed` | What was actually seeded, including whether `PYTHONHASHSEED` really took effect |
| `package_versions` | Versions of the numeric stack |

A dirty tree is recorded as dirty. Published numbers should come from a clean
one — the flag exists so that rule is checkable rather than trusted.

## Checking you got the published result

```bash
# same dataset?
uv run sphere data --dataset text_aggregate     # prints the fingerprint

# same split?
uv run python -c "
from sentimentsphere.data.splits import SplitManifest
from pathlib import Path
print(SplitManifest.load(Path('manifests/text_aggregate-v1.json')).fingerprint)"
```

Both fingerprints are pinned in `tests/golden/expected.json`, so
`uv run pytest tests/golden` fails if either drifted.

## Why the split cannot move under you

`sphere data` will not overwrite an existing manifest with a different one. Run
it again with a different seed and it exits nonzero rather than silently
re-splitting — otherwise every previously published metric would quietly stop
meaning anything. A corpus change gets a new versioned filename, never an edit
in place.

Training validates the manifest against the dataset fingerprint before it
starts, so a modified CSV fails loudly instead of training on a split that no
longer describes the data.

## Determinism, honestly

`seed_everything` seeds Python, NumPy, and (when present) Torch and CUDA, and
returns a report of what it actually reached. Two caveats are recorded rather
than papered over:

- `PYTHONHASHSEED` only takes effect before the interpreter starts. The seed
  report verifies the *real* hashing behaviour by comparing against a fresh
  subprocess, instead of trusting the environment variable.
- GPU determinism is best-effort. Torch's deterministic algorithms are
  requested with `warn_only=True`, so a nondeterministic kernel warns rather
  than aborting. Bitwise reproducibility on GPU is not claimed.

The text baseline is CPU-only and deterministic in practice: repeated runs on
the same commit give identical metrics.

## Measuring latency

```bash
make latency        # writes reports/latency.json
```

In-process against the FastAPI app, so it excludes network transit. Modalities
with no installed model are recorded as `unavailable` rather than estimated.
Measured for the text head: p50 5.0 ms, p95 8.6 ms.

## Scoring the v1 artifacts

```bash
make install-legacy
make eval-baseline
```

Requires the v1 weights in `artifacts/v1/` — they are not tracked in git. With
nothing to read, the command reports every artifact as unavailable and exits
nonzero rather than reporting a successful audit of nothing.

Note that this rescore is *not* an independent holdout: the v1 models' training
membership is unknown, so the numbers reproduce the historical comparison and
nothing stronger. The report says so in its `evaluation` field.

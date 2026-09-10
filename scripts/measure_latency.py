"""Measure per-modality request latency against the local app.

Writes reports/latency.json. Only modalities with an installed model are
measured; the rest are recorded as unavailable rather than estimated.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from sentimentsphere.core.config import Settings
from sentimentsphere.core.io import write_json
from sentimentsphere.core.provenance import collect
from sentimentsphere.serving.api import create_app

SAMPLES = [
    "I am absolutely thrilled with how this turned out",
    "that is completely revolting and i want nothing to do with it",
    "the meeting has been moved to three o'clock on tuesday",
    "i am terrified about what happens next",
    "i miss them more than i can say",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    # The benchmark is a local load generator, not a client to be throttled.
    settings = Settings(requests_per_minute=10 * (args.runs + args.warmup))
    report: dict[str, object] = {
        "provenance": collect({"runs": args.runs}).to_dict(),
        "note": "in-process TestClient; excludes network transit",
    }
    with TestClient(create_app(settings)) as client:
        available = client.get("/v1/models").json()["models"]
        for modality in ("text", "audio", "image", "fusion"):
            if not available.get(modality, {}).get("available"):
                report[modality] = {"status": "unavailable"}
                continue
            for i in range(args.warmup):
                client.post("/v1/predict/text", json={"text": SAMPLES[i % len(SAMPLES)]})
            timings = []
            for i in range(args.runs):
                started = time.perf_counter()
                response = client.post("/v1/predict/text", json={"text": SAMPLES[i % len(SAMPLES)]})
                timings.append((time.perf_counter() - started) * 1000)
                if response.status_code != 200:
                    raise SystemExit(f"unexpected {response.status_code}: {response.text}")
            ordered = sorted(timings)
            report[modality] = {
                "status": "measured",
                "runs": args.runs,
                "p50_ms": round(statistics.median(ordered), 3),
                "p95_ms": round(ordered[int(0.95 * len(ordered)) - 1], 3),
                "p99_ms": round(ordered[int(0.99 * len(ordered)) - 1], 3),
                "mean_ms": round(statistics.fmean(ordered), 3),
                "model_id": available[modality]["model_id"],
            }

    path = Path(settings.reports_dir) / "latency.json"
    write_json(path, report)
    print(f"wrote {path}")
    for modality in ("text", "audio", "image", "fusion"):
        print(f"  {modality}: {report[modality]}")


if __name__ == "__main__":
    main()

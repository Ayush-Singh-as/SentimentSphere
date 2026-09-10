"""Dataset dispatch and local preparation; source availability is explicit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentimentsphere.core.config import Settings
from sentimentsphere.core.io import write_json
from sentimentsphere.data.audio import combine_audio, load_audio
from sentimentsphere.data.records import Dataset
from sentimentsphere.data.splits import make_split
from sentimentsphere.data.text import load_text


def load_dataset(name: str, settings: Settings) -> Dataset:
    root = settings.data_dir / "raw"
    if name == "text_aggregate":
        return load_text(root / "text_emotion_aggregate.csv")
    if name in {"tess", "ravdess", "crema_d", "savee"}:
        return load_audio(root / name, name)
    if name == "speech_combined":
        return combine_audio(
            [
                load_audio(root / corpus, corpus)
                for corpus in ("tess", "ravdess", "crema_d", "savee")
            ]
        )
    raise ValueError(f"Dataset loader unavailable: {name}")


def prepare_dataset(name: str, settings: Settings, seed: int = 1337) -> dict[str, Any]:
    dataset = load_dataset(name, settings)
    paths: list[Path] = []
    if name == "tess":
        for actor in sorted({s.group for s in dataset.samples}):
            path = settings.manifests_dir / f"tess-loso-{actor.split(':')[-1].lower()}-v1.json"
            make_split(dataset, seed, held_out_actor=actor).save(path)
            paths.append(path)
    else:
        path = settings.manifests_dir / f"{name}-v1.json"
        make_split(dataset, seed).save(path)
        paths.append(path)
    summary = {
        "dataset": name,
        "fingerprint": dataset.fingerprint,
        "audit": dataset.audit,
        "manifests": [str(p) for p in paths],
    }
    write_json(settings.reports_dir / "data" / f"{name}.json", summary)
    return summary

"""Strict filename parsing for speech corpora; actor identity is part of every row."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from sentimentsphere.core.io import sha256_file, stable_digest
from sentimentsphere.core.labels import normalize, to_index
from sentimentsphere.data.records import Dataset, Sample


def parse_audio_name(path: Path, corpus: str) -> tuple[str, str]:
    """Return actor and raw emotion; malformed names never silently disappear."""
    parts = path.stem.split("_")
    if corpus == "tess":
        if len(parts) != 3:
            raise ValueError(f"Malformed TESS filename: {path.name}")
        actor = parts[0].upper()
        # Explicitly resolve the observed typo using its enclosing actor directory.
        if (
            actor == "OA"
            and path.name == "OA_bite_neutral.wav"
            and path.parent.name == "OAF_neutral"
        ):
            actor = "OAF"
        if actor not in {"OAF", "YAF"} or not path.parent.name.upper().startswith(actor + "_"):
            raise ValueError(f"Unverified TESS actor/directory: {path}")
        return actor, parts[-1]
    if corpus == "ravdess":
        fields = path.stem.split("-")
        if len(fields) != 7 or any(not f.isdigit() for f in fields):
            raise ValueError(f"Malformed RAVDESS filename: {path.name}")
        if fields[1] != "01":
            raise ValueError(f"Expected RAVDESS speech, not song: {path.name}")
        return fields[-1], fields[2]
    if corpus == "crema_d":
        if len(parts) != 4 or not parts[0].isdigit():
            raise ValueError(f"Malformed CREMA-D filename: {path.name}")
        return parts[0], parts[2]
    if corpus == "savee":
        if len(parts) == 2:
            actor, utterance = parts
        elif len(parts) == 1:
            actor, utterance = path.parent.name, parts[0]
        else:
            raise ValueError(f"Malformed SAVEE filename: {path.name}")
        if actor.upper() not in {"DC", "JE", "JK", "KL"}:
            raise ValueError(f"Unknown SAVEE actor: {actor}")
        emotion = utterance.rstrip("0123456789").lower()
        if emotion == utterance.lower():
            raise ValueError(f"Missing SAVEE utterance number: {path.name}")
        return actor.upper(), emotion
    raise ValueError(f"Unsupported speech corpus: {corpus}")


def load_audio(root: Path, corpus: str) -> Dataset:
    files = sorted(root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files in {root}; acquire {corpus} first")
    samples = []
    checksums = {}
    corrections = []
    for path in files:
        actor, raw = parse_audio_name(path, corpus)
        label = normalize(raw, corpus)
        if label is None:
            continue
        relative = path.relative_to(root).as_posix()
        checksum = sha256_file(path)
        checksums[relative] = checksum
        if path.name.startswith("OA_"):
            corrections.append(
                {"path": relative, "actor": actor, "reason": "verified enclosing OAF directory"}
            )
        samples.append(
            Sample(
                id=hashlib.sha256(f"{corpus}/{relative}".encode()).hexdigest()[:24],
                label=to_index(label),
                group=f"{corpus}:{actor}",
                content_hash=checksum,
                path=str(path),
                corpus=corpus,
            )
        )
    return Dataset(
        corpus,
        tuple(samples),
        stable_digest(checksums),
        {
            "files": len(samples),
            "actors": dict(Counter(s.group for s in samples)),
            "support": dict(Counter(str(s.label) for s in samples)),
            "corrections": corrections,
        },
    )


def combine_audio(datasets: list[Dataset]) -> Dataset:
    if not datasets:
        raise ValueError("At least one speech corpus is required")
    return Dataset(
        "speech_combined",
        tuple(s for d in datasets for s in d.samples),
        stable_digest({d.name: d.fingerprint for d in datasets}),
        {d.name: d.audit for d in datasets},
    )

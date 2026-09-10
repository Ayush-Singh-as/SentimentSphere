"""Shared fixtures: small synthetic corpora, so unit tests need no real datasets."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentimentsphere.core.config import Settings
from sentimentsphere.data.records import Dataset, Sample

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_CSV = REPO_ROOT / "data" / "raw" / "text_emotion_aggregate.csv"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted in tmp_path so no test writes into the working tree."""
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        manifests_dir=tmp_path / "manifests",
    )


def make_text_csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Emotion", "Text"])
        writer.writerows(rows)
    return path


@pytest.fixture
def tiny_text_csv(tmp_path: Path) -> Path:
    """Covers every canonical class plus a duplicate, a conflict, and an exclusion."""
    rows = [
        ("joy", "what a wonderful day"),
        ("joy", "What a Wonderful Day"),  # case/duplicate -> collapses
        ("anger", "this makes me furious"),
        ("disgust", "that is revolting"),
        ("fear", "i am terrified of this"),
        ("neutral", "the meeting is at three"),
        ("sadness", "i miss them so much"),
        ("surprise", "i did not expect that"),
        ("shame", "i feel ashamed"),  # excluded label
        ("joy", "ambiguous line"),
        ("anger", "ambiguous line"),  # conflicting group -> dropped
        ("neutral", "   "),  # text empties out after normalization
    ]
    return make_text_csv(tmp_path / "tiny.csv", rows)


def synthetic_dataset(name: str = "synthetic", per_class: int = 8, groups: int = 6) -> Dataset:
    """A grouped dataset large enough for stratified and grouped splitting."""
    samples = []
    for label in range(7):
        for i in range(per_class):
            index = label * per_class + i
            samples.append(
                Sample(
                    id=f"s{index:04d}",
                    label=label,
                    group=f"{name}:actor{index % groups:02d}",
                    content_hash=f"hash{index:04d}",
                    text=f"sample text number {index} for class {label}",
                    corpus=name,
                )
            )
    return Dataset(name, tuple(samples), source_hash=f"src-{name}")


@pytest.fixture
def make_dataset():
    return synthetic_dataset


@pytest.fixture
def grouped_dataset() -> Dataset:
    return synthetic_dataset()

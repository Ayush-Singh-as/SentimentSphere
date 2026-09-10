"""Frozen split manifests with explicit identity, coverage, and leakage checks."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from sentimentsphere.core.io import read_json, stable_digest, write_json
from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.data.records import Dataset, Sample

Partition = Literal["train", "dev", "test"]


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    dataset: str
    dataset_hash: str
    labels: list[str] = Field(default_factory=lambda: list(CANONICAL))
    seed: int
    policy: str
    assignments: dict[str, Partition]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_digest(self.model_dump(mode="json"))

    def save(self, path: Path) -> None:
        if path.exists():
            existing = self.load(path)
            if existing.fingerprint != self.fingerprint:
                raise FileExistsError(
                    f"Frozen manifest differs: {path}; use a new versioned filename"
                )
            return
        write_json(path, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> SplitManifest:
        return cls.model_validate(read_json(path))

    def validate_dataset(self, dataset: Dataset) -> None:
        if self.version != 1 or self.labels != list(CANONICAL):
            raise ValueError("Unsupported manifest version or label order")
        if self.dataset != dataset.name or self.dataset_hash != dataset.fingerprint:
            raise ValueError("Dataset content/labels changed since the split was frozen")
        ids = [s.id for s in dataset.samples]
        if len(ids) != len(set(ids)) or set(ids) != set(self.assignments):
            raise ValueError("Manifest must assign every sample exactly once")
        by_split = {split: self.samples(dataset, split) for split in ("train", "dev", "test")}
        if any(not samples for samples in by_split.values()):
            raise ValueError("Train, dev, and test must each contain samples")
        for field in ("group", "content_hash"):
            seen: dict[str, str] = {}
            for split, samples in by_split.items():
                for sample in samples:
                    key = str(getattr(sample, field))
                    # TESS has only two speakers: dev is a content-disjoint subset
                    # of the training actor. Only train/dev actor sharing is allowed.
                    partition = (
                        "fit"
                        if field == "group" and self.policy == "tess-loso-v1" and split != "test"
                        else split
                    )
                    if key in seen and seen[key] != partition:
                        raise ValueError(f"{field} leakage between partitions: {key}")
                    seen[key] = partition

    def samples(self, dataset: Dataset, partition: Partition) -> list[Sample]:
        return [s for s in dataset.samples if self.assignments.get(s.id) == partition]


def make_split(
    dataset: Dataset, seed: int = 1337, *, held_out_actor: str | None = None
) -> SplitManifest:
    """Stratify text; group audio actors; reserve test actor for TESS LOSO."""
    rows = sorted(dataset.samples, key=lambda s: s.id)
    ids = np.arange(len(rows))
    targets = np.array([s.label for s in rows])
    groups = np.array([s.group for s in rows])
    policy = "stratified-text-v1"
    if held_out_actor is not None:
        policy = "tess-loso-v1"
        if dataset.name != "tess" or len(set(groups)) != 2 or held_out_actor not in groups:
            raise ValueError("TESS LOSO requires two actors and a valid held-out actor")
        test = ids[groups == held_out_actor]
        fit = ids[groups != held_out_actor]
        train, dev = train_test_split(fit, test_size=0.2, random_state=seed, stratify=targets[fit])
    elif dataset.name == "text_aggregate":
        fit, test = train_test_split(ids, test_size=0.2, random_state=seed, stratify=targets)
        train, dev = train_test_split(fit, test_size=0.2, random_state=seed, stratify=targets[fit])
    else:
        policy = "actor-grouped-v1"
        if len(set(groups)) < 5:
            raise ValueError("Three-way actor split needs at least five actors; use TESS LOSO")
        fit, test = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                ids, targets, groups
            )
        )
        train_rel, dev_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
                fit, targets[fit], groups[fit]
            )
        )
        train, dev = fit[train_rel], fit[dev_rel]
    assignments: dict[str, Partition] = {}
    for split, indices in (("train", train), ("dev", dev), ("test", test)):
        for i in indices:
            assignments[rows[i].id] = split  # type: ignore[assignment]
    manifest = SplitManifest(
        dataset=dataset.name,
        dataset_hash=dataset.fingerprint,
        seed=seed,
        policy=policy,
        assignments=assignments,
        metadata={
            "audit": dataset.audit,
            "support": {
                p: dict(Counter(str(s.label) for s in rows if assignments[s.id] == p))
                for p in ("train", "dev", "test")
            },
            "actors": {
                p: sorted({s.group for s in rows if assignments[s.id] == p})
                if dataset.name != "text_aggregate"
                else []
                for p in ("train", "dev", "test")
            },
        },
    )
    manifest.validate_dataset(dataset)
    return manifest

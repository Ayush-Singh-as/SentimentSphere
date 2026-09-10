"""Portable sample identities shared by loaders and frozen split manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sentimentsphere.core.io import stable_digest


@dataclass(frozen=True)
class Sample:
    id: str
    label: int
    group: str
    content_hash: str
    text: str = ""
    path: str = ""
    corpus: str = ""


@dataclass(frozen=True)
class Dataset:
    name: str
    samples: tuple[Sample, ...]
    source_hash: str
    audit: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_digest(
            {
                "name": self.name,
                "source_hash": self.source_hash,
                "samples": [
                    [s.id, s.label, s.group, s.content_hash]
                    for s in sorted(self.samples, key=lambda row: row.id)
                ],
            }
        )

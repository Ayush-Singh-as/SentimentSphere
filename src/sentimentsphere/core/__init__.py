"""Cross-cutting primitives: label space, determinism, provenance, config."""

from __future__ import annotations

from sentimentsphere.core.labels import (
    CANONICAL,
    INDEX,
    NUM_CLASSES,
    Emotion,
    UnknownLabelError,
    from_index,
    normalize,
    to_index,
)
from sentimentsphere.core.provenance import Provenance, collect, hash_config, hash_file
from sentimentsphere.core.seeding import DEFAULT_SEED, seed_everything

__all__ = [
    "CANONICAL",
    "DEFAULT_SEED",
    "INDEX",
    "NUM_CLASSES",
    "Emotion",
    "Provenance",
    "UnknownLabelError",
    "collect",
    "from_index",
    "hash_config",
    "hash_file",
    "normalize",
    "seed_everything",
    "to_index",
]

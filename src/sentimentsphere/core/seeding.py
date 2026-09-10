"""Determinism controls.

Every training and evaluation entry point calls :func:`seed_everything` before
touching data. v1 had no seeding at all, which is part of why its three visual
training runs produced three different numbers that could not be reconciled.

Full bit-for-bit reproducibility on GPU is not free — cuDNN's autotuner and some
kernels are nondeterministic by default. :func:`seed_everything` therefore takes
a ``deterministic`` flag: on for anything whose number gets published, off for
exploratory sweeps where the throughput matters more.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DEFAULT_SEED = 1337


@lru_cache(maxsize=16)
def _hash_seed_matches(seed: int) -> bool:
    """Compare actual hashing with a fresh interpreter, not a mutable env claim."""
    probe = "SentimentSphere hash-seed verification"
    try:
        child = subprocess.run(
            [sys.executable, "-c", f"print(hash({probe!r}))"],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return int(child.stdout.strip()) == hash(probe)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


@dataclass(frozen=True, slots=True)
class SeedReport:
    """What was actually seeded, for inclusion in a run's provenance stamp."""

    seed: int
    deterministic: bool
    torch_seeded: bool
    cuda_seeded: bool
    pythonhashseed_effective: bool


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> SeedReport:
    """Seed every RNG the project can reach.

    Args:
        seed: The seed to apply to ``random``, ``numpy``, and ``torch``.
        deterministic: If True, also force deterministic cuDNN/cuBLAS kernels
            and disable the autotuner. Costs throughput; required for any run
            whose metrics are published.

    Returns:
        A :class:`SeedReport` describing what was reachable and seeded.

    Note:
        ``PYTHONHASHSEED`` only affects string hashing if it is set *before* the
        interpreter starts. Setting it here is a best-effort no-op for the
        current process; the ``Makefile`` exports it so that subprocess-launched
        runs get it for real. The returned report says which case applied.
    """
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    hashseed_already_set = _hash_seed_matches(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global seed needed for third-party libs

    torch_seeded = False
    cuda_seeded = False
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            cuda_seeded = True
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # Published deterministic runs fail if an operation cannot meet the contract.
        torch.use_deterministic_algorithms(deterministic)

    return SeedReport(
        seed=seed,
        deterministic=deterministic,
        torch_seeded=torch_seeded,
        cuda_seeded=cuda_seeded,
        pythonhashseed_effective=hashseed_already_set,
    )


def worker_init_fn(worker_id: int) -> None:
    """Per-worker seeding for ``torch.utils.data.DataLoader``.

    Without this, every dataloader worker inherits the same numpy RNG state and
    applies *identical* random augmentation to different batches.
    """
    import torch

    # Torch already includes worker_id in initial_seed; adding it twice also risks overflow.
    base = torch.initial_seed() % 2**32
    np.random.seed(base)  # noqa: NPY002
    random.seed(base)

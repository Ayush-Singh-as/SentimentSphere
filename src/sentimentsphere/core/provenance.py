"""Run provenance stamping.

Every artifact this project writes — ``metrics.json``, a split manifest, an
exported ONNX model — carries a provenance block saying exactly which code,
config, and data produced it.

This is the direct fix for the central v1 failure: three visual training runs
produced three different accuracies (0.573 notebook / 0.672 report / unknown for
the shipped weights) and there was no way to tell which code produced which
number, or whether the shipped ``model.h5`` corresponded to any of them.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _git(*args: str) -> str | None:
    """Run a git command, returning None if git or the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_sha(short: bool = False) -> str | None:
    """Current commit SHA, or None outside a git checkout."""
    return _git("rev-parse", "--short", "HEAD") if short else _git("rev-parse", "HEAD")


def git_is_dirty() -> bool | None:
    """True if tracked or untracked (non-ignored) files differ from HEAD.

    A dirty tree means the recorded SHA does not fully describe the code that
    ran, so published numbers should come from a clean tree.
    """
    status = _git("status", "--porcelain", "--untracked-files=normal")
    return None if status is None else bool(status)


def hash_config(config: Any) -> str:
    """Stable SHA-256 (first 16 hex chars) of a config object.

    Key order is normalised so that a reordered YAML file does not read as a
    different experiment.
    """
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def hash_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 (first 16 hex chars) of a file's bytes.

    Used to pin dataset and weight files, so a silently-swapped artifact is
    detectable rather than mysterious.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to reproduce or invalidate a result."""

    git_sha: str | None
    git_dirty: bool | None
    config_hash: str | None
    python_version: str
    platform: str
    package_versions: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "torch",
    "transformers",
    "librosa",
    "onnxruntime",
    "tensorflow",
    "tensorflow-cpu",
    "tf-keras",
    "scipy",
    "soundfile",
    "neattext",
    "pillow",
    "fastapi",
    "gradio",
)


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            continue
    return found


def collect(config: Any | None = None, **extra: Any) -> Provenance:
    """Snapshot the current environment.

    Args:
        config: The run's resolved config; hashed into ``config_hash``.
        **extra: Anything else worth pinning — split manifest hash, dataset
            checksums, seed report.
    """
    return Provenance(
        git_sha=git_sha(),
        git_dirty=git_is_dirty(),
        config_hash=hash_config(config) if config is not None else None,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_versions=_package_versions(),
        extra=extra,
    )

"""Versioned local artifact bundles and registry; no missing-weight fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sentimentsphere.core.io import read_json, sha256_file, write_json
from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES
from sentimentsphere.core.types import FloatArray, probabilities


class ModelUnavailableError(RuntimeError):
    """A requested trained model has not been installed or failed integrity checks."""


def align_probabilities(scores: Any, classes: Any) -> FloatArray:
    values = np.asarray(scores, dtype=np.float64)
    source = np.asarray(classes)
    if (
        values.ndim != 2
        or len(source) != values.shape[1]
        or len(set(source.tolist())) != len(source)
    ):
        raise ValueError("Invalid estimator class/probability correspondence")
    if (
        not np.issubdtype(source.dtype, np.integer)
        or (source < 0).any()
        or (source >= NUM_CLASSES).any()
    ):
        raise ValueError("Estimator classes must be canonical integer indices")
    output = np.zeros((len(values), NUM_CLASSES), dtype=np.float64)
    output[:, source] = values
    return probabilities(output)


def save_sklearn_bundle(directory: Path, model: Any, metadata: dict[str, Any]) -> None:
    """Save a trusted locally trained estimator; only fixed model bundles are loaded."""
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "model.joblib"
    joblib.dump(model, target)
    write_json(
        directory / "metadata.json",
        {
            **metadata,
            "schema_version": 1,
            "labels": list(CANONICAL),
            "files": {"model.joblib": sha256_file(target)},
        },
    )


def verify_bundle(directory: Path) -> dict[str, Any]:
    try:
        metadata = read_json(directory / "metadata.json")
        if metadata.get("schema_version") != 1 or metadata.get("labels") != list(CANONICAL):
            raise ValueError("Unsupported model schema or incompatible label order")
        if not metadata.get("files"):
            raise ValueError("Missing artifact checksums")
        for relative, expected in metadata["files"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
                raise ValueError(f"Artifact checksum/path mismatch: {relative}")
        return metadata
    except (OSError, ValueError, KeyError) as error:
        raise ModelUnavailableError(f"Model bundle unavailable at {directory}: {error}") from error


def register_model(root: Path, modality: str, directory: Path) -> None:
    """Atomically set the active bundle after its integrity has been verified."""
    verify_bundle(directory)
    relative = directory.resolve().relative_to(root.resolve()).as_posix()
    registry_path = root / "registry.json"
    registry = read_json(registry_path) if registry_path.exists() else {}
    registry[modality] = relative
    write_json(registry_path, registry)


def registered_path(root: Path, modality: str) -> Path:
    try:
        registry = read_json(root / "registry.json")
        relative = registry[modality]
        if not isinstance(relative, str):
            raise ValueError("Model registry entry must be a relative path string")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Model registry path escapes artifact directory")
        return path
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ModelUnavailableError(
            f"No installed {modality} model; train or install a verified bundle"
        ) from error

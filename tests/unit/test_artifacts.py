"""Artifact integrity: a missing or tampered model must refuse, never fail open.

v1's speech app printed an error for missing weights and then predicted with
randomly initialised ones. Every assertion here exists to keep that impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES
from sentimentsphere.inference.artifacts import (
    ModelUnavailableError,
    align_probabilities,
    register_model,
    registered_path,
    save_sklearn_bundle,
    verify_bundle,
)


class DummyModel:
    """Stands in for a fitted estimator; joblib round-trips it fine."""

    classes_ = np.array([0, 3])

    def predict_proba(self, rows):
        return np.tile([0.25, 0.75], (len(rows), 1))


def make_bundle(root: Path, name: str = "run-1") -> Path:
    directory = root / "models" / name
    save_sklearn_bundle(
        directory,
        DummyModel(),
        {"model_id": name, "kind": "sklearn_text", "modality": "text", "temperature": 1.0},
    )
    return directory


# --------------------------------------------------------------------------- #
# probability alignment
# --------------------------------------------------------------------------- #


def test_align_scatters_into_canonical_columns():
    out = align_probabilities([[0.25, 0.75]], np.array([0, 3]))
    assert out.shape == (1, NUM_CLASSES)
    assert out[0, 0] == 0.25
    assert out[0, 3] == 0.75
    assert out[0, [1, 2, 4, 5, 6]].sum() == 0.0


def test_align_rejects_noncanonical_or_duplicate_classes():
    with pytest.raises(ValueError):
        align_probabilities([[0.5, 0.5]], np.array([0, NUM_CLASSES]))  # out of range
    with pytest.raises(ValueError):
        align_probabilities([[0.5, 0.5]], np.array([1, 1]))  # duplicated
    with pytest.raises(ValueError):
        align_probabilities([[0.5, 0.5]], np.array(["joy", "anger"]))  # not indices
    with pytest.raises(ValueError):
        align_probabilities([[0.5, 0.5]], np.array([0]))  # width mismatch


def test_align_still_enforces_the_probability_contract():
    with pytest.raises(ValueError):
        align_probabilities([[0.3, 0.3]], np.array([0, 1]))  # does not sum to one


# --------------------------------------------------------------------------- #
# bundles
# --------------------------------------------------------------------------- #


def test_saved_bundle_verifies(tmp_path: Path):
    metadata = verify_bundle(make_bundle(tmp_path))
    assert metadata["labels"] == list(CANONICAL)
    assert metadata["schema_version"] == 1


def test_save_refuses_to_overwrite(tmp_path: Path):
    make_bundle(tmp_path)
    with pytest.raises(FileExistsError):
        make_bundle(tmp_path)


def test_verify_detects_tampered_weights(tmp_path: Path):
    directory = make_bundle(tmp_path)
    (directory / "model.joblib").write_bytes(b"swapped")
    with pytest.raises(ModelUnavailableError, match="checksum"):
        verify_bundle(directory)


def test_verify_rejects_incompatible_label_order(tmp_path: Path):
    directory = make_bundle(tmp_path)
    path = directory / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["labels"] = list(reversed(CANONICAL))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelUnavailableError, match="label order"):
        verify_bundle(directory)


def test_verify_rejects_missing_bundle(tmp_path: Path):
    with pytest.raises(ModelUnavailableError):
        verify_bundle(tmp_path / "nope")


def test_verify_rejects_checksum_path_escape(tmp_path: Path):
    directory = make_bundle(tmp_path)
    path = directory / "metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"] = {"../../escape.bin": "0" * 64}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        verify_bundle(directory)


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_register_and_resolve(tmp_path: Path):
    directory = make_bundle(tmp_path)
    register_model(tmp_path, "text", directory)
    assert registered_path(tmp_path, "text") == directory.resolve()


def test_registry_survives_a_moved_root(tmp_path: Path):
    """Entries are relative, so the artifacts dir can be mounted anywhere."""
    directory = make_bundle(tmp_path)
    register_model(tmp_path, "text", directory)
    entry = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))["text"]
    assert not Path(entry).is_absolute()


def test_unregistered_modality_raises(tmp_path: Path):
    with pytest.raises(ModelUnavailableError, match="No installed audio model"):
        registered_path(tmp_path, "audio")


def test_registry_rejects_escaping_entry(tmp_path: Path):
    (tmp_path / "registry.json").write_text(json.dumps({"text": "../../etc"}), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        registered_path(tmp_path, "text")


def test_register_refuses_an_unverifiable_bundle(tmp_path: Path):
    broken = tmp_path / "models" / "broken"
    broken.mkdir(parents=True)
    with pytest.raises(ModelUnavailableError):
        register_model(tmp_path, "text", broken)

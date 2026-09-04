"""Tests for determinism and provenance stamping."""

from __future__ import annotations

import random

import numpy as np

from sentimentsphere.core import provenance
from sentimentsphere.core.seeding import DEFAULT_SEED, seed_everything


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())  # noqa: NPY002 - exercising the legacy global RNG on purpose


def test_seeding_is_reproducible():
    seed_everything(DEFAULT_SEED)
    first = _draw()
    seed_everything(DEFAULT_SEED)
    assert _draw() == first


def test_different_seeds_diverge():
    seed_everything(1)
    a = _draw()
    seed_everything(2)
    assert _draw() != a


def test_seed_report_describes_what_was_seeded():
    report = seed_everything(DEFAULT_SEED, deterministic=False)
    assert report.seed == DEFAULT_SEED
    assert report.deterministic is False
    # torch is an optional extra; the report must be honest either way
    assert isinstance(report.torch_seeded, bool)
    assert report.cuda_seeded is False or report.torch_seeded


def test_config_hash_ignores_key_order():
    """A reordered YAML file is the same experiment and must hash the same."""
    a = {"lr": 3e-5, "model": "deberta-v3-base", "epochs": 5}
    b = {"epochs": 5, "model": "deberta-v3-base", "lr": 3e-5}
    assert provenance.hash_config(a) == provenance.hash_config(b)


def test_config_hash_detects_real_change():
    a = {"lr": 3e-5, "epochs": 5}
    b = {"lr": 5e-5, "epochs": 5}
    assert provenance.hash_config(a) != provenance.hash_config(b)


def test_hash_file(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"abc" * 1000)
    q = tmp_path / "same.bin"
    q.write_bytes(b"abc" * 1000)
    r = tmp_path / "different.bin"
    r.write_bytes(b"abd" * 1000)
    assert provenance.hash_file(p) == provenance.hash_file(q)
    assert provenance.hash_file(p) != provenance.hash_file(r)
    assert len(provenance.hash_file(p)) == 16


def test_collect_snapshots_environment():
    prov = provenance.collect({"model": "x"}, split_manifest="deadbeef")
    assert prov.python_version.startswith("3.")
    assert prov.config_hash is not None
    assert prov.extra["split_manifest"] == "deadbeef"
    assert "numpy" in prov.package_versions
    assert prov.to_dict()["config_hash"] == prov.config_hash


def test_collect_without_config():
    assert provenance.collect().config_hash is None

"""Split determinism and no-leakage — the Phase 1 exit criterion.

v1's headline 100% was a two-speaker corpus split at random. These assertions
are what makes that class of failure a red test rather than a press release.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sentimentsphere.data.records import Dataset, Sample
from sentimentsphere.data.splits import SplitManifest, make_split


def tess_like() -> Dataset:
    """Two actors reciting the same words, i.e. exactly the v1 leakage trap."""
    samples = []
    for a, actor in enumerate(("OAF", "YAF")):
        for label in range(7):
            for i in range(10):
                index = a * 70 + label * 10 + i
                samples.append(
                    Sample(
                        id=f"t{index:04d}",
                        label=label,
                        group=f"tess:{actor}",
                        content_hash=f"c{index:04d}",
                        path=f"{actor}_{i}_{label}.wav",
                        corpus="tess",
                    )
                )
    return Dataset("tess", tuple(samples), source_hash="tess-src")


def test_split_is_deterministic(grouped_dataset):
    a = make_split(grouped_dataset, seed=7)
    b = make_split(grouped_dataset, seed=7)
    assert a.fingerprint == b.fingerprint
    assert make_split(grouped_dataset, seed=8).assignments != a.assignments


def test_every_sample_assigned_exactly_once(grouped_dataset):
    manifest = make_split(grouped_dataset, seed=7)
    assert set(manifest.assignments) == {s.id for s in grouped_dataset.samples}
    total = sum(len(manifest.samples(grouped_dataset, p)) for p in ("train", "dev", "test"))
    assert total == len(grouped_dataset.samples)


def test_no_actor_overlap_across_partitions(grouped_dataset):
    manifest = make_split(grouped_dataset, seed=7)
    assert manifest.policy == "actor-grouped-v1"
    seen = {
        p: {s.group for s in manifest.samples(grouped_dataset, p)} for p in ("train", "dev", "test")
    }
    assert not seen["train"] & seen["test"]
    assert not seen["dev"] & seen["test"]
    assert not seen["train"] & seen["dev"]


def test_grouped_split_refuses_too_few_actors(make_dataset):
    with pytest.raises(ValueError, match="at least five actors"):
        make_split(make_dataset("small", per_class=4, groups=3), seed=1)


def test_tess_loso_holds_out_the_test_speaker_entirely():
    dataset = tess_like()
    manifest = make_split(dataset, seed=1337, held_out_actor="tess:OAF")
    assert manifest.policy == "tess-loso-v1"
    test_actors = {s.group for s in manifest.samples(dataset, "test")}
    fit_actors = {s.group for p in ("train", "dev") for s in manifest.samples(dataset, p)}
    assert test_actors == {"tess:OAF"}
    assert fit_actors == {"tess:YAF"}, "calibration must never see the held-out speaker"


def test_tess_loso_rejects_unknown_actor():
    with pytest.raises(ValueError):
        make_split(tess_like(), held_out_actor="tess:ZZZ")


def test_content_hash_never_crosses_partitions():
    """Identical audio in train and test is leakage even with distinct ids."""
    dataset = tess_like()
    manifest = make_split(dataset, held_out_actor="tess:OAF")
    by_split = {
        p: [s.content_hash for s in manifest.samples(dataset, p)] for p in ("train", "dev", "test")
    }
    assert not set(by_split["train"]) & set(by_split["test"])
    assert not set(by_split["dev"]) & set(by_split["test"])


def test_validate_dataset_detects_content_drift(grouped_dataset):
    manifest = make_split(grouped_dataset, seed=7)
    mutated = Dataset(grouped_dataset.name, grouped_dataset.samples, source_hash="different-source")
    with pytest.raises(ValueError, match="changed since the split was frozen"):
        manifest.validate_dataset(mutated)


def test_validate_dataset_detects_missing_sample(grouped_dataset):
    manifest = make_split(grouped_dataset, seed=7)
    truncated = Dataset(
        grouped_dataset.name, grouped_dataset.samples[:-1], grouped_dataset.source_hash
    )
    with pytest.raises(ValueError):
        manifest.validate_dataset(truncated)


def test_manifest_roundtrip_and_freeze(tmp_path: Path, grouped_dataset):
    manifest = make_split(grouped_dataset, seed=7)
    path = tmp_path / "m.json"
    manifest.save(path)
    assert SplitManifest.load(path).fingerprint == manifest.fingerprint
    manifest.save(path)  # identical rewrite is a no-op
    with pytest.raises(FileExistsError, match="Frozen manifest differs"):
        make_split(grouped_dataset, seed=9).save(path)


@pytest.mark.parametrize("name", ["text_aggregate-v1", "tess-loso-oaf-v1", "tess-loso-yaf-v1"])
def test_committed_manifests_are_wellformed(name):
    manifest = SplitManifest.load(Path("manifests") / f"{name}.json")
    assert manifest.version == 1
    assert set(manifest.assignments.values()) == {"train", "dev", "test"}

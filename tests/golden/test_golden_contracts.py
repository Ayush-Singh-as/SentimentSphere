"""Golden contracts: values that must not drift without a deliberate decision.

These are cheap and slightly boring on purpose. Each one pins something that a
downstream artifact depends on, so an accidental change fails here rather than
silently invalidating every published number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentimentsphere.core.io import stable_digest
from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES, SOURCE_MAPS, normalize, to_index
from sentimentsphere.data.splits import SplitManifest
from sentimentsphere.data.text import clean_text, text_identity

GOLDEN = Path(__file__).parent / "expected.json"


def load_golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_canonical_order_is_pinned():
    """Index meaning is baked into every trained artifact and manifest."""
    assert list(CANONICAL) == load_golden()["canonical"]
    assert NUM_CLASSES == 7


def test_source_map_digest_is_pinned():
    """Any change to a dataset vocabulary must be an explicit golden update."""
    digest = stable_digest(
        {
            source: {k: (v.value if v else None) for k, v in mapping.items()}
            for source, mapping in SOURCE_MAPS.items()
        }
    )
    assert digest == load_golden()["source_maps_digest"], (
        "a source vocabulary changed; update tests/golden/expected.json only if intended"
    )


@pytest.mark.parametrize(
    ("raw", "source", "expected"),
    [
        ("ps", "tess", "surprise"),
        ("calm", "ravdess", "neutral"),
        ("05", "ravdess", "anger"),
        ("ANG", "crema_d", "anger"),
        ("sa", "savee", "sadness"),
        ("happy", "fer2013", "joy"),
        ("female_angry", "v1_speech_conv1d", "anger"),
    ],
)
def test_pinned_label_normalizations(raw, source, expected):
    label = normalize(raw, source)
    assert label is not None and label.value == expected


def test_text_identity_is_stable_across_releases():
    """The split manifest keys are these hashes; changing them re-splits everything."""
    for text, expected in load_golden()["text_identities"].items():
        assert text_identity(text) == expected


def test_clean_text_is_pinned():
    for raw, expected in load_golden()["clean_text"].items():
        assert clean_text(raw) == expected


def test_committed_manifest_fingerprints_are_pinned():
    """A manifest edit invalidates every metric computed against it."""
    for name, expected in load_golden()["manifest_fingerprints"].items():
        manifest = SplitManifest.load(Path("manifests") / f"{name}.json")
        assert manifest.fingerprint == expected, f"{name} changed; metrics no longer comparable"


def test_index_roundtrip():
    for i, label in enumerate(CANONICAL):
        assert to_index(label) == i

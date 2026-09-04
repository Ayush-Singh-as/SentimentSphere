"""Tests for the canonical label space.

Several of these lock in specific v1 defects so they cannot reappear. Where that
is the case the test name and docstring say which defect.
"""

from __future__ import annotations

import pytest

from sentimentsphere.core import labels


# --------------------------------------------------------------------------- #
# The ordering contract
# --------------------------------------------------------------------------- #
def test_canonical_is_alphabetical():
    """CANONICAL must stay sorted.

    LabelEncoder, flow_from_directory, and sorted(listdir()) all emit classes in
    alphabetical order. Matching them is what lets an index from any of those
    line up with ours without a translation table.
    """
    assert list(labels.CANONICAL) == sorted(labels.CANONICAL)


def test_canonical_has_seven_classes():
    assert labels.NUM_CLASSES == 7
    assert len(labels.CANONICAL) == 7
    assert len(set(labels.CANONICAL)) == 7, "duplicate label in CANONICAL"


def test_index_round_trips():
    for i, label in enumerate(labels.CANONICAL):
        assert labels.to_index(label) == i
        assert labels.from_index(i) == label


def test_enum_and_tuple_agree():
    assert tuple(e.value for e in labels.Emotion) == labels.CANONICAL


def test_from_index_rejects_out_of_range():
    with pytest.raises(IndexError):
        labels.from_index(labels.NUM_CLASSES)


def test_to_index_rejects_non_canonical():
    """'happy' is a raw label, not a canonical one; it must not silently resolve."""
    with pytest.raises(labels.UnknownLabelError):
        labels.to_index("happy")


# --------------------------------------------------------------------------- #
# Totality: every source vocabulary resolves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", sorted(labels.SOURCE_MAPS))
def test_every_source_maps_into_canonical_or_none(source: str):
    """No source may map to a label outside the canonical space."""
    for raw, canon in labels.SOURCE_MAPS[source].items():
        if canon is None:
            assert raw in labels.EXCLUSION_REASONS, (
                f"{source}:{raw} is excluded but has no documented reason; "
                f"add one to EXCLUSION_REASONS"
            )
        else:
            assert canon in labels.CANONICAL, f"{source}:{raw} -> {canon!r} is not canonical"


@pytest.mark.parametrize("source", sorted(labels.SOURCE_MAPS))
def test_normalize_accepts_entire_vocabulary(source: str):
    for raw in labels.SOURCE_MAPS[source]:
        result = labels.normalize(raw, source)
        assert result is None or result in labels.CANONICAL


@pytest.mark.parametrize("source", sorted(labels.SOURCE_MAPS))
def test_source_keys_are_lowercase(source: str):
    """normalize() lowercases its input, so uppercase keys would be unreachable."""
    for raw in labels.SOURCE_MAPS[source]:
        assert raw == raw.lower(), f"{source}:{raw!r} can never be matched"


def test_normalize_is_case_and_whitespace_insensitive():
    assert labels.normalize("  ANGRY ", "tess") is labels.Emotion.ANGER


def test_normalize_rejects_unknown_label():
    with pytest.raises(labels.UnknownLabelError, match="not in the 'tess' vocabulary"):
        labels.normalize("ennui", "tess")


def test_normalize_rejects_unknown_source():
    with pytest.raises(labels.UnknownLabelError, match="unknown source"):
        labels.normalize("anger", "not_a_dataset")


# --------------------------------------------------------------------------- #
# Regression: v1 defects
# --------------------------------------------------------------------------- #
def test_fer2013_directory_order_is_canonical():
    """REGRESSION (v1 visual app): label order must come from sorted dir names.

    FER-2013 unpacks to directories that sort as
        angry, disgust, fear, happy, neutral, sad, surprise
    which maps 1:1 onto CANONICAL. The v1 app hardcoded
        ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    scrambling indices 4, 5, and 6 — so every neutral/sad/surprise prediction
    was reported under the wrong name.
    """
    fer_dirs_sorted = sorted(labels.SOURCE_MAPS["fer2013"])
    mapped = [labels.normalize(d, "fer2013") for d in fer_dirs_sorted]
    assert mapped == list(labels.CANONICAL)


def test_v1_visual_hardcoded_order_was_wrong():
    """Documents the v1 bug so the contrast is executable, not just prose."""
    v1_hardcoded = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    v1_normalized = [labels.normalize(name.lower(), "fer2013") for name in v1_hardcoded]
    assert v1_normalized != list(labels.CANONICAL)
    # Precisely: the first four were right, the last three were rotated.
    assert v1_normalized[:4] == list(labels.CANONICAL[:4])
    assert v1_normalized[4:] == [
        labels.Emotion.SADNESS,
        labels.Emotion.SURPRISE,
        labels.Emotion.NEUTRAL,
    ]
    assert list(labels.CANONICAL[4:]) == [
        labels.Emotion.NEUTRAL,
        labels.Emotion.SADNESS,
        labels.Emotion.SURPRISE,
    ]


def test_tess_ps_means_pleasant_surprise():
    """REGRESSION: v1 left TESS's 'ps' token unmapped, so it was its own class."""
    assert labels.normalize("ps", "tess") is labels.Emotion.SURPRISE


def test_shame_is_excluded_with_a_reason():
    assert labels.normalize("shame", "text_aggregate") is None
    assert labels.excluded_labels("text_aggregate") == ("shame",)
    assert "0.42%" in labels.EXCLUSION_REASONS["shame"]


def test_ravdess_calm_folds_into_neutral():
    assert labels.normalize("02", "ravdess") is labels.Emotion.NEUTRAL
    assert labels.normalize("calm", "ravdess") is labels.Emotion.NEUTRAL


def test_v1_speech_gender_is_stripped():
    """v2 drops the gender component of the inherited 10-class label space."""
    assert labels.normalize("female_angry", "v1_speech_conv1d") is labels.Emotion.ANGER
    assert labels.normalize("male_angry", "v1_speech_conv1d") is labels.Emotion.ANGER
    canonical_values = {
        labels.normalize(k, "v1_speech_conv1d") for k in labels.SOURCE_MAPS["v1_speech_conv1d"]
    }
    assert canonical_values == {
        labels.Emotion.ANGER,
        labels.Emotion.FEAR,
        labels.Emotion.JOY,
        labels.Emotion.NEUTRAL,
        labels.Emotion.SADNESS,
    }
    assert not any("male" in str(v) for v in canonical_values)


def test_crema_d_has_no_surprise():
    """Documents a real coverage gap rather than hiding it.

    CREMA-D provides 6 of the 7 canonical classes. Per-corpus support must be
    reported so a macro-F1 over the union is not read as if every corpus
    contributed every class.
    """
    covered = {labels.normalize(k, "crema_d") for k in labels.SOURCE_MAPS["crema_d"]}
    assert labels.Emotion.SURPRISE not in covered
    assert len(covered) == 6


def test_meld_needs_no_remapping():
    """MELD was chosen as the fusion corpus because its labels are already canonical."""
    assert sorted(labels.SOURCE_MAPS["meld"]) == sorted(labels.CANONICAL)
    for raw in labels.SOURCE_MAPS["meld"]:
        assert labels.normalize(raw, "meld") == raw

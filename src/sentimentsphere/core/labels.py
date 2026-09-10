"""Canonical emotion label space and per-dataset mappings.

This module is the **only** place in the codebase where emotion label strings are
allowed to be defined. Every dataset loader, model head, metric, and serving
response resolves its labels through here.

Why this file exists
--------------------
The v1 project had four incompatible label spaces at once — text had 8 classes,
TESS had 7 (spelled differently), the speech app had 10 (gender x emotion), and
FER-2013 had 7 in directory order. Nothing could be fused, and the visual app
shipped a **hardcoded list in the wrong order**, silently mislabelling three of
its seven classes. See ``docs/audit.md``.

Ordering contract
-----------------
``CANONICAL`` is **alphabetical**, and that is load-bearing rather than
cosmetic: ``sklearn.preprocessing.LabelEncoder``,
``keras.preprocessing.image.ImageDataGenerator.flow_from_directory``, and
``sorted(os.listdir(...))`` all order classes alphabetically. Matching that
convention means an index produced by any of them lines up with ours.

Never write a literal list of label names next to a model's output vector.
Always index through :data:`CANONICAL` or :func:`from_index`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Emotion(StrEnum):
    """The seven canonical emotion classes.

    Members are alphabetical; see the module docstring for why that matters.
    """

    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    NEUTRAL = "neutral"
    SADNESS = "sadness"
    SURPRISE = "surprise"


CANONICAL: Final[tuple[str, ...]] = tuple(e.value for e in Emotion)
"""Canonical label order. Index i of any model's probability vector is CANONICAL[i]."""

NUM_CLASSES: Final[int] = len(CANONICAL)

INDEX: Final[dict[str, int]] = {label: i for i, label in enumerate(CANONICAL)}
"""label -> index. Inverse of :data:`CANONICAL`."""


# --------------------------------------------------------------------------- #
# Deliberate exclusions
# --------------------------------------------------------------------------- #
# Mapping a raw label to None means "recognised, and deliberately dropped".
# That is different from an unknown label, which is an error. Keeping the two
# distinct is what stops a typo in a dataset from being silently discarded.
EXCLUSION_REASONS: Final[dict[str, str]] = {
    "shame": (
        "Only present in the text aggregate, where it is 146 of 34,792 rows (0.42%). "
        "No speech or vision corpus provides it, so it can never participate in fusion, "
        "and a test-set F1 computed over ~29 examples is noise, not a measurement."
    ),
}


# --------------------------------------------------------------------------- #
# Per-source raw -> canonical maps
# --------------------------------------------------------------------------- #
# Keys are always lowercased before lookup (see `normalize`).

_TEXT_AGGREGATE: Final[dict[str, str | None]] = {
    "anger": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "joy": Emotion.JOY,
    "neutral": Emotion.NEUTRAL,
    "sadness": Emotion.SADNESS,
    "surprise": Emotion.SURPRISE,
    "shame": None,  # excluded — see EXCLUSION_REASONS
}

# TESS filenames end in the emotion token, e.g. OAF_back_angry.wav -> "angry".
# "ps" is Toronto's abbreviation for "pleasant surprise".
_TESS: Final[dict[str, str | None]] = {
    "angry": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "happy": Emotion.JOY,
    "neutral": Emotion.NEUTRAL,
    "ps": Emotion.SURPRISE,
    "sad": Emotion.SADNESS,
}

# RAVDESS encodes emotion as the 3rd dash-separated field, "01".."08".
# "calm" has no canonical counterpart and is folded into neutral; this is the
# conventional treatment in the SER literature and is recorded in the model card.
_RAVDESS: Final[dict[str, str | None]] = {
    "01": Emotion.NEUTRAL,
    "02": Emotion.NEUTRAL,  # calm -> neutral
    "03": Emotion.JOY,
    "04": Emotion.SADNESS,
    "05": Emotion.ANGER,
    "06": Emotion.FEAR,
    "07": Emotion.DISGUST,
    "08": Emotion.SURPRISE,
    # spelled-out aliases, for hand-written configs and tests
    "neutral": Emotion.NEUTRAL,
    "calm": Emotion.NEUTRAL,
    "happy": Emotion.JOY,
    "sad": Emotion.SADNESS,
    "angry": Emotion.ANGER,
    "fearful": Emotion.FEAR,
    "disgust": Emotion.DISGUST,
    "surprised": Emotion.SURPRISE,
}

# CREMA-D uses 3-letter codes. It has no surprise class — a coverage gap that the
# per-corpus support table in reports/ must show rather than paper over.
_CREMA_D: Final[dict[str, str | None]] = {
    "ang": Emotion.ANGER,
    "dis": Emotion.DISGUST,
    "fea": Emotion.FEAR,
    "hap": Emotion.JOY,
    "neu": Emotion.NEUTRAL,
    "sad": Emotion.SADNESS,
}

# SAVEE prefixes the filename with a 1-2 char emotion code.
# Order matters when parsing: "sa" and "su" must be tried before "s".
_SAVEE: Final[dict[str, str | None]] = {
    "a": Emotion.ANGER,
    "d": Emotion.DISGUST,
    "f": Emotion.FEAR,
    "h": Emotion.JOY,
    "n": Emotion.NEUTRAL,
    "sa": Emotion.SADNESS,
    "su": Emotion.SURPRISE,
}

# FER-2013 ships as directories. sorted() gives:
#   angry, disgust, fear, happy, neutral, sad, surprise
# which maps 1:1 onto CANONICAL's alphabetical order. The v1 visual app instead
# hardcoded [...,'Sad','Surprise','Neutral'], scrambling indices 4/5/6.
# tests/unit/test_labels.py::test_fer2013_directory_order_is_canonical locks this.
_FER2013: Final[dict[str, str | None]] = {
    "angry": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "happy": Emotion.JOY,
    "neutral": Emotion.NEUTRAL,
    "sad": Emotion.SADNESS,
    "surprise": Emotion.SURPRISE,
}

# MELD's label set is already exactly canonical. This is the main reason it was
# chosen as the fusion corpus — no remapping, no dropped classes.
_MELD: Final[dict[str, str | None]] = {
    "anger": Emotion.ANGER,
    "disgust": Emotion.DISGUST,
    "fear": Emotion.FEAR,
    "joy": Emotion.JOY,
    "neutral": Emotion.NEUTRAL,
    "sadness": Emotion.SADNESS,
    "surprise": Emotion.SURPRISE,
}

# The inherited v1 speech model predicted gender x emotion. v2 strips the gender
# component: shipping a gender classifier as a side effect of emotion detection
# is a liability with no benefit to the stated task. Retained here solely so
# Phase 1 can score the v1 artifact against the canonical space.
_V1_SPEECH_CONV1D: Final[dict[str, str | None]] = {
    "female_angry": Emotion.ANGER,
    "female_calm": Emotion.NEUTRAL,
    "female_fearful": Emotion.FEAR,
    "female_happy": Emotion.JOY,
    "female_sad": Emotion.SADNESS,
    "male_angry": Emotion.ANGER,
    "male_calm": Emotion.NEUTRAL,
    "male_fearful": Emotion.FEAR,
    "male_happy": Emotion.JOY,
    "male_sad": Emotion.SADNESS,
}

SOURCE_MAPS: Final[dict[str, dict[str, str | None]]] = {
    "text_aggregate": _TEXT_AGGREGATE,
    "tess": _TESS,
    "ravdess": _RAVDESS,
    "crema_d": _CREMA_D,
    "savee": _SAVEE,
    "fer2013": _FER2013,
    "meld": _MELD,
    "v1_speech_conv1d": _V1_SPEECH_CONV1D,
}
"""Registry of every label vocabulary the project ingests."""


class UnknownLabelError(KeyError):
    """A raw label is not in the source's vocabulary.

    Raised rather than returning None so that a dataset-format change or a typo
    fails loudly at load time instead of quietly shrinking the training set.
    """


def normalize(raw: str, source: str) -> Emotion | None:
    """Map a dataset-native label onto the canonical space.

    Args:
        raw: The label as it appears in the dataset (case-insensitive).
        source: A key of :data:`SOURCE_MAPS`, e.g. ``"tess"``.

    Returns:
        The canonical :class:`Emotion`, or ``None`` if this label is
        deliberately excluded (see :data:`EXCLUSION_REASONS`).

    Raises:
        UnknownLabelError: If ``source`` is unregistered, or ``raw`` is not in
            that source's vocabulary.
    """
    try:
        mapping = SOURCE_MAPS[source]
    except KeyError:
        raise UnknownLabelError(
            f"unknown source {source!r}; expected one of {sorted(SOURCE_MAPS)}"
        ) from None

    key = raw.strip().lower()
    if key not in mapping:
        raise UnknownLabelError(
            f"label {raw!r} is not in the {source!r} vocabulary "
            f"({sorted(mapping)}). If this label is new, add it to "
            f"sentimentsphere.core.labels — do not map it ad hoc at the call site."
        )
    value = mapping[key]
    return Emotion(value) if value is not None else None


def to_index(label: str | Emotion) -> int:
    """Canonical label -> its index in :data:`CANONICAL`."""
    try:
        return INDEX[str(label)]
    except KeyError:
        raise UnknownLabelError(
            f"{label!r} is not canonical; expected one of {CANONICAL}. Call normalize() first."
        ) from None


def from_index(index: int) -> Emotion:
    """Index in a probability vector -> the canonical label it denotes."""
    if not 0 <= index < NUM_CLASSES:
        raise IndexError(f"index {index} out of range for {NUM_CLASSES} classes")
    try:
        return Emotion(CANONICAL[index])
    except IndexError:
        raise IndexError(f"index {index} out of range for {NUM_CLASSES} classes") from None


def excluded_labels(source: str) -> tuple[str, ...]:
    """Raw labels this source defines that v2 deliberately drops."""
    mapping = SOURCE_MAPS[source]
    return tuple(sorted(k for k, v in mapping.items() if v is None))

"""The probability contract is load-bearing: it must reject, never repair."""

from __future__ import annotations

import numpy as np
import pytest

from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES
from sentimentsphere.core.types import Prediction, probabilities


def uniform(n: int = 2) -> np.ndarray:
    return np.full((n, NUM_CLASSES), 1 / NUM_CLASSES)


def test_accepts_valid_matrix():
    assert probabilities(uniform()).shape == (2, NUM_CLASSES)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, NUM_CLASSES)),  # empty
        np.full((2, NUM_CLASSES - 1), 1 / (NUM_CLASSES - 1)),  # wrong width
        np.full(NUM_CLASSES, 1 / NUM_CLASSES),  # one-dimensional
        np.zeros((2, NUM_CLASSES)),  # rows do not sum to one
    ],
)
def test_rejects_malformed(bad):
    with pytest.raises(ValueError):
        probabilities(bad)


def test_rejects_nan_and_out_of_range():
    values = uniform()
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        probabilities(values)
    negative = np.zeros((1, NUM_CLASSES))
    negative[0, 0], negative[0, 1] = -0.5, 1.5
    with pytest.raises(ValueError):
        probabilities(negative)


def test_prediction_label_matches_argmax():
    scores = np.zeros(NUM_CLASSES)
    scores[3] = 1.0
    prediction = Prediction.from_scores(scores, model_id="m", modality="text", elapsed_ms=1.0)
    assert prediction.label == CANONICAL[3]
    assert prediction.scores[CANONICAL[3]] == pytest.approx(1.0)
    assert set(prediction.scores) == set(CANONICAL)


def test_prediction_rejects_unnormalized():
    with pytest.raises(ValueError):
        Prediction.from_scores(np.ones(NUM_CLASSES), model_id="m", modality="text", elapsed_ms=0.0)


def test_prediction_forbids_extra_fields():
    with pytest.raises(ValueError):
        Prediction(
            label=CANONICAL[0],
            scores=dict.fromkeys(CANONICAL, 1 / NUM_CLASSES),
            model_id="m",
            modality="text",
            elapsed_ms=0.0,
            surprise_field=1,  # type: ignore[call-arg]
        )

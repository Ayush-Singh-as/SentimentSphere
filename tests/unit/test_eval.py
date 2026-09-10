"""Metrics, calibration, and robustness: the numbers that get published."""

from __future__ import annotations

import numpy as np
import pytest

from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES
from sentimentsphere.eval.calibration import TemperatureScaler
from sentimentsphere.eval.metrics import (
    classification_metrics,
    reliability_bins,
    validate_targets,
)
from sentimentsphere.eval.robustness import add_noise, typo_text


def onehot(targets: list[int], classes: int = NUM_CLASSES) -> np.ndarray:
    """Perfectly confident, perfectly correct predictions."""
    return np.eye(classes)[targets]


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def test_perfect_predictions_score_one():
    targets = list(range(NUM_CLASSES))
    report = classification_metrics(targets, onehot(targets))
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["ece"] == pytest.approx(0.0)
    assert report["brier_score"] == pytest.approx(0.0)
    assert report["unsupported_classes"] == []


def test_macro_f1_counts_zero_support_classes():
    """A model evaluated on two classes must not look perfect over seven."""
    targets = [0, 0, 1, 1]
    report = classification_metrics(targets, onehot(targets))
    assert report["accuracy"] == 1.0
    assert report["supported_macro_f1"] == 1.0
    assert report["macro_f1"] == pytest.approx(2 / NUM_CLASSES)
    assert set(report["unsupported_classes"]) == set(CANONICAL[2:])


def test_confusion_matrix_orientation():
    # reference class 0, predicted class 1
    scores = np.zeros((1, NUM_CLASSES))
    scores[0, 1] = 1.0
    matrix = classification_metrics([0], scores)["confusion_matrix"]
    assert matrix[0][1] == 1
    assert matrix[1][0] == 0


def test_ece_detects_overconfidence():
    """Always fully confident, only half right -> ECE near 0.5."""
    targets = [0, 1] * 20
    scores = np.zeros((40, NUM_CLASSES))
    scores[:, 0] = 1.0
    assert classification_metrics(targets, scores)["ece"] == pytest.approx(0.5, abs=0.02)


def test_reliability_bins_partition_all_samples():
    targets = [0, 1, 2]
    bins = reliability_bins(targets, onehot(targets), bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 3
    assert bins[-1]["count"] == 3, "confidence 1.0 belongs in the top bin"


def test_reliability_bins_rejects_zero_bins():
    with pytest.raises(ValueError):
        reliability_bins([0], onehot([0]), bins=0)


@pytest.mark.parametrize("bad", [[0.5], [NUM_CLASSES], [-1]])
def test_validate_targets_rejects_bad_indices(bad):
    with pytest.raises(ValueError):
        validate_targets(np.asarray(bad), 1, NUM_CLASSES)


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], onehot([0]))


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def overconfident(seed: int = 0, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Correct 70% of the time but predicting ~0.99 — the classic miscalibration."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, NUM_CLASSES, size=n)
    logits = rng.normal(scale=0.3, size=(n, NUM_CLASSES))
    correct = rng.random(n) < 0.7
    logits[np.arange(n), np.where(correct, targets, (targets + 1) % NUM_CLASSES)] += 6.0
    return softmax_rows(logits), targets


def test_temperature_scaling_reduces_ece():
    scores, targets = overconfident()
    before = classification_metrics(targets, scores)["ece"]
    scaler = TemperatureScaler.fit(scores, targets)
    after = classification_metrics(targets, scaler.transform(scores))["ece"]
    assert scaler.temperature > 1.0, "an overconfident head needs softening"
    assert after < before


def test_temperature_scaling_preserves_argmax():
    scores, targets = overconfident()
    scaled = TemperatureScaler.fit(scores, targets).transform(scores)
    assert np.array_equal(scores.argmax(1), scaled.argmax(1))


def test_identity_temperature_is_a_noop():
    scores, _ = overconfident()
    assert np.allclose(TemperatureScaler(1.0).transform(scores), scores)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_temperature_must_be_positive_and_finite(bad):
    with pytest.raises(ValueError):
        TemperatureScaler(bad)


def test_calibration_refuses_zero_probability_target():
    """Fitting on a class the head can never emit would be an infinite loss."""
    scores = np.zeros((2, NUM_CLASSES))
    scores[:, 0] = 1.0
    with pytest.raises(ValueError, match="missing a development target class"):
        TemperatureScaler.fit(scores, np.array([0, 1]))


def test_transform_keeps_structurally_zero_classes_at_zero():
    scores = np.zeros((1, NUM_CLASSES))
    scores[0, :2] = 0.5
    out = TemperatureScaler(2.0).transform(scores)
    assert out[0, 2:].sum() == pytest.approx(0.0)
    assert out.sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #


def test_add_noise_hits_the_requested_snr():
    rng = np.random.default_rng(1)
    wave = rng.normal(size=8000)
    noisy = add_noise(wave, snr_db=10.0)
    measured = 10 * np.log10(np.mean(wave**2) / np.mean((noisy - wave) ** 2))
    assert measured == pytest.approx(10.0, abs=0.5)


def test_add_noise_is_seeded():
    wave = np.random.default_rng(2).normal(size=100)
    assert np.array_equal(add_noise(wave, 5.0, seed=3), add_noise(wave, 5.0, seed=3))
    assert not np.array_equal(add_noise(wave, 5.0, seed=3), add_noise(wave, 5.0, seed=4))


def test_add_noise_on_silence_is_unchanged():
    assert np.array_equal(add_noise(np.zeros(10), 5.0), np.zeros(10))


def test_add_noise_rejects_empty_or_nonfinite():
    with pytest.raises(ValueError):
        add_noise(np.array([]), 5.0)
    with pytest.raises(ValueError):
        add_noise(np.array([np.nan]), 5.0)


def test_typo_text_preserves_length_and_is_seeded():
    text = "the quick brown fox jumps over the lazy dog"
    out = typo_text(text, probability=0.3, seed=5)
    assert len(out) == len(text)
    assert out != text
    assert typo_text(text, 0.3, seed=5) == out


def test_typo_text_zero_probability_is_identity():
    assert typo_text("hello world", probability=0.0) == "hello world"


def test_typo_text_rejects_bad_probability():
    with pytest.raises(ValueError):
        typo_text("x", probability=1.5)

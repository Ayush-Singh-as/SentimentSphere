"""The sweep must measure degradation, not manufacture a flattering curve."""

from __future__ import annotations

import numpy as np
import pytest

from sentimentsphere.core.labels import NUM_CLASSES
from sentimentsphere.eval.robustness import text_sweep


def perfect_predictor(targets: list[int]):
    """Ignores its input and is always right — the clean-baseline reference."""

    def predict(texts: list[str]):
        return np.eye(NUM_CLASSES)[targets[: len(texts)]]

    return predict


def keyword_predictor(texts: list[str]) -> np.ndarray:
    """Brittle on purpose: only an exact token match scores class 1."""
    scores = np.full((len(texts), NUM_CLASSES), 1e-6)
    for i, text in enumerate(texts):
        scores[i, 1 if "wonderful" in text else 0] = 1.0
    return scores / scores.sum(axis=1, keepdims=True)


def test_sweep_reports_every_level():
    targets = [0, 1, 2, 3]
    texts = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    sweep = text_sweep(texts, targets, perfect_predictor(targets), levels=(0.0, 0.1, 0.5))
    assert sweep["samples"] == 4
    assert [level["typo_probability"] for level in sweep["levels"]] == [0.0, 0.1, 0.5]
    assert sweep["levels"][0]["macro_f1_delta"] == 0.0, "the clean run is the reference"


def test_sweep_detects_brittleness():
    """A keyword matcher must visibly degrade as its keyword gets scrambled."""
    texts = ["what a wonderful thing"] * 40
    targets = [1] * 40
    sweep = text_sweep(texts, targets, keyword_predictor, levels=(0.0, 0.4))
    clean, noisy = sweep["levels"]
    assert clean["accuracy"] == 1.0
    assert noisy["accuracy"] < clean["accuracy"]
    assert noisy["macro_f1_delta"] < 0


def test_sweep_is_deterministic():
    texts = ["alpha beta gamma"] * 10
    targets = [2] * 10
    a = text_sweep(texts, targets, keyword_predictor, levels=(0.3,), seed=5)
    b = text_sweep(texts, targets, keyword_predictor, levels=(0.3,), seed=5)
    assert a == b


def test_sweep_perturbs_identical_items_differently():
    """A single shared seed would corrupt every copy of a string identically."""
    seen: list[list[str]] = []
    texts = ["wonderful wonderful wonderful"] * 30
    targets = [1] * 30

    def capture(batch: list[str]) -> np.ndarray:
        seen.append(list(batch))
        return keyword_predictor(batch)

    text_sweep(texts, targets, capture, levels=(0.0, 0.4), seed=1)
    perturbed = seen[1]
    assert len(set(perturbed)) > 1, "identical inputs must not share one perturbation"


def test_sweep_rejects_mismatched_input():
    with pytest.raises(ValueError):
        text_sweep(["a"], [0, 1], keyword_predictor)
    with pytest.raises(ValueError):
        text_sweep([], [], keyword_predictor)

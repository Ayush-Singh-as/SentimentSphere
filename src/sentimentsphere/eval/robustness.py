"""Seeded perturbations for repeatable robustness evaluation, not augmentation claims."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from sentimentsphere.core.types import FloatArray
from sentimentsphere.eval.metrics import classification_metrics


def add_noise(waveform: FloatArray, snr_db: float, seed: int = 1337) -> FloatArray:
    values = np.asarray(waveform, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite(snr_db) or not values.size:
        raise ValueError("Expected finite nonempty waveform and SNR")
    power = float(np.mean(values**2))
    if not power:
        return values.copy()
    noise = np.random.default_rng(seed).normal(size=values.shape)
    noise *= np.sqrt(power / (10 ** (snr_db / 10) * np.mean(noise**2)))
    return values + noise


def typo_text(text: str, probability: float = 0.05, seed: int = 1337) -> str:
    if not 0 <= probability <= 1:
        raise ValueError("Typo probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i].isalpha() and chars[i + 1].isalpha() and rng.random() < probability:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 1
        i += 1
    return "".join(chars)


def text_sweep(
    texts: Sequence[str],
    targets: Sequence[int],
    predict: Callable[[list[str]], Any],
    *,
    levels: Sequence[float] = (0.0, 0.02, 0.05, 0.10, 0.20),
    seed: int = 1337,
) -> dict[str, Any]:
    """Score a text head under increasing character-transposition noise.

    Reported as degradation from the clean baseline, because the absolute
    number at a given typo rate means little on its own.
    """
    if not texts or len(texts) != len(targets):
        raise ValueError("texts and targets must be nonempty and the same length")
    results = []
    baseline: float | None = None
    for level in levels:
        perturbed = (
            list(texts)
            if not level
            # Vary the seed per item so every string is not perturbed identically.
            else [typo_text(t, level, seed=seed + i) for i, t in enumerate(texts)]
        )
        metrics = classification_metrics(targets, predict(perturbed))
        if baseline is None:
            baseline = metrics["macro_f1"]
        results.append(
            {
                "typo_probability": level,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "macro_f1_delta": metrics["macro_f1"] - baseline,
            }
        )
    return {
        "perturbation": "adjacent-character transposition, seeded per item",
        "samples": len(texts),
        "levels": results,
    }

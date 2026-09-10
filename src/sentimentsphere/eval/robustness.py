"""Seeded perturbations for repeatable robustness evaluation, not augmentation claims."""

from __future__ import annotations

import numpy as np

from sentimentsphere.core.types import FloatArray


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

"""Single-temperature probability calibration fitted on development data only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

from sentimentsphere.core.types import FloatArray, probabilities
from sentimentsphere.eval.metrics import validate_targets


@dataclass(frozen=True)
class TemperatureScaler:
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature must be finite and positive")

    def transform(self, scores: Any) -> FloatArray:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Expected a probability matrix")
        probabilities(values, classes=values.shape[1])
        # Keep unavailable classes at exactly zero when calibrating a partial head.
        with np.errstate(divide="ignore"):
            logits = np.log(values) / self.temperature
        return np.asarray(softmax(logits, axis=1), dtype=np.float64)

    @classmethod
    def fit(cls, dev_scores: Any, dev_targets: Any) -> TemperatureScaler:
        values = np.asarray(dev_scores, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("Expected a probability matrix")
        probabilities(values, classes=values.shape[1])
        truth = validate_targets(dev_targets, len(values), values.shape[1])
        if np.any(values[np.arange(len(values)), truth] == 0):
            raise ValueError("Cannot calibrate a head missing a development target class")
        with np.errstate(divide="ignore"):
            logits = np.log(values)

        def nll(log_temperature: float) -> float:
            scaled = logits / np.exp(log_temperature)
            return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(truth)), truth]))

        result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        if not result.success or result.fun > nll(0.0):
            return cls()
        return cls(float(np.exp(result.x)))

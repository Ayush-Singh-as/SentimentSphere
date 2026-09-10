"""Validated shared inference contracts; vector order is never caller-defined."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from sentimentsphere.core.labels import CANONICAL, NUM_CLASSES, Emotion

FloatArray = NDArray[np.float64]
Modality = Literal["text", "audio", "image", "fusion"]


def probabilities(values: Any, *, classes: int = NUM_CLASSES) -> FloatArray:
    """Validate, rather than silently repair, a probability matrix."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != classes or not len(array):
        raise ValueError(f"Expected nonempty (n, {classes}) probabilities, got {array.shape}")
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probability rows must sum to one")
    return array


class Prediction(BaseModel):
    """One canonical distribution with model identity and measured elapsed time."""

    model_config = ConfigDict(extra="forbid")
    label: Emotion
    scores: dict[str, float]
    model_id: str
    modality: Modality
    elapsed_ms: float = Field(ge=0)
    calibrated: bool = False
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_scores(
        cls,
        scores: Any,
        *,
        model_id: str,
        modality: Modality,
        elapsed_ms: float,
        calibrated: bool = False,
        warnings: list[str] | None = None,
    ) -> Prediction:
        values = probabilities(np.asarray(scores).reshape(1, -1))[0]
        return cls(
            label=Emotion(CANONICAL[int(np.argmax(values))]),
            scores=dict(zip(CANONICAL, values.tolist(), strict=True)),
            model_id=model_id,
            modality=modality,
            elapsed_ms=elapsed_ms,
            calibrated=calibrated,
            warnings=warnings or [],
        )

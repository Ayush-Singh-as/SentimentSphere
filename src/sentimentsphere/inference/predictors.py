"""Local predictors load verified trained artifacts and own preprocessing/calibration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib

from sentimentsphere.core.config import Settings
from sentimentsphere.core.types import Prediction
from sentimentsphere.eval.calibration import TemperatureScaler
from sentimentsphere.inference.artifacts import (
    ModelUnavailableError,
    align_probabilities,
    registered_path,
    verify_bundle,
)


class TextPredictor:
    def __init__(self, directory: Path, settings: Settings) -> None:
        self.metadata = verify_bundle(directory)
        if self.metadata.get("kind") != "sklearn_text":
            raise ModelUnavailableError("Unsupported text artifact kind")
        self.model = joblib.load(directory / "model.joblib")
        self.calibration = TemperatureScaler(float(self.metadata["temperature"]))
        self.settings = settings

    def predict(self, text: str) -> Prediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter nonempty text")
        if len(text) > self.settings.max_text_chars:
            raise ValueError(f"Text exceeds {self.settings.max_text_chars} characters")
        started = time.perf_counter()
        scores = align_probabilities(self.model.predict_proba([text]), self.model.classes_)
        return Prediction.from_scores(
            self.calibration.transform(scores)[0],
            model_id=self.metadata["model_id"],
            modality="text",
            elapsed_ms=(time.perf_counter() - started) * 1000,
            calibrated=bool(self.metadata.get("calibrated")),
        )


class PredictorRegistry:
    """Application-owned cache; resources are loaded once per server process."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._predictors: dict[str, Any] = {}

    def get(self, modality: str) -> Any:
        if modality not in self._predictors:
            directory = registered_path(self.settings.artifacts_dir, modality)
            if modality == "text":
                self._predictors[modality] = TextPredictor(directory, self.settings)
            else:
                raise ModelUnavailableError(f"No implemented predictor for {modality}")
        return self._predictors[modality]

    def describe(self) -> dict[str, Any]:
        result = {}
        for modality in ("text", "audio", "image", "fusion"):
            try:
                metadata = verify_bundle(registered_path(self.settings.artifacts_dir, modality))
                result[modality] = {
                    "available": True,
                    "model_id": metadata["model_id"],
                    "calibrated": metadata.get("calibrated", False),
                }
            except ModelUnavailableError:
                result[modality] = {"available": False}
        return result

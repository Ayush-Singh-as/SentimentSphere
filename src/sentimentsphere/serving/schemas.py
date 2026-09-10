"""Request and response contracts.

Responses carry the model id and the calibration flag, so a caller can always
tell which weights produced a number and whether its confidence was calibrated.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.core.types import Modality


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=100_000)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    scores: dict[str, float]
    model_id: str
    modality: Modality
    elapsed_ms: float
    calibrated: bool
    warnings: list[str] = Field(default_factory=list)
    request_id: str


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str
    code: str
    request_id: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    version: str
    labels: list[str] = Field(default_factory=lambda: list(CANONICAL))


class ModelsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    labels: list[str] = Field(default_factory=lambda: list(CANONICAL))
    models: dict[str, dict[str, object]]

"""Validated local paths, serving limits, and reproducible training configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment overrides use SPHERE_; credentials are never part of run configs."""

    model_config = SettingsConfigDict(env_prefix="SPHERE_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    manifests_dir: Path = Path("manifests")
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    max_audio_seconds: float = Field(default=30, gt=0, le=300)
    max_video_seconds: float = Field(default=30, gt=0, le=300)
    max_image_pixels: int = Field(default=16_000_000, gt=0)
    max_text_chars: int = Field(default=5000, ge=1)
    requests_per_minute: int = Field(default=30, ge=1)
    inference_concurrency: int = Field(default=1, ge=1, le=8)


class TrainConfig(BaseModel):
    """Unknown keys fail so misspelled hyperparameters cannot silently change a run."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    modality: Literal["text", "audio", "image", "fusion"]
    architecture: Literal["tfidf_lr", "mfcc_svm", "deberta", "wavlm", "convnext", "meta_lr"]
    dataset: str
    manifest: Path
    seed: int = Field(default=1337, ge=0, le=2**32 - 1)
    epochs: int = Field(default=5, ge=1)
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=3e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_length: int = Field(default=256, ge=8)
    pretrained: str | None = None
    revision: str | None = None
    class_weight: Literal["balanced"] | None = "balanced"
    regularization: float = Field(default=4.0, gt=0)
    max_features: int = Field(default=100_000, ge=100)
    deterministic: bool = True
    device: str = "auto"
    num_workers: int = Field(default=0, ge=0)
    gradient_accumulation: int = Field(default=1, ge=1)
    warmup_ratio: float = Field(default=0.1, ge=0, lt=1)
    layer_lr_decay: float = Field(default=0.9, gt=0, le=1)
    track: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        with Path(path).open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
        return cls.model_validate(payload)

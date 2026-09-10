"""Fit on train, calibrate on dev, evaluate once on a frozen test partition."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from sentimentsphere.core.config import Settings, TrainConfig
from sentimentsphere.core.io import write_json
from sentimentsphere.core.provenance import collect
from sentimentsphere.core.seeding import seed_everything
from sentimentsphere.data.records import Dataset
from sentimentsphere.data.splits import SplitManifest
from sentimentsphere.eval.calibration import TemperatureScaler
from sentimentsphere.eval.metrics import classification_metrics
from sentimentsphere.eval.report import write_report
from sentimentsphere.inference.artifacts import (
    align_probabilities,
    register_model,
    save_sklearn_bundle,
)
from sentimentsphere.models.text import make_text_model


def train_text(config: TrainConfig, dataset: Dataset, settings: Settings) -> dict[str, Any]:
    if config.architecture != "tfidf_lr" or config.modality != "text":
        raise ValueError("Text baseline requires modality=text and architecture=tfidf_lr")
    manifest = SplitManifest.load(config.manifest)
    manifest.validate_dataset(dataset)
    seed_report = seed_everything(config.seed, deterministic=config.deterministic)
    train = manifest.samples(dataset, "train")
    dev = manifest.samples(dataset, "dev")
    test = manifest.samples(dataset, "test")
    model = make_text_model(config)
    model.fit([s.text for s in train], np.array([s.label for s in train]))
    dev_scores = align_probabilities(model.predict_proba([s.text for s in dev]), model.classes_)
    calibration = TemperatureScaler.fit(dev_scores, np.array([s.label for s in dev]))
    raw = align_probabilities(model.predict_proba([s.text for s in test]), model.classes_)
    calibrated = calibration.transform(raw)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{config.name}-{config.seed}-{timestamp}"
    from dataclasses import asdict

    provenance = collect(
        config.model_dump(mode="json"),
        split_manifest=manifest.fingerprint,
        dataset=dataset.fingerprint,
        seed=asdict(seed_report),
    ).to_dict()
    report = {
        "run_id": run_id,
        "model": config.architecture,
        "config": config.model_dump(mode="json"),
        "provenance": provenance,
        "evaluation": "frozen-test; model fit=train; temperature fit=dev",
        "temperature": calibration.temperature,
        "uncalibrated_metrics": classification_metrics([s.label for s in test], raw),
        "metrics": classification_metrics([s.label for s in test], calibrated),
    }
    directory = settings.artifacts_dir / "models" / run_id
    save_sklearn_bundle(
        directory,
        model,
        {
            "model_id": run_id,
            "kind": "sklearn_text",
            "modality": "text",
            "temperature": calibration.temperature,
            "calibrated": True,
            "provenance": provenance,
            "config": config.model_dump(mode="json"),
            "training_ids": [s.id for s in train],
            "calibration_ids": [s.id for s in dev],
        },
    )
    write_report(settings.reports_dir / run_id, report)
    write_json(
        settings.reports_dir / run_id / "predictions.json",
        {
            "split_manifest": manifest.fingerprint,
            "ids": [s.id for s in test],
            "targets": [s.label for s in test],
            "probabilities": calibrated.tolist(),
        },
    )
    register_model(settings.artifacts_dir, "text", directory)
    return report

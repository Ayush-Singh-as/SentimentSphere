"""Historical artifact audit and text rescoring with explicit comparability limits."""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from sentimentsphere.core.config import Settings
from sentimentsphere.core.io import sha256_file, write_json
from sentimentsphere.core.provenance import collect
from sentimentsphere.eval.metrics import classification_metrics
from sentimentsphere.eval.report import write_report


def inspect_h5(path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        payload = json.loads(handle.attrs.get("model_config", "{}"))
        config = payload.get("config", {})
        layers = config.get("layers", []) if isinstance(config, dict) else config
        weights = []

        def visit(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                weights.append({"name": name, "shape": list(value.shape)})

        handle.visititems(visit)
        return {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "keras_version": str(handle.attrs.get("keras_version", "unknown")),
            "layers": [
                {"type": layer["class_name"], "config": layer["config"]} for layer in layers
            ],
            "weight_datasets": weights,
            "usable": bool(layers and weights),
        }


def score_legacy_text(settings: Settings) -> dict[str, Any]:
    """Reproduce the historical raw/cleaned comparison, not an independent holdout claim."""
    import neattext.functions as nfx

    data_path = settings.data_dir / "raw" / "text_emotion_aggregate.csv"
    model_path = settings.artifacts_dir / "v1" / "v1_text_countvec_lr.pkl"
    with data_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    targets = np.array([row["Emotion"] for row in rows])
    _, selected = train_test_split(
        np.arange(len(rows)), test_size=0.2, random_state=42, stratify=targets
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(model_path)
    vocabulary = tuple(str(label) for label in model.classes_)
    truth = np.array([vocabulary.index(str(targets[i])) for i in selected])
    raw = [rows[i]["Text"] for i in selected]
    cleaned = [nfx.remove_stopwords(nfx.remove_userhandles(text)) for text in raw]
    raw_scores = model.predict_proba(raw)
    clean_scores = model.predict_proba(cleaned)
    provenance = collect(
        {"split": "stratified_20pct", "seed": 42, "labels": vocabulary},
        dataset_sha256=sha256_file(data_path),
        model_sha256=sha256_file(model_path),
        test_row_indices=selected.tolist(),
    ).to_dict()
    report = {
        "model": "v1_text_countvec_lr",
        "provenance": provenance,
        "warnings": sorted({str(w.message) for w in caught}),
        "evaluation": "historical-rescore; original training membership unknown; eight classes; raw duplicates retained",
        "metrics": classification_metrics(truth, raw_scores, labels=vocabulary),
        "cleaned_metrics": classification_metrics(truth, clean_scores, labels=vocabulary),
        "prediction_change_fraction": float(
            np.mean(raw_scores.argmax(1) != clean_scores.argmax(1))
        ),
    }
    write_report(settings.reports_dir / "baseline" / "text-raw", report)
    write_report(
        settings.reports_dir / "baseline" / "text-cleaned",
        {
            **report,
            "metrics": report["cleaned_metrics"],
        },
    )
    return report


def audit_legacy(settings: Settings) -> dict[str, Any]:
    """Record each artifact honestly; missing datasets never become fabricated scores."""
    results: dict[str, Any] = {}
    try:
        text = score_legacy_text(settings)
        results["text"] = {
            "status": "rescored",
            "raw_accuracy": text["metrics"]["accuracy"],
            "cleaned_accuracy": text["cleaned_metrics"]["accuracy"],
        }
    except (OSError, ValueError, ImportError) as error:
        results["text"] = {"status": "unavailable", "reason": str(error)}
    for modality, filename in (
        ("audio", "v1_speech_conv1d.h5"),
        ("image", "v1_vision_cnn.h5"),
        ("tess_empty", "v1_tess_lstm_EMPTY.h5"),
    ):
        try:
            metadata = inspect_h5(settings.artifacts_dir / "v1" / filename)
            results[modality] = {
                "status": "requires_evaluation" if metadata["usable"] else "invalid_empty_artifact",
                "artifact": metadata,
            }
        except (OSError, ValueError, ImportError) as error:
            results[modality] = {"status": "unavailable", "reason": str(error)}
    write_json(settings.reports_dir / "baseline" / "inventory.json", results)
    return results

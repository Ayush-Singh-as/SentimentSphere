"""End-to-end: load -> freeze split -> train -> calibrate -> register -> predict.

Small synthetic corpus, so this runs in CI without any downloaded dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentimentsphere.core.config import Settings, TrainConfig
from sentimentsphere.core.io import read_json
from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.data.catalog import load_dataset, prepare_dataset
from sentimentsphere.data.splits import SplitManifest
from sentimentsphere.inference.artifacts import ModelUnavailableError
from sentimentsphere.inference.predictors import PredictorRegistry
from sentimentsphere.training.baseline import train_text

VOCAB = {
    "joy": ["delighted", "wonderful", "thrilled", "joyful", "great news"],
    "anger": ["furious", "outraged", "livid", "infuriating", "so angry"],
    "disgust": ["revolting", "disgusting", "gross", "nauseating", "vile"],
    "fear": ["terrified", "afraid", "scared", "frightening", "dreadful"],
    "neutral": ["the meeting", "at three", "on the table", "a report", "some notes"],
    "sadness": ["heartbroken", "so sad", "miserable", "grieving", "i miss them"],
    "surprise": ["astonished", "unexpected", "no way", "shocking", "startled"],
}


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Settings, TrainConfig]:
    """A learnable synthetic corpus with a frozen manifest, all under tmp_path."""
    settings = Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        manifests_dir=tmp_path / "manifests",
    )
    csv_path = settings.data_dir / "raw" / "text_emotion_aggregate.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Emotion", "Text"])
        for emotion, phrases in VOCAB.items():
            for i in range(40):
                writer.writerow([emotion, f"{phrases[i % len(phrases)]} example {i}"])
    summary = prepare_dataset("text_aggregate", settings, seed=1337)
    config = TrainConfig(
        name="test_text",
        modality="text",
        architecture="tfidf_lr",
        dataset="text_aggregate",
        manifest=Path(summary["manifests"][0]),
        seed=1337,
        max_features=5000,
    )
    return settings, config


def test_prepare_writes_manifest_and_audit(prepared):
    settings, config = prepared
    assert config.manifest.exists()
    audit = read_json(settings.reports_dir / "data" / "text_aggregate.json")
    assert audit["fingerprint"]
    assert audit["audit"]["counts"]["retained_rows"] == 280


def test_full_train_and_predict_cycle(prepared):
    settings, config = prepared
    dataset = load_dataset("text_aggregate", settings)
    report = train_text(config, dataset, settings)

    # the report is complete and honestly stamped
    assert report["metrics"]["labels"] == list(CANONICAL)
    assert report["provenance"]["config_hash"]
    assert report["provenance"]["extra"]["split_manifest"]
    assert report["metrics"]["accuracy"] > 0.5, "synthetic corpus should be learnable"

    # reports and figures landed on disk
    directory = settings.reports_dir / report["run_id"]
    assert (directory / "metrics.json").exists()
    assert (directory / "confusion.svg").exists()
    assert (directory / "calibration.svg").exists()

    # test-set predictions are recoverable for later fusion work
    predictions = read_json(directory / "predictions.json")
    assert len(predictions["ids"]) == len(predictions["probabilities"])

    # and the registered model round-trips through the predictor
    registry = PredictorRegistry(settings)
    assert registry.describe()["text"]["available"] is True
    prediction = registry.get("text").predict("what wonderful news, i am thrilled")
    assert prediction.label in CANONICAL
    assert prediction.calibrated is True
    assert sum(prediction.scores.values()) == pytest.approx(1.0)


def test_predictor_rejects_empty_and_oversized_text(prepared):
    settings, config = prepared
    train_text(config, load_dataset("text_aggregate", settings), settings)
    predictor = PredictorRegistry(settings).get("text")
    with pytest.raises(ValueError):
        predictor.predict("   ")
    with pytest.raises(ValueError, match="exceeds"):
        predictor.predict("x" * (settings.max_text_chars + 1))


def test_unavailable_modalities_are_reported_not_faked(prepared):
    settings, _ = prepared
    registry = PredictorRegistry(settings)
    described = registry.describe()
    assert described["audio"] == {"available": False}
    with pytest.raises(ModelUnavailableError):
        registry.get("audio")


def test_training_refuses_a_stale_manifest(prepared):
    """Changing the data after freezing a split must not silently retrain."""
    settings, config = prepared
    dataset = load_dataset("text_aggregate", settings)
    manifest = SplitManifest.load(config.manifest)
    csv_path = settings.data_dir / "raw" / "text_emotion_aggregate.csv"
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(["joy", "a brand new unseen row"])
    changed = load_dataset("text_aggregate", settings)
    assert changed.fingerprint != dataset.fingerprint
    with pytest.raises(ValueError):
        manifest.validate_dataset(changed)


def test_config_architecture_is_enforced(prepared):
    settings, config = prepared
    wrong = config.model_copy(update={"architecture": "deberta"})
    with pytest.raises(ValueError, match="tfidf_lr"):
        train_text(wrong, load_dataset("text_aggregate", settings), settings)

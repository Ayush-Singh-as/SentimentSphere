"""API contract tests, including the two behaviours v1 got wrong.

An absent model must return 503, and every answer must name the model that
produced it.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentimentsphere.core.config import Settings, TrainConfig
from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.data.catalog import load_dataset, prepare_dataset
from sentimentsphere.serving.api import create_app
from sentimentsphere.training.baseline import train_text

PHRASES = {
    "joy": "delighted and thrilled",
    "anger": "furious and outraged",
    "disgust": "revolting and vile",
    "fear": "terrified and afraid",
    "neutral": "the meeting at three",
    "sadness": "heartbroken and miserable",
    "surprise": "astonished and unexpected",
}


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        reports_dir=tmp_path / "reports",
        manifests_dir=tmp_path / "manifests",
    )


@pytest.fixture
def empty_client(tmp_path: Path):
    """A server with no installed model — must refuse, not improvise."""
    with TestClient(create_app(build_settings(tmp_path))) as client:
        yield client


@pytest.fixture
def trained_client(tmp_path: Path):
    settings = build_settings(tmp_path)
    csv_path = settings.data_dir / "raw" / "text_emotion_aggregate.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Emotion", "Text"])
        for emotion, phrase in PHRASES.items():
            for i in range(40):
                writer.writerow([emotion, f"{phrase} sample {i}"])
    summary = prepare_dataset("text_aggregate", settings, seed=1337)
    config = TrainConfig(
        name="api_text",
        modality="text",
        architecture="tfidf_lr",
        dataset="text_aggregate",
        manifest=Path(summary["manifests"][0]),
        max_features=2000,
    )
    train_text(config, load_dataset("text_aggregate", settings), settings)
    with TestClient(create_app(settings)) as client:
        yield client


# --------------------------------------------------------------------------- #
# ops
# --------------------------------------------------------------------------- #


def test_healthz(empty_client):
    response = empty_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["labels"] == list(CANONICAL)


def test_request_id_is_returned_and_echoed(empty_client):
    assert empty_client.get("/healthz").headers["x-request-id"]
    supplied = empty_client.get("/healthz", headers={"x-request-id": "abc123"})
    assert supplied.headers["x-request-id"] == "abc123"


def test_models_reports_unavailable_without_lying(empty_client):
    body = empty_client.get("/v1/models").json()
    assert body["models"]["text"]["available"] is False
    assert body["labels"] == list(CANONICAL)


def test_openapi_renders(empty_client):
    schema = empty_client.get("/openapi.json").json()
    assert "/v1/predict/text" in schema["paths"]
    assert "/v1/predict/audio" in schema["paths"], "missing modalities stay in the contract"


# --------------------------------------------------------------------------- #
# prediction
# --------------------------------------------------------------------------- #


def test_missing_model_returns_503_not_a_guess(empty_client):
    response = empty_client.post("/v1/predict/text", json={"text": "hello"})
    assert response.status_code == 503
    body = response.json()
    assert body["code"] == "model_unavailable"
    assert "label" not in body


def test_predict_text_succeeds_and_names_its_model(trained_client):
    response = trained_client.post("/v1/predict/text", json={"text": "delighted and thrilled"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CANONICAL
    assert body["model_id"].startswith("api_text-")
    assert body["calibrated"] is True
    assert sum(body["scores"].values()) == pytest.approx(1.0)
    assert set(body["scores"]) == set(CANONICAL)
    assert body["request_id"] == response.headers["x-request-id"]


def test_models_available_after_training(trained_client):
    assert trained_client.get("/v1/models").json()["models"]["text"]["available"] is True


@pytest.mark.parametrize(
    "payload",
    [{}, {"text": ""}, {"text": "hi", "extra": 1}, {"text": 5}],
)
def test_malformed_requests_are_422(trained_client, payload):
    assert trained_client.post("/v1/predict/text", json=payload).status_code == 422


def test_oversized_text_is_rejected(trained_client):
    response = trained_client.post("/v1/predict/text", json={"text": "x" * 6000})
    assert response.status_code == 422
    assert response.json()["code"] == "text_too_long"


def test_whitespace_only_text_is_rejected(trained_client):
    response = trained_client.post("/v1/predict/text", json={"text": "   "})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_input"


@pytest.mark.parametrize("modality", ["audio", "image", "video"])
def test_untrained_modalities_return_501(trained_client, modality):
    response = trained_client.post(f"/v1/predict/{modality}")
    assert response.status_code == 501
    assert response.json()["code"] == "not_implemented"


def test_rate_limit_returns_429_but_spares_ops_routes(tmp_path: Path):
    settings = build_settings(tmp_path)
    settings.requests_per_minute = 3
    with TestClient(create_app(settings)) as client:
        codes = [
            client.post("/v1/predict/text", json={"text": "hello"}).status_code for _ in range(5)
        ]
        assert codes.count(429) == 2, "the limiter must engage after the configured budget"
        assert client.get("/healthz").status_code == 200, "health checks must stay reachable"

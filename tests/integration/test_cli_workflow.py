"""The documented workflow, exercised as a user runs it: data -> train -> predict."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from sentimentsphere.cli import app

runner = CliRunner()

PHRASES = {
    "joy": "delighted and thrilled",
    "anger": "furious and outraged",
    "disgust": "revolting and vile",
    "fear": "terrified and afraid",
    "neutral": "the meeting at three",
    "sadness": "heartbroken and miserable",
    "surprise": "astonished and unexpected",
}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway project root, so the CLI's default relative paths are safe."""
    csv_path = tmp_path / "data" / "raw" / "text_emotion_aggregate.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Emotion", "Text"])
        for emotion, phrase in PHRASES.items():
            for i in range(40):
                writer.writerow([emotion, f"{phrase} sample {i}"])
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "text.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "cli_text",
                "modality": "text",
                "architecture": "tfidf_lr",
                "dataset": "text_aggregate",
                "manifest": "manifests/text_aggregate-v1.json",
                "max_features": 2000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_data_then_train_then_predict(project: Path):
    data = runner.invoke(app, ["data", "--dataset", "text_aggregate"])
    assert data.exit_code == 0, data.output
    assert (project / "manifests" / "text_aggregate-v1.json").exists()

    train = runner.invoke(app, ["train", "--config", "configs/text.yaml"])
    assert train.exit_code == 0, train.output
    assert "accuracy" in train.output

    predict = runner.invoke(app, ["predict", "--text", "delighted and thrilled", "--json"])
    assert predict.exit_code == 0, predict.output
    body = json.loads(predict.output)
    assert body["model_id"].startswith("cli_text-")
    assert body["calibrated"] is True
    assert sum(body["scores"].values()) == pytest.approx(1.0)


def test_predict_table_output(project: Path):
    runner.invoke(app, ["data", "--dataset", "text_aggregate"])
    runner.invoke(app, ["train", "--config", "configs/text.yaml"])
    result = runner.invoke(app, ["predict", "--text", "furious and outraged"])
    assert result.exit_code == 0
    assert "probability" in result.output


def test_data_is_idempotent(project: Path):
    """Re-freezing an unchanged dataset must not error or rewrite the manifest."""
    first = runner.invoke(app, ["data", "--dataset", "text_aggregate"])
    path = project / "manifests" / "text_aggregate-v1.json"
    before = path.read_bytes()
    second = runner.invoke(app, ["data", "--dataset", "text_aggregate"])
    assert first.exit_code == second.exit_code == 0
    assert path.read_bytes() == before


def test_reseeding_a_frozen_split_is_refused(project: Path):
    """A frozen manifest is frozen: a different seed must not silently replace it."""
    assert runner.invoke(app, ["data", "--dataset", "text_aggregate"]).exit_code == 0
    result = runner.invoke(app, ["data", "--dataset", "text_aggregate", "--seed", "99"])
    assert result.exit_code == 1
    assert "unavailable" in result.output


def test_train_refuses_an_unimplemented_architecture(project: Path):
    runner.invoke(app, ["data", "--dataset", "text_aggregate"])
    (project / "configs" / "deberta.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "cli_deberta",
                "modality": "text",
                "architecture": "deberta",
                "dataset": "text_aggregate",
                "manifest": "manifests/text_aggregate-v1.json",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["train", "--config", "configs/deberta.yaml"])
    assert result.exit_code == 1
    assert "No implemented trainer" in result.output


def test_info_json_is_parseable():
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    assert "git_sha" in json.loads(result.output)

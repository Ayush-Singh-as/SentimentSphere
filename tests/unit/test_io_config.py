"""Atomic writes and validated configs — the two things a bad run must not corrupt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentimentsphere.core.config import Settings, TrainConfig
from sentimentsphere.core.io import read_json, sha256_file, stable_digest, write_json


def test_write_json_is_atomic_on_failure(tmp_path: Path):
    target = tmp_path / "metrics.json"
    write_json(target, {"ok": 1})
    with pytest.raises((TypeError, ValueError)):
        write_json(target, {"bad": {1, 2}})  # sets are not JSON-serializable
    assert read_json(target) == {"ok": 1}
    assert not list(tmp_path.glob(".metrics.json.*")), "temp file left behind"


def test_write_json_rejects_nan(tmp_path: Path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "n.json", {"x": float("nan")})


def test_read_json_rejects_non_object(tmp_path: Path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        read_json(path)


def test_stable_digest_is_key_order_independent():
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
    assert stable_digest({"a": 1}) != stable_digest({"a": 2})


def test_sha256_file_is_full_length(tmp_path: Path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_settings_env_prefix(monkeypatch):
    monkeypatch.setenv("SPHERE_MAX_TEXT_CHARS", "42")
    assert Settings().max_text_chars == 42


def test_settings_reject_out_of_range(monkeypatch):
    monkeypatch.setenv("SPHERE_MAX_UPLOAD_MB", "9999")
    with pytest.raises(ValueError):
        Settings()


def test_train_config_forbids_unknown_keys(tmp_path: Path):
    payload = {
        "name": "x",
        "modality": "text",
        "architecture": "tfidf_lr",
        "dataset": "d",
        "manifest": "m.json",
        "lerning_rate": 1e-3,  # typo must fail loudly
    }
    path = tmp_path / "c.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(path)


def test_shipped_text_config_loads():
    config = TrainConfig.from_yaml("configs/text/tfidf_lr.yaml")
    assert (config.modality, config.architecture) == ("text", "tfidf_lr")

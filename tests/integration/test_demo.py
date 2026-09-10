"""The demo must build and must never invent a prediction for an absent model."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio", reason="demo extra not installed")

APP_DIR = Path(__file__).resolve().parents[2] / "apps" / "space"


@pytest.fixture
def demo_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Import apps/space/app.py against an empty, temporary artifacts root."""
    monkeypatch.setenv("SPHERE_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SPHERE_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.syspath_prepend(str(APP_DIR))
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    yield module
    sys.modules.pop("app", None)


def test_demo_builds(demo_module):
    assert isinstance(demo_module.build(), gr.Blocks)


def test_classify_reports_a_missing_model(demo_module):
    scores, detail = demo_module.classify("hello there")
    assert scores == {}, "no model installed must yield no scores at all"
    assert "No text model is installed" in detail


def test_classify_rejects_empty_text(demo_module):
    scores, detail = demo_module.classify("   ")
    assert scores == {}
    assert "Enter some text" in detail


def test_metrics_tab_handles_no_reports(demo_module):
    assert "No reports generated yet" in demo_module.metrics_markdown()


def test_about_names_the_limitations(demo_module):
    about = demo_module.ABOUT
    assert "predicted expression category" in about
    assert "label noise" in about
    assert "Barrett" in about


def test_pending_tabs_say_they_are_unimplemented(demo_module):
    assert "Not implemented" in demo_module.PENDING.format(name="voice")

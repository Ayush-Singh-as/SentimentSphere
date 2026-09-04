"""Smoke tests for the ``sphere`` CLI.

The Phase 0 exit criterion is that ``sphere --help`` works on a clean install,
so it is a test rather than a manual check.
"""

from __future__ import annotations

from typer.testing import CliRunner

from sentimentsphere import __version__
from sentimentsphere.cli import app

runner = CliRunner()


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sphere" in result.output


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_labels_lists_canonical_space():
    result = runner.invoke(app, ["labels"])
    assert result.exit_code == 0
    for label in ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"):
        assert label in result.output


def test_labels_for_one_source():
    result = runner.invoke(app, ["labels", "--source", "tess"])
    assert result.exit_code == 0
    assert "ps" in result.output


def test_labels_rejects_unknown_source():
    result = runner.invoke(app, ["labels", "--source", "nope"])
    assert result.exit_code == 1


def test_info_json():
    result = runner.invoke(app, ["info", "--json"])
    assert result.exit_code == 0
    assert "python_version" in result.output


def test_unimplemented_commands_exit_nonzero():
    """A command that cannot do its job must fail loudly.

    v1's speech app printed an error for missing weights and then predicted with
    randomly initialised ones. Exit code 2 here is the opposite of that.
    """
    for cmd in ("data", "eval", "train", "predict", "serve"):
        result = runner.invoke(app, [cmd])
        assert result.exit_code == 2, f"sphere {cmd} should exit 2 until implemented"

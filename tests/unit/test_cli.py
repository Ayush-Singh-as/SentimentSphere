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


def test_commands_that_cannot_work_fail_loudly(tmp_path, monkeypatch):
    """A command that cannot do its job must fail loudly.

    v1's speech app printed an error for missing weights and then predicted with
    randomly initialised ones. A nonzero exit here is the opposite of that.
    """
    monkeypatch.chdir(tmp_path)  # no data, no artifacts, no configs
    for argv in (
        ["data", "--dataset", "tess"],
        ["eval", "--baseline"],
        ["train", "--config", "absent.yaml"],
        ["predict", "--text", "hello"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code != 0, f"sphere {' '.join(argv)} should not report success"


def test_required_options_are_enforced():
    """Missing --config / --text is a usage error, not a silent no-op."""
    for cmd in ("train", "predict"):
        assert runner.invoke(app, [cmd]).exit_code == 2


def test_eval_without_baseline_explains_itself():
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "--baseline" in result.output


def test_unknown_dataset_exits_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["data", "--dataset", "not_a_dataset"])
    assert result.exit_code == 1
    assert "unavailable" in result.output

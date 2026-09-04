"""``sphere`` — the project's single entry point.

Commands that depend on later phases raise a clear "not yet" rather than
silently doing nothing. That is deliberate: v1's speech app printed an error
when its weights were missing and then went on to predict with randomly
initialised weights, which is the worst of both worlds.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sentimentsphere import __version__
from sentimentsphere.core import labels, provenance

app = typer.Typer(
    name="sphere",
    help="Multimodal emotion recognition: text, speech, face, and calibrated late fusion.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _pending(command: str, phase: str) -> None:
    console.print(f"[yellow]![/] [bold]sphere {command}[/] lands in {phase}.")
    console.print("  Roadmap and acceptance criteria: [cyan]UPGRADE_PLAN.md[/]")
    raise typer.Exit(code=2)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command("labels")
def labels_cmd(
    source: Annotated[
        str | None,
        typer.Option(help="Show one dataset's raw->canonical mapping, e.g. 'tess'."),
    ] = None,
) -> None:
    """Show the canonical label space, or one dataset's mapping onto it."""
    if source is None:
        table = Table(title="Canonical label space (alphabetical — order is load-bearing)")
        table.add_column("index", justify="right", style="cyan")
        table.add_column("label", style="bold")
        for i, label in enumerate(labels.CANONICAL):
            table.add_row(str(i), label)
        console.print(table)
        console.print(
            f"\n[dim]{labels.NUM_CLASSES} classes. Sources: {', '.join(sorted(labels.SOURCE_MAPS))}[/]"
        )
        for raw, reason in labels.EXCLUSION_REASONS.items():
            console.print(f"\n[yellow]excluded[/] [bold]{raw}[/]: {reason}")
        return

    if source not in labels.SOURCE_MAPS:
        console.print(
            f"[red]unknown source[/] {source!r}; expected one of {sorted(labels.SOURCE_MAPS)}"
        )
        raise typer.Exit(code=1)

    table = Table(title=f"{source}: raw -> canonical")
    table.add_column("raw", style="dim")
    table.add_column("canonical", style="bold")
    table.add_column("index", justify="right", style="cyan")
    for raw, canon in sorted(labels.SOURCE_MAPS[source].items()):
        if canon is None:
            table.add_row(raw, "[yellow]excluded[/]", "-")
        else:
            table.add_row(raw, str(canon), str(labels.to_index(canon)))
    console.print(table)


@app.command()
def info(
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Print environment provenance — what a run here would be stamped with."""
    prov = provenance.collect()
    if as_json:
        console.print_json(json.dumps(prov.to_dict()))
        return

    table = Table(title="Run provenance")
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("version", __version__)
    table.add_row("git sha", prov.git_sha or "[dim]not a git checkout[/]")
    dirty = prov.git_dirty
    table.add_row(
        "git tree",
        "[red]dirty — published metrics must come from a clean tree[/]"
        if dirty
        else ("clean" if dirty is False else "[dim]unknown[/]"),
    )
    table.add_row("python", prov.python_version)
    table.add_row("platform", prov.platform)
    for name, ver in sorted(prov.package_versions.items()):
        table.add_row(f"  {name}", ver)
    console.print(table)


@app.command("data")
def data_cmd() -> None:
    """Download and verify datasets."""
    _pending("data", "Phase 1 (evaluation harness)")


@app.command("eval")
def eval_cmd() -> None:
    """Score a model against a frozen split and write reports/."""
    _pending("eval", "Phase 1 (evaluation harness)")


@app.command("train")
def train_cmd() -> None:
    """Train a modality head from a config."""
    _pending("train", "Phases 2-4 (text / speech / face)")


@app.command("predict")
def predict_cmd() -> None:
    """Run inference over text, audio, image, or video."""
    _pending("predict", "Phase 6 (serving)")


@app.command()
def serve() -> None:
    """Start the FastAPI inference server."""
    _pending("serve", "Phase 6 (serving)")


if __name__ == "__main__":
    app()

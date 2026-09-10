"""``sphere`` — the project's single entry point.

A command that cannot do its job exits nonzero and says why, rather than
producing something that looks like an answer. That is deliberate: v1's speech
app printed an error when its weights were missing and then went on to predict
with randomly initialised ones, which is the worst of both worlds.

Heavy imports live inside the handlers so ``sphere --help`` and ``sphere labels``
stay fast on a core-only install.
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
def data_cmd(
    dataset: Annotated[
        str, typer.Option(help="text_aggregate, tess, ravdess, crema_d, savee, speech_combined.")
    ] = "text_aggregate",
    seed: Annotated[int, typer.Option(help="Seed for the frozen split.")] = 1337,
) -> None:
    """Inventory a local dataset and freeze its split manifest."""
    from sentimentsphere.core.config import Settings
    from sentimentsphere.data.catalog import prepare_dataset

    try:
        summary = prepare_dataset(dataset, Settings(), seed=seed)
    except (OSError, ValueError, KeyError) as error:
        console.print(f"[red]dataset unavailable[/] {dataset}: {error}")
        raise typer.Exit(code=1) from error

    console.print(f"[green]ok[/] {dataset}  fingerprint [cyan]{summary['fingerprint'][:16]}[/]")
    for path in summary["manifests"]:
        console.print(f"  manifest {path}")
    console.print_json(json.dumps(summary["audit"], default=str))


@app.command("eval")
def eval_cmd(
    baseline: Annotated[
        bool, typer.Option("--baseline", help="Audit and rescore the archived v1 artifacts.")
    ] = False,
) -> None:
    """Score models against frozen splits and write reports/."""
    from sentimentsphere.core.config import Settings
    from sentimentsphere.eval.legacy import audit_legacy

    if not baseline:
        console.print("[yellow]![/] pass [bold]--baseline[/] to audit the v1 artifacts.")
        console.print("  Trained-model evaluation is written by [cyan]sphere train[/].")
        raise typer.Exit(code=2)

    results = audit_legacy(Settings())
    table = Table(title="v1 artifact audit (absent inputs are reported, never estimated)")
    table.add_column("artifact", style="cyan")
    table.add_column("status", style="bold")
    table.add_column("detail")
    for name, payload in results.items():
        detail = payload.get("reason", "")
        if payload["status"] == "rescored":
            detail = (
                f"raw acc {payload['raw_accuracy']:.4f} -> "
                f"cleaned acc {payload['cleaned_accuracy']:.4f}"
            )
        elif "artifact" in payload:
            detail = f"{payload['artifact']['bytes']} bytes, keras {payload['artifact']['keras_version']}"
        table.add_row(name, payload["status"], str(detail)[:80])
    console.print(table)
    if all(payload["status"] == "unavailable" for payload in results.values()):
        console.print("[red]no v1 artifact could be read[/]; install artifacts/v1 and the")
        console.print("  baseline+legacy extras, then rerun. Nothing was scored.")
        raise typer.Exit(code=1)


@app.command("train")
def train_cmd(
    config: Annotated[str, typer.Option("--config", help="Path to a training YAML.")],
) -> None:
    """Train a modality head from a config, then calibrate and register it."""
    from sentimentsphere.core.config import Settings, TrainConfig
    from sentimentsphere.data.catalog import load_dataset
    from sentimentsphere.training.baseline import train_text

    settings = Settings()
    try:
        run = TrainConfig.from_yaml(config)
        dataset = load_dataset(run.dataset, settings)
        if run.modality != "text" or run.architecture != "tfidf_lr":
            raise ValueError(f"No implemented trainer for {run.modality}/{run.architecture}")
        report = train_text(run, dataset, settings)
    except (OSError, ValueError, KeyError) as error:
        console.print(f"[red]training failed[/]: {error}")
        raise typer.Exit(code=1) from error

    metrics = report["metrics"]
    table = Table(title=f"{report['run_id']} — frozen test split")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    for key in ("samples", "accuracy", "macro_f1", "weighted_f1", "ece"):
        value = metrics[key]
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    table.add_row("temperature", f"{report['temperature']:.4f}")
    console.print(table)
    console.print(f"[dim]reports/{report['run_id']}/[/]")


@app.command("predict")
def predict_cmd(
    text: Annotated[str, typer.Option("--text", help="Text to classify.")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Run inference with the registered models."""
    from sentimentsphere.inference.artifacts import ModelUnavailableError
    from sentimentsphere.inference.predictors import PredictorRegistry

    try:
        prediction = PredictorRegistry().get("text").predict(text)
    except (ModelUnavailableError, ValueError) as error:
        console.print(f"[red]prediction unavailable[/]: {error}")
        raise typer.Exit(code=1) from error

    if as_json:
        console.print_json(prediction.model_dump_json())
        return
    table = Table(title=f"{prediction.label} ({prediction.elapsed_ms:.0f} ms)")
    table.add_column("label", style="cyan")
    table.add_column("probability", justify="right")
    for label, score in sorted(prediction.scores.items(), key=lambda kv: -kv[1]):
        table.add_row(label, f"{score:.4f}")
    console.print(table)
    console.print(f"[dim]{prediction.model_id} · calibrated={prediction.calibrated}[/]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Start the FastAPI inference server."""
    try:
        import uvicorn
    except ImportError as error:
        console.print("[red]serving extra not installed[/]: uv sync --extra serving")
        raise typer.Exit(code=1) from error

    uvicorn.run("sentimentsphere.serving.api:app", host=host, port=port)


if __name__ == "__main__":
    app()

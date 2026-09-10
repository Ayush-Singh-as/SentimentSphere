"""The public demo.

Client-side capture only. v1 read the *server's* microphone and camera, which
cannot work for a remote visitor; Gradio's browser sources are what make the
hosted demo possible at all.

Tabs for heads that do not exist yet say so, rather than being hidden. A demo
that quietly omits two thirds of its stated scope is the same dishonesty as one
that fakes it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr

from sentimentsphere import __version__
from sentimentsphere.core.config import Settings
from sentimentsphere.core.io import read_json
from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.inference.artifacts import ModelUnavailableError
from sentimentsphere.inference.predictors import PredictorRegistry

SETTINGS = Settings()
REGISTRY = PredictorRegistry(SETTINGS)

EXAMPLES = [
    "I cannot believe we actually pulled this off, this is amazing",
    "I have been dreading this all week and I still am",
    "the report is due on friday and the meeting is at three",
    "I am still furious about how that was handled",
    "everything reminds me of them and I cannot stop crying",
]

PENDING = (
    "### Not implemented\n\n"
    "No {name} head has been trained yet, so this tab has nothing honest to show. "
    "See `UPGRADE_PLAN.md` phases 3-5. The alternative — shipping a tab backed by "
    "untrained weights — is exactly the v1 failure this rebuild exists to correct."
)


def classify(text: str) -> tuple[dict[str, float], str]:
    """Return label confidences plus the provenance line under them."""
    if not text or not text.strip():
        return {}, "Enter some text."
    try:
        prediction = REGISTRY.get("text").predict(text)
    except ModelUnavailableError:
        return {}, "**No text model is installed.** Train one with `make train-text`."
    except ValueError as error:
        return {}, f"**{error}**"
    return prediction.scores, (
        f"**{prediction.label}** · {prediction.elapsed_ms:.0f} ms · "
        f"`{prediction.model_id}` · calibrated={prediction.calibrated}"
    )


def metrics_markdown() -> str:
    """Render whatever reports exist, so the demo shows its own error analysis."""
    reports = sorted(Path(SETTINGS.reports_dir).glob("*/metrics.json"))
    if not reports:
        return "No reports generated yet. Run `make reproduce-text`."
    lines = [
        "| run | samples | accuracy | macro F1 | weighted F1 | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for path in reports:
        try:
            payload: dict[str, Any] = read_json(path)
            metrics = payload.get("metrics", {})
            lines.append(
                f"| `{path.parent.name}` | {metrics['samples']} | {metrics['accuracy']:.4f} "
                f"| {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} | {metrics['ece']:.4f} |"
            )
        except (OSError, KeyError, ValueError):
            continue
    lines.append(
        "\nMacro F1 is averaged over all seven declared classes, including any with "
        "zero support, so a partial head cannot look complete."
    )
    return "\n".join(lines)


ABOUT = f"""
## What this is

Emotion recognition over a single canonical seven-label space
(`{", ".join(CANONICAL)}`), rebuilt from a coursework project whose own audit
found a 100% accuracy that was a data leak, a deployed app running on randomly
initialised weights, and eight points of accuracy lost to a two-line
preprocessing mismatch.

## What is actually running here

Only the **text** head. Speech, face, and fusion are not trained yet and their
tabs say so. Every prediction shows the model id that produced it and whether
its confidence was calibrated.

## What this is not

A read on how someone feels. The output is a **predicted expression category**
for a piece of text, from a corpus with known label noise — the training data
contains identical sentences carrying conflicting labels, which puts a hard
ceiling on any achievable accuracy. Facial and vocal expression do not map
reliably onto internal emotional states (Barrett et al., 2019).

Version `{__version__}`.
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="SentimentSphere") as demo:
        gr.Markdown("# SentimentSphere\nCalibrated multimodal emotion recognition.")
        with gr.Tab("Text"):
            with gr.Row():
                with gr.Column():
                    box = gr.Textbox(
                        label="Text", lines=4, placeholder="Type or pick an example below"
                    )
                    button = gr.Button("Analyse", variant="primary")
                    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=box)
                with gr.Column():
                    chart = gr.Label(label="Confidence", num_top_classes=7)
                    detail = gr.Markdown()
            button.click(classify, inputs=box, outputs=[chart, detail])
            box.submit(classify, inputs=box, outputs=[chart, detail])
        for name in ("Voice", "Face", "Video (fusion)"):
            with gr.Tab(name):
                gr.Markdown(PENDING.format(name=name.split(" ")[0].lower()))
        with gr.Tab("Metrics"):
            gr.Markdown("## Measured performance\n")
            gr.Markdown(metrics_markdown())
        with gr.Tab("How it works"):
            gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )

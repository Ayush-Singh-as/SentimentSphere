"""Portable reports: machine-readable metrics plus standalone SVG figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sentimentsphere.core.io import write_json


def write_report(directory: Path, report: dict[str, Any], *, figures: bool = True) -> None:
    write_json(directory / "metrics.json", report)
    if figures:
        render_figures(directory, report.get("metrics", report))


def render_figures(directory: Path, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Reference",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.colorbar(plot, ax=axis)
    fig.savefig(directory / "confusion.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(labels, [metrics["per_class"][label]["f1"] for label in labels])
    axes[0].set(ylim=(0, 1), title="Per-class F1", ylabel="F1")
    plt.setp(axes[0].get_xticklabels(), rotation=40, ha="right")
    bins = [b for b in metrics["reliability_bins"] if b["count"]]
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
    axes[1].set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean confidence",
        ylabel="Accuracy",
        title=f"Reliability (ECE={metrics['ece']:.3f})",
    )
    fig.savefig(directory / "calibration.svg")
    plt.close(fig)

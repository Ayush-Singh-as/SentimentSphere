"""Canonical classification metrics including unsupported-class disclosure."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from sentimentsphere.core.labels import CANONICAL
from sentimentsphere.core.types import FloatArray, probabilities


def validate_targets(targets: Any, size: int, classes: int) -> Any:
    values = np.asarray(targets)
    if values.shape != (size,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Targets must be a one-dimensional integer array matching predictions")
    if (values < 0).any() or (values >= classes).any():
        raise ValueError("Target index outside label space")
    return values


def reliability_bins(targets: Any, scores: FloatArray, bins: int = 15) -> list[dict[str, Any]]:
    if bins < 1:
        raise ValueError("bins must be positive")
    values = probabilities(scores, classes=scores.shape[1])
    truth = validate_targets(targets, len(values), values.shape[1])
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1) == truth
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    result = []
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        result.append(
            {
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": count,
                "accuracy": float(correct[selected].mean()) if count else None,
                "confidence": float(confidence[selected].mean()) if count else None,
            }
        )
    return result


def classification_metrics(
    targets: Any,
    scores: Any,
    *,
    labels: tuple[str, ...] = CANONICAL,
) -> dict[str, Any]:
    values = probabilities(scores, classes=len(labels))
    truth = validate_targets(targets, len(values), len(labels))
    predicted = values.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    bins = reliability_bins(truth, values)
    ece = sum(b["count"] * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]) / len(
        values
    )
    supported = support > 0
    return {
        "samples": len(values),
        "labels": list(labels),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(np.mean(f1)),
        "supported_macro_f1": float(np.mean(f1[supported])),
        "weighted_f1": float(np.average(f1, weights=support)),
        "ece": float(ece),
        "negative_log_likelihood": float(log_loss(truth, values, labels=list(range(len(labels))))),
        "brier_score": float(np.mean(np.sum((values - np.eye(len(labels))[truth]) ** 2, axis=1))),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "unsupported_classes": [
            label for label, count in zip(labels, support, strict=True) if not count
        ],
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=list(range(len(labels)))
        ).tolist(),
        "reliability_bins": bins,
        "macro_f1_definition": "unweighted mean over the full declared label space, including zero-support classes",
    }

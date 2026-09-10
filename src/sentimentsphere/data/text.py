"""Text aggregate ingestion: explicit exclusions, conflict audit, duplicate grouping."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sentimentsphere.core.io import sha256_file
from sentimentsphere.core.labels import normalize, to_index
from sentimentsphere.data.records import Dataset, Sample


def clean_text(text: str) -> str:
    """Preserve negation/punctuation; normalize only Unicode and whitespace."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def text_identity(text: str) -> str:
    return hashlib.sha256(clean_text(text).casefold().encode()).hexdigest()


def load_text(path: Path) -> Dataset:
    """Deduplicate normalized text and exclude conflicting groups before splitting.

    Labels are used only for a fixed, documented dataset-construction rule. No
    model-dependent filtering or test-informed vocabulary fitting takes place.
    """
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    raw_pairs: Counter[tuple[str, str]] = Counter()
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Emotion", "Text"]:
            raise ValueError("Text CSV must have exactly Emotion,Text columns")
        for row in reader:
            raw, text = row["Emotion"], row["Text"]
            if raw is None or text is None:
                raise ValueError("CSV row has missing fields")
            counts["raw_rows"] += 1
            raw_pairs[(raw, text)] += 1
            label = normalize(raw, "text_aggregate")
            if label is None:
                counts["excluded_shame"] += 1
                continue
            cleaned = clean_text(text)
            if not cleaned:
                counts["empty_text"] += 1
                continue
            groups[text_identity(cleaned)].append((cleaned, to_index(label)))
    samples = []
    for identity, rows in sorted(groups.items()):
        if len({label for _, label in rows}) > 1:
            counts["conflicting_groups"] += 1
            counts["conflicting_rows"] += len(rows)
            continue
        text, index = sorted(rows)[0]
        counts["collapsed_duplicate_rows"] += len(rows) - 1
        samples.append(
            Sample(
                id=identity[:24],
                label=index,
                group=identity,
                content_hash=identity,
                text=text,
                corpus="text_aggregate",
            )
        )
    if not samples:
        raise ValueError("No usable text samples")
    counts["exact_duplicate_rows"] = sum(n - 1 for n in raw_pairs.values())
    counts["retained_rows"] = len(samples)
    return Dataset(
        "text_aggregate",
        tuple(samples),
        sha256_file(path),
        {
            "counts": dict(counts),
            "policy": "nfkc-whitespace-casefold-groups; drop-conflicts; one-row-per-group; exclude-shame-v1",
            "support": dict(Counter(str(s.label) for s in samples)),
        },
    )

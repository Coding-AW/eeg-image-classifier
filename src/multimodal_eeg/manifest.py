"""Immutable stimulus-manifest records and alignment validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MANIFEST_FIELDS = (
    "split",
    "eeg_row_index",
    "class_id",
    "stimulus_id",
    "image_path",
    "class_name",
)


@dataclass(frozen=True)
class StimulusRow:
    split: str
    eeg_row_index: int
    class_id: int
    stimulus_id: str
    image_path: str
    class_name: str


def read_manifest(path: str | Path) -> list[StimulusRow]:
    """Read declared alignment without inferring it from file ordering."""

    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError(f"Manifest columns must be exactly: {', '.join(MANIFEST_FIELDS)}")
        rows = []
        for raw in reader:
            try:
                rows.append(
                    StimulusRow(
                        raw["split"].strip(),
                        int(raw["eeg_row_index"]),
                        int(raw["class_id"]),
                        raw["stimulus_id"].strip(),
                        raw["image_path"].strip(),
                        raw["class_name"].strip(),
                    )
                )
            except (TypeError, ValueError) as error:
                raise ValueError("Manifest indices and class IDs must be integers.") from error
    if not rows:
        raise ValueError("Manifest cannot be empty.")
    return rows


def validate_manifest(
    rows: list[StimulusRow],
    images_root: str | Path,
    expected_labels: dict[str, NDArray[np.integer]] | None = None,
) -> dict[str, object]:
    """Validate paths, row coverage, labels, and class-name consistency."""

    root = Path(images_root).resolve()
    report: dict[str, object] = {"splits": {}}
    class_names: dict[tuple[str, int], str] = {}
    for row in rows:
        if row.split not in {"seen", "unseen"}:
            raise ValueError("Manifest split must be seen or unseen.")
        if row.eeg_row_index < 0 or row.class_id < 0 or not row.stimulus_id or not row.class_name:
            raise ValueError(
                "Manifest identifiers, names, and indices must be non-empty/non-negative."
            )
        resolved = (root / row.image_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"Image path escapes images root: {row.image_path}") from error
        if not resolved.is_file():
            raise FileNotFoundError(f"Stimulus image is missing: {resolved}")
        previous = class_names.setdefault((row.split, row.class_id), row.class_name)
        if previous != row.class_name:
            raise ValueError(f"Class {row.class_id} has inconsistent names.")
    for split in ("seen", "unseen"):
        selected = [row for row in rows if row.split == split]
        if not selected:
            continue
        indices = np.asarray([row.eeg_row_index for row in selected])
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Duplicate EEG row indices in {split} manifest.")
        if not np.array_equal(np.sort(indices), np.arange(len(indices))):
            raise ValueError(f"{split} manifest must cover every EEG row exactly once.")
        ordered = sorted(selected, key=lambda row: row.eeg_row_index)
        labels = np.asarray([row.class_id for row in ordered], dtype=np.int64)
        if expected_labels is not None:
            if split not in expected_labels:
                raise ValueError(f"Expected labels are missing split {split}.")
            np.testing.assert_array_equal(labels, np.asarray(expected_labels[split]).reshape(-1))
        report["splits"][split] = {
            "rows": len(selected),
            "classes": len(np.unique(labels)),
            "unique_stimuli": len({row.stimulus_id for row in selected}),
        }
    return report

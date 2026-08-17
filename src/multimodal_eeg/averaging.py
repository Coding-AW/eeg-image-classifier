"""Leakage-safe utilities shared by the NICE-style averaged-EEG study."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .config import ExperimentConfig
from .data import normalize_labels

REPETITION_COUNTS = (1, 2, 5, 10, 20, 40, 80)
CONTROL_SEEDS = (101, 211, 307)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_ranks(
    scores: np.ndarray, truth: np.ndarray, candidates: np.ndarray
) -> dict[str, float]:
    """Recalculate ranks without calling the production metric implementation."""

    candidate_index = {int(label): index for index, label in enumerate(candidates)}
    ranks = []
    for row, label in zip(scores, truth, strict=True):
        target = candidate_index[int(label)]
        ranks.append(1 + int(np.sum(row > row[target])))
    values = np.asarray(ranks)
    return {
        "top1_accuracy": float(np.mean(values <= 1)),
        "top5_accuracy": float(np.mean(values <= 5)),
        "mean_reciprocal_rank": float(np.mean(1 / values)),
        "median_rank": float(np.median(values)),
    }


def load_repeated_eeg(path: Path, config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    values = loadmat(path)
    raw = np.asarray(values["data"], dtype=np.float64)
    labels = normalize_labels(np.asarray(values["class_idx"]))
    if raw.shape != (16_000, 17, 100):
        raise ValueError(f"Unexpected repeated EEG shape: {raw.shape}")
    classes, counts = np.unique(labels, return_counts=True)
    if not np.array_equal(classes, np.arange(200)) or not np.all(counts == 80):
        raise ValueError("Repeated EEG must contain 80 rows for each of 200 classes.")
    return raw[:, :, config.eeg_start_index : config.eeg_stop_index], labels


def average_groups(
    raw: np.ndarray,
    group_codes: np.ndarray,
    trial_ids: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average subsets chosen from immutable trial IDs, never physical row position."""

    if len(raw) != len(group_codes) or len(raw) != len(trial_ids):
        raise ValueError("EEG rows, group codes, and immutable trial IDs must align.")
    if len(np.unique(trial_ids)) != len(trial_ids):
        raise ValueError("Immutable trial IDs must be unique.")
    classes, counts = np.unique(group_codes, return_counts=True)
    if count < 1 or np.any(counts < count):
        raise ValueError("Requested repetitions exceed an immutable stimulus group.")
    rows = []
    for group_code in classes:
        mask = group_codes == group_code
        group = raw[mask]
        identities = trial_ids[mask]
        priorities = np.asarray(
            [
                hashlib.sha256(f"{seed}:{identity}".encode()).digest()
                for identity in identities
            ],
            dtype="S32",
        )
        selected = np.argsort(priorities, kind="stable")[:count]
        rows.append(group[selected].mean(axis=0))
    flattened = np.stack(rows).reshape(len(classes), -1)
    return flattened, classes

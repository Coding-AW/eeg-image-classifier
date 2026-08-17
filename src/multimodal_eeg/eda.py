"""Focused, deterministic diagnostics for the NICE-style study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .array_utils import normalize_rows
from .classical_study import prototype_bank
from .data import Modalities
from .decoder import DecoderBundle

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def array_hash(values: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def assert_semantics_identical(sources: Mapping[str, Modalities]) -> str:
    """Assert participant-independent labels and semantic rows, returning a stable hash."""

    if not sources:
        raise ValueError("At least one participant source is required.")
    iterator = iter(sorted(sources.items()))
    first_name, first = next(iterator)
    expected = (array_hash(first.labels), array_hash(first.image))
    for name, data in iterator:
        observed = (array_hash(data.labels), array_hash(data.image))
        if observed != expected:
            raise ValueError(f"Semantic rows differ between {first_name} and {name}.")
    return expected[1]


def eeg_qc(participant: str, split: str, data: Modalities) -> dict[str, object]:
    values = np.asarray(data.eeg, dtype=np.float64)
    finite = np.isfinite(values)
    variances = np.var(values, axis=0)
    norms = np.linalg.norm(values, axis=1)
    counts = np.unique(data.labels, return_counts=True)[1]
    return {
        "participant": participant,
        "split": split,
        "samples": len(values),
        "classes": len(counts),
        "trials_per_class_min": int(counts.min()),
        "trials_per_class_max": int(counts.max()),
        "features": values.shape[1],
        "nonfinite_fraction": float(1 - finite.mean()),
        "constant_feature_fraction": float(np.mean(variances < 1e-12)),
        "feature_mean": float(values.mean()),
        "feature_sd": float(values.std(ddof=1)),
        "total_variance": float(variances.sum()),
        "sample_norm_mean": float(norms.mean()),
        "sample_norm_sd": float(norms.std(ddof=1)),
    }


def semantic_geometry(
    source: str, split: str, features: FloatArray, labels: IntArray
) -> dict[str, object]:
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    classes, prototypes = prototype_bank(values, labels)
    centered = prototypes - prototypes.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    probabilities = variance / max(variance.sum(), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-15))))
    similarities = prototypes @ prototypes.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = similarities.max(axis=1)
    nearest_indices = similarities.argmax(axis=1)
    hub_counts = np.bincount(nearest_indices, minlength=len(classes))
    normalized = normalize_rows(values)
    class_index = np.searchsorted(classes, labels)
    correct = np.sum(normalized * prototypes[class_index], axis=1)
    other = similarities[class_index].max(axis=1)
    margins = correct - other
    off_diagonal = similarities[np.isfinite(similarities)]
    return {
        "semantic_source": source,
        "split": split,
        "samples": len(values),
        "classes": len(classes),
        "dimensions": values.shape[1],
        "semantic_sha256": array_hash(values),
        "effective_rank": effective_rank,
        "common_direction_energy": float(np.sum(prototypes.mean(axis=0) ** 2)),
        "between_class_cosine": float(off_diagonal.mean()),
        "nearest_neighbor_cosine": float(nearest.mean()),
        "prototype_margin_mean": float(margins.mean()),
        "prototype_margin_sd": float(margins.std(ddof=1)),
        "prototype_hubness_skew": float(
            np.mean(((hub_counts - hub_counts.mean()) / max(hub_counts.std(), 1e-12)) ** 3)
        ),
    }


def rank_diagnostics(
    participant: str,
    source: str,
    bundle: DecoderBundle,
    eeg: FloatArray,
    truth: IntArray,
) -> tuple[dict[str, object], NDArray[np.int64]]:
    labels, scores = bundle.predict_scores(eeg)
    order = np.argsort(-scores, axis=1)
    ranked = labels[order]
    matches = ranked == np.asarray(truth)[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError("Every truth label must occur in the bundle candidate bank.")
    ranks = np.argmax(matches, axis=1).astype(np.int64) + 1
    quantiles = np.quantile(ranks, [0.1, 0.25, 0.5, 0.75, 0.9])
    return (
        {
            "participant": participant,
            "semantic_source": source,
            "samples": len(ranks),
            "top1_accuracy": float(np.mean(ranks <= 1)),
            "top5_accuracy": float(np.mean(ranks <= min(5, len(labels)))),
            "mean_reciprocal_rank": float(np.mean(1 / ranks)),
            "rank_mean": float(ranks.mean()),
            "rank_p10": float(quantiles[0]),
            "rank_p25": float(quantiles[1]),
            "rank_median": float(quantiles[2]),
            "rank_p75": float(quantiles[3]),
            "rank_p90": float(quantiles[4]),
        },
        ranks,
    )


def validate_rank_metrics(
    diagnostic: Mapping[str, object], published: Mapping[str, object], tolerance: float = 1e-12
) -> None:
    for field in ("top1_accuracy", "top5_accuracy", "mean_reciprocal_rank"):
        if not np.isclose(
            float(diagnostic[field]), float(published[field]), atol=tolerance, rtol=0
        ):
            raise ValueError(f"Reconstructed {field} does not match the published metric.")

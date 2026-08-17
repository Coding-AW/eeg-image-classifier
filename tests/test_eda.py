from __future__ import annotations

import numpy as np
import pytest

from multimodal_eeg.data import Modalities
from multimodal_eeg.eda import (
    assert_semantics_identical,
    eeg_qc,
    rank_diagnostics,
    semantic_geometry,
    validate_rank_metrics,
)


def _modalities(image: np.ndarray | None = None) -> Modalities:
    return Modalities(
        eeg=np.arange(24, dtype=float).reshape(6, 4),
        image=np.eye(3, dtype=float).repeat(2, axis=0) if image is None else image,
        text=None,
        labels=np.repeat(np.arange(3), 2),
    )


def test_eeg_qc_and_semantic_geometry_are_finite_and_deterministic() -> None:
    data = _modalities()
    qc = eeg_qc("sub-01", "seen", data)
    geometry = semantic_geometry("example", "seen", data.image, data.labels)
    assert qc["samples"] == 6
    assert qc["classes"] == 3
    assert qc["nonfinite_fraction"] == 0
    assert geometry == semantic_geometry("example", "seen", data.image, data.labels)
    numeric = [value for value in geometry.values() if isinstance(value, (int, float))]
    assert np.isfinite(numeric).all()


def test_semantic_deduplication_rejects_participant_difference() -> None:
    first = _modalities()
    assert len(assert_semantics_identical({"sub-01": first, "sub-02": _modalities()})) == 64
    changed = first.image.copy()
    changed[0, 0] += 0.01
    with pytest.raises(ValueError, match="Semantic rows differ"):
        assert_semantics_identical({"sub-01": first, "sub-02": _modalities(changed)})


class _Scores:
    def predict_scores(self, eeg):
        return np.array([10, 20, 30]), np.array([[3, 2, 1], [1, 2, 3]], dtype=float)


def test_rank_reconstruction_and_metric_validation() -> None:
    diagnostic, ranks = rank_diagnostics(
        "sub-01", "example", _Scores(), np.zeros((2, 4)), np.array([10, 20])
    )
    np.testing.assert_array_equal(ranks, [1, 2])
    expected = {
        "top1_accuracy": 0.5,
        "top5_accuracy": 1.0,
        "mean_reciprocal_rank": 0.75,
    }
    validate_rank_metrics(diagnostic, expected)
    with pytest.raises(ValueError, match="does not match"):
        validate_rank_metrics(diagnostic, {**expected, "top1_accuracy": 0.4})

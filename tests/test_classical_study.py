from dataclasses import replace

import numpy as np
import pytest

from multimodal_eeg.classical_study import (
    MatchedPreprocessor,
    PreprocessingSpec,
    calibrated_scores,
    class_disjoint_folds,
    embedding_eda,
    fit_and_score,
    holm_adjust,
    paired_change_statistics,
    run_baseline_anchored_participant,
)
from multimodal_eeg.config import ExperimentConfig
from multimodal_eeg.data import Modalities


def _modalities(classes: int = 6, trials: int = 4, seed: int = 4) -> Modalities:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), trials)
    semantic_prototypes = rng.normal(size=(classes, 8))
    semantics = semantic_prototypes[labels] + rng.normal(scale=0.05, size=(len(labels), 8))
    mapping = rng.normal(size=(8, 12))
    eeg = semantics @ mapping + rng.normal(scale=0.1, size=(len(labels), 12))
    return Modalities(eeg, semantics, None, labels)


def test_class_folds_are_disjoint_and_deterministic() -> None:
    labels = np.repeat(np.arange(12), 3)
    first = class_disjoint_folds(labels, 3, 17)
    second = class_disjoint_folds(labels[::-1], 3, 17)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert len(np.unique(np.concatenate(first))) == 12


def test_preprocessor_fits_only_passed_rows() -> None:
    data = _modalities()
    fitted = MatchedPreprocessor(PreprocessingSpec(8, 4, False, 1, True), 17).fit(
        data.eeg[:16], data.image[:16], data.labels[:16]
    )
    assert fitted.provenance_["fit_sample_count"] == 16
    before = fitted.semantic_mean_.copy()
    fitted.semantics(data.image[16:])
    np.testing.assert_array_equal(before, fitted.semantic_mean_)


def test_eda_reports_geometry_without_nonfinite_values() -> None:
    data = _modalities()
    report = embedding_eda(data.image, data.labels)
    assert report["dimensions"] == 8
    assert report["classes"] == 6
    assert report["trials_per_class_min"] == report["trials_per_class_max"] == 4
    assert report["total_variance"] > 0
    assert report["within_class_cosine"] > report["between_class_cosine"]
    assert report["nearest_prototype_accuracy"] > 0.9


def test_calibration_shapes_and_rejects_rank_preserving_standardization() -> None:
    scores = np.arange(40, dtype=float).reshape(8, 5)
    assert calibrated_scores(scores, "csls", 2).shape == scores.shape
    with pytest.raises(ValueError, match="cosine or csls"):
        calibrated_scores(scores, "standardized")


def test_paired_statistics_and_holm_are_deterministic() -> None:
    differences = np.linspace(0.01, 0.1, 10)
    first = paired_change_statistics(differences, bootstrap_samples=500)
    second = paired_change_statistics(differences, bootstrap_samples=500)
    assert first == second
    assert first["ci_low"] > 0
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert adjusted == [0.03, 0.08, 0.2]


def test_query_prediction_accepts_no_query_images() -> None:
    reference = _modalities(seed=1)
    external = _modalities(seed=2)
    first, provenance, predictions = fit_and_score(
        reference, external, PreprocessingSpec(), 1.0, 17
    )
    assert provenance["query_interface"] == ["eeg", "candidate_prototype_bank"]
    permuted = external.image.copy()
    for label in np.unique(external.labels):
        rows = np.flatnonzero(external.labels == label)
        permuted[rows] = permuted[rows[::-1]]
    second, _, repeated = fit_and_score(
        reference,
        Modalities(external.eeg, permuted, None, external.labels),
        PreprocessingSpec(),
        1.0,
        17,
    )
    assert first == second
    np.testing.assert_array_equal(predictions, repeated)


def test_baseline_is_preserved_in_every_comparison() -> None:
    reference = _modalities(seed=8)
    external = _modalities(seed=9)
    alternative_reference = Modalities(
        reference.eeg, reference.image[:, ::-1], None, reference.labels
    )
    alternative_external = Modalities(
        external.eeg, external.image[:, ::-1], None, external.labels
    )
    config = replace(
        ExperimentConfig(),
        tuning_folds=2,
        classical_semantic_dimensions=(4,),
        classical_eeg_dimensions=(None,),
        classical_remove_components=(0,),
        classical_whitening=(False,),
        ridge_alpha_candidates=(1.0,),
        classical_csls_neighbors=(2,),
    )
    rows, report, predictions = run_baseline_anchored_participant(
        "sub-test",
        {"cornet-s": reference, "alternative": alternative_reference},
        {"cornet-s": external, "alternative": alternative_external},
        config,
    )
    assert report["baseline"]["provenance"]["query_interface"] == [
        "eeg",
        "candidate_prototype_bank",
    ]
    assert "S0:cornet-s" in predictions
    assert {row.baseline_id for row in rows} == {"S0-cornet-ridge"}
    assert all(row.change_from_baseline == 0 for row in rows if row.stage == "S0")
    repeated_rows, repeated_report, repeated_predictions = run_baseline_anchored_participant(
        "sub-test",
        {"cornet-s": reference, "alternative": alternative_reference},
        {"cornet-s": external, "alternative": alternative_external},
        config,
        full=False,
    )
    np.testing.assert_array_equal(
        predictions["S0:cornet-s"], repeated_predictions["S0:cornet-s"]
    )
    assert report["prediction_hashes"]["S0:cornet-s"] == repeated_report[
        "prediction_hashes"
    ]["S0:cornet-s"]
    assert repeated_rows[0].configuration_hash == rows[0].configuration_hash

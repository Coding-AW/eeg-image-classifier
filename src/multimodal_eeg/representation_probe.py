"""Reference-only selection for the frozen DINOv2 representation probe."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .classical_study import PreprocessingSpec, reference_scores
from .config import ExperimentConfig
from .data import Modalities
from .embedding_features import dinov2_variant_names


def select_dinov2_variant(
    references: Mapping[str, Mapping[str, Modalities]], config: ExperimentConfig
) -> dict[str, object]:
    """Lock one layer/pooling candidate without loading evaluation features."""

    expected = dinov2_variant_names()
    if tuple(references) != expected:
        raise ValueError(f"Expected ordered DINOv2 variants {expected}.")
    participants = tuple(config.subjects)
    fixed_spec = PreprocessingSpec()
    records = []
    for variant in expected:
        if tuple(references[variant]) != participants:
            raise ValueError(f"{variant} does not contain the complete participant set.")
        fold_scores = []
        participant_records = {}
        for participant in participants:
            scores = [
                float(value)
                for value in reference_scores(
                    references[variant][participant],
                    fixed_spec,
                    config.classical_baseline_alpha,
                    config.tuning_folds,
                    config.random_seed,
                )
            ]
            fold_scores.extend(scores)
            participant_records[participant] = {
                "fold_scores": scores,
                "mean": float(np.mean(scores)),
            }
        records.append(
            {
                "variant": variant,
                "fold_scores": fold_scores,
                "mean": float(np.mean(fold_scores)),
                "participants": participant_records,
            }
        )
    baseline = next(item for item in records if item["variant"] == "layer-12-cls")
    baseline_scores = np.asarray(baseline["fold_scores"])
    for item in records:
        item["beats_final_cls_every_fold"] = bool(
            item["variant"] == baseline["variant"]
            or np.all(np.asarray(item["fold_scores"]) > baseline_scores)
        )
    eligible = [item for item in records if item["beats_final_cls_every_fold"]]
    selected = max(eligible, key=lambda item: item["mean"])
    return {
        "selection_split": "reference-class folds only",
        "selection_protocol": "fixed raw semantics, baseline ridge, cosine retrieval",
        "fixed_spec": {
            "eeg_dimension": fixed_spec.eeg_dimension,
            "semantic_dimension": fixed_spec.semantic_dimension,
            "whiten": fixed_spec.whiten,
            "remove_components": fixed_spec.remove_components,
            "center_semantics": fixed_spec.center_semantics,
        },
        "fixed_alpha": config.classical_baseline_alpha,
        "evaluation_features_loaded": False,
        "baseline_variant": baseline["variant"],
        "selected_variant": selected["variant"],
        "retained_change": selected["variant"] != baseline["variant"],
        "records": records,
    }

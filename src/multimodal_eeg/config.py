"""Typed configuration for the focused EEG decoding study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Every setting that can alter the NICE-style study result."""

    subjects: tuple[str, ...] = tuple(f"sub-{index:02d}" for index in range(1, 11))
    eeg_start_index: int = 27
    eeg_stop_index: int = 60
    image_components: int = 100
    reference_class_limit: int | None = None
    evaluation_class_limit: int = 200
    tuning_class_limit: int = 200
    tuning_folds: int = 3
    random_seed: int = 17
    ridge_alpha_candidates: tuple[float, ...] = (10.0, 100.0, 1000.0)
    classical_baseline_alpha: float = 100.0
    classical_semantic_dimensions: tuple[int, ...] = (32, 64, 100)
    classical_eeg_dimensions: tuple[int | None, ...] = (None, 64, 128)
    classical_remove_components: tuple[int, ...] = (0, 1, 2)
    classical_whitening: tuple[bool, ...] = (False, True)
    classical_csls_neighbors: tuple[int, ...] = (5, 10, 20)

    def __post_init__(self) -> None:
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("subjects must be non-empty and unique.")
        if self.eeg_start_index < 0 or self.eeg_stop_index <= self.eeg_start_index:
            raise ValueError("The EEG interval must be a non-empty, non-negative slice.")
        if self.evaluation_class_limit < 2 or self.tuning_class_limit < 2:
            raise ValueError("Class limits must include at least two classes.")
        if self.tuning_folds < 2:
            raise ValueError("tuning_folds must be at least two.")
        if any(value <= 0 for value in self.ridge_alpha_candidates):
            raise ValueError("Ridge candidates must be positive.")
        if self.classical_baseline_alpha <= 0:
            raise ValueError("The baseline ridge penalty must be positive.")
        if any(value < 1 for value in self.classical_semantic_dimensions):
            raise ValueError("Semantic dimensions must be positive.")
        if any(value is not None and value < 1 for value in self.classical_eeg_dimensions):
            raise ValueError("EEG dimensions must be positive or None.")
        if any(value < 0 for value in self.classical_remove_components):
            raise ValueError("Removed component counts cannot be negative.")
        if any(value < 1 for value in self.classical_csls_neighbors):
            raise ValueError("CSLS neighbourhood sizes must be positive.")

    @property
    def subject(self) -> str:
        return self.subjects[0]

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        tuple_fields = {
            "subjects", "ridge_alpha_candidates", "classical_semantic_dimensions",
            "classical_eeg_dimensions", "classical_remove_components",
            "classical_whitening", "classical_csls_neighbors",
        }
        for field in tuple_fields & values.keys():
            values[field] = tuple(values[field])
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        for key, value in tuple(values.items()):
            if isinstance(value, tuple):
                values[key] = list(value)
        return values

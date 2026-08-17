"""Explicit loading and validation for the distributed feature matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from .config import ExperimentConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Modalities:
    """Aligned feature matrices and their one-dimensional class labels."""

    eeg: FloatArray
    image: FloatArray
    text: FloatArray | None
    labels: IntArray

    def __post_init__(self) -> None:
        arrays = [self.eeg, self.image]
        if self.text is not None:
            arrays.append(self.text)
        sample_counts = {array.shape[0] for array in arrays} | {self.labels.shape[0]}
        if len(sample_counts) != 1:
            raise ValueError("All modalities and labels must contain the same samples.")
        if self.labels.ndim != 1:
            raise ValueError("labels must be one-dimensional.")
        if any(array.ndim != 2 for array in arrays):
            raise ValueError("Feature arrays must be two-dimensional.")
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Feature arrays contain non-finite values.")

    def take(self, indices: NDArray[np.integer]) -> Modalities:
        """Select the same rows from every modality."""

        return Modalities(
            eeg=self.eeg[indices],
            image=self.image[indices],
            text=None if self.text is None else self.text[indices],
            labels=self.labels[indices],
        )

    def limit_classes(self, maximum: int | None) -> Modalities:
        """Keep the numerically first classes without relying on row ordering."""

        if maximum is None:
            return self
        selected = np.unique(self.labels)[:maximum]
        return self.take(np.flatnonzero(np.isin(self.labels, selected)))


def normalize_labels(labels: NDArray[np.integer]) -> IntArray:
    """Return flat integer labels and convert a contiguous one-based range to zero-based."""

    result = np.asarray(labels, dtype=np.int64).reshape(-1)
    if result.size == 0:
        raise ValueError("Labels cannot be empty.")
    unique = np.unique(result)
    if unique[0] == 1 and np.array_equal(unique, np.arange(1, unique[-1] + 1)):
        result = result - 1
    return result


def _required_matrix(path: Path, key: str) -> NDArray[np.generic]:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature file is missing: {path}")
    contents = loadmat(path)
    if key not in contents:
        raise KeyError(f"{path} does not contain the expected '{key}' matrix.")
    return np.asarray(contents[key])


def load_feature_split(
    dataset_root: str | Path,
    split: str,
    config: ExperimentConfig,
    subject: str | None = None,
) -> Modalities:
    """Load one feature split from the legacy THINGS-EEG2 feature archive.

    Parameters
    ----------
    dataset_root:
        Directory containing ``brain_feature``, ``visual_feature`` and
        ``textual_feature``.
    split:
        ``"seen"`` loads training categories; ``"unseen"`` loads the external
        category set.
    config:
        Dataset and preprocessing-independent experiment settings.
    """

    if split not in {"seen", "unseen"}:
        raise ValueError("split must be either 'seen' or 'unseen'.")

    root = Path(dataset_root)
    selected_subject = config.subject if subject is None else subject
    brain_dir = root / "brain_feature" / "17channels" / selected_subject
    if split == "seen":
        brain_file = brain_dir / "eeg_train_data_within.mat"
        visual_file = (
            root
            / "visual_feature"
            / "ThingsTrain"
            / "pytorch"
            / "cornet_s"
            / selected_subject
            / "feat_pca_train.mat"
        )
    else:
        brain_file = brain_dir / "eeg_test_data.mat"
        visual_file = (
            root
            / "visual_feature"
            / "ThingsTest"
            / "pytorch"
            / "cornet_s"
            / selected_subject
            / "feat_pca_test.mat"
        )

    raw_eeg = _required_matrix(brain_file, "data")
    if raw_eeg.ndim != 3 or config.eeg_stop_index > raw_eeg.shape[2]:
        raise ValueError("EEG data do not support the configured time slice.")
    eeg = raw_eeg[:, :, config.eeg_start_index : config.eeg_stop_index].reshape(
        raw_eeg.shape[0], -1
    )
    labels = normalize_labels(_required_matrix(brain_file, "class_idx"))
    image = _required_matrix(visual_file, "data")[:, : config.image_components]

    limit = config.reference_class_limit if split == "seen" else config.evaluation_class_limit
    limit = config.reference_class_limit if split == "seen" else config.evaluation_class_limit
    return Modalities(
        eeg=np.asarray(eeg, dtype=np.float64),
        image=np.asarray(image, dtype=np.float64),
        text=None,
        labels=labels,
    ).limit_classes(limit)



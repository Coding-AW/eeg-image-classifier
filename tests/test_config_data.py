import numpy as np
import pytest

from multimodal_eeg.config import ExperimentConfig
from multimodal_eeg.data import Modalities, normalize_labels


def test_normalize_contiguous_one_based_labels() -> None:
    labels = normalize_labels(np.array([[1, 2, 3, 1]]))
    np.testing.assert_array_equal(labels, [0, 1, 2, 0])


def test_noncontiguous_identifiers_are_not_rewritten() -> None:
    labels = normalize_labels(np.array([1, 3, 3]))
    np.testing.assert_array_equal(labels, [1, 3, 3])


def test_modalities_reject_misaligned_samples() -> None:
    with pytest.raises(ValueError, match="same samples"):
        Modalities(np.ones((4, 2)), np.ones((3, 2)), None, np.arange(4))


def test_class_limit_is_independent_of_row_order() -> None:
    labels = np.array([8, 2, 8, 5, 2, 5])
    values = np.arange(6)[:, None].astype(float)
    data = Modalities(values, values + 10, None, labels).limit_classes(2)
    assert set(data.labels) == {2, 5}
    np.testing.assert_array_equal(data.eeg[:, 0], [1, 3, 4, 5])


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(tuning_folds=1)

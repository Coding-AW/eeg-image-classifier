"""Small numerical helpers shared by extraction, fitting, and inference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def normalize_rows(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return finite unit-length rows without changing row order."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Row normalization requires a finite two-dimensional array.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)

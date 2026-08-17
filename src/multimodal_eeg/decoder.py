"""Reusable, participant-specific EEG-to-semantic decoder bundles."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import sklearn
from numpy.typing import NDArray
from sklearn.linear_model import Ridge

from .array_utils import normalize_rows
from .classical_study import (
    MatchedPreprocessor,
    PreprocessingSpec,
    calibrated_scores,
    prototype_bank,
)
from .data import Modalities

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BUNDLE_FORMAT = 1


@dataclass(frozen=True)
class CandidateBank:
    """Raw frozen semantic features and one label per feature row."""

    features: FloatArray
    labels: IntArray
    semantic_source: str | None = None
    feature_provenance: dict[str, object] | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        labels = np.asarray(self.labels)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError("Candidate features and labels must be aligned 2D/1D arrays.")
        if len(features) < 2 or not np.isfinite(features).all():
            raise ValueError("Candidate bank requires at least two finite feature rows.")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Candidate labels must be integers.")


@dataclass
class DecoderBundle:
    """A fitted decoder plus its default candidate bank and provenance."""

    participant: str
    semantic_source: str
    transform: MatchedPreprocessor
    decoder: Ridge
    candidate_labels: IntArray
    candidate_prototypes: FloatArray
    raw_semantic_width: int
    eeg_width: int
    retrieval: str
    neighbors: int
    seed: int
    configuration: dict[str, object]
    feature_provenance: dict[str, object]
    package_versions: dict[str, str]
    format_version: int = BUNDLE_FORMAT

    def _bank(self, prototypes: CandidateBank | None) -> tuple[IntArray, FloatArray]:
        if prototypes is None:
            return self.candidate_labels, self.candidate_prototypes
        if prototypes.semantic_source not in (None, self.semantic_source):
            raise ValueError("Candidate semantic source does not match the decoder bundle.")
        if (
            prototypes.feature_provenance is not None
            and prototypes.feature_provenance != self.feature_provenance
        ):
            raise ValueError("Candidate feature provenance does not match the decoder bundle.")
        if prototypes.features.shape[1] != self.raw_semantic_width:
            raise ValueError(
                f"Expected {self.raw_semantic_width} semantic features, "
                f"received {prototypes.features.shape[1]}."
            )
        labels, raw = prototype_bank(prototypes.features, prototypes.labels)
        return labels, self.transform.semantics(raw)

    def predict_scores(
        self, eeg: FloatArray, prototypes: CandidateBank | None = None
    ) -> tuple[IntArray, FloatArray]:
        values = np.asarray(eeg, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.eeg_width:
            raise ValueError(f"EEG input must have shape (samples, {self.eeg_width}).")
        if not np.isfinite(values).all():
            raise ValueError("EEG input contains non-finite values.")
        labels, bank = self._bank(prototypes)
        query = normalize_rows(self.decoder.predict(self.transform.eeg(values)))
        scores = calibrated_scores(query @ bank.T, self.retrieval, self.neighbors)
        return labels, scores

    def predict(self, eeg: FloatArray, prototypes: CandidateBank | None = None) -> IntArray:
        labels, scores = self.predict_scores(eeg, prototypes)
        return labels[np.argmax(scores, axis=1)]


def fit_decoder(
    participant: str,
    semantic_source: str,
    reference: Modalities,
    candidates: CandidateBank,
    spec: PreprocessingSpec,
    alpha: float,
    seed: int,
    *,
    retrieval: str = "cosine",
    neighbors: int = 10,
    configuration: dict[str, object] | None = None,
    feature_provenance: dict[str, object] | None = None,
) -> DecoderBundle:
    """Fit a reusable decoder using reference EEG/semantics only."""

    if candidates.features.shape[1] != reference.image.shape[1]:
        raise ValueError("Reference and candidate semantic dimensions do not match.")
    transform = MatchedPreprocessor(spec, seed).fit(
        reference.eeg, reference.image, reference.labels
    )
    decoder = Ridge(alpha=alpha).fit(
        transform.eeg(reference.eeg), transform.semantics(reference.image)
    )
    labels, raw_prototypes = prototype_bank(candidates.features, candidates.labels)
    return DecoderBundle(
        participant=participant,
        semantic_source=semantic_source,
        transform=transform,
        decoder=decoder,
        candidate_labels=labels,
        candidate_prototypes=transform.semantics(raw_prototypes),
        raw_semantic_width=reference.image.shape[1],
        eeg_width=reference.eeg.shape[1],
        retrieval=retrieval,
        neighbors=neighbors,
        seed=seed,
        configuration=dict(configuration or {}),
        feature_provenance=dict(feature_provenance or {}),
        package_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(bundle: DecoderBundle, path: str | Path) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    try:
        portable_path = path.resolve().relative_to(Path(__file__).resolve().parents[2]).as_posix()
    except ValueError:
        portable_path = path.name
    return {
        "participant": bundle.participant,
        "semantic_source": bundle.semantic_source,
        "path": portable_path,
        "sha256": _sha256(path),
        "format_version": bundle.format_version,
        "eeg_width": bundle.eeg_width,
        "semantic_width": bundle.raw_semantic_width,
        "candidate_classes": len(bundle.candidate_labels),
        "retrieval": bundle.retrieval,
        "neighbors": bundle.neighbors,
        "configuration": json.dumps(bundle.configuration, sort_keys=True),
    }


def load_bundle(
    path: str | Path, *, trusted: bool = False, expected_sha256: str | None = None
) -> DecoderBundle:
    """Load a trusted local bundle; joblib files must never come from untrusted sources."""

    if not trusted:
        raise ValueError("Refusing to load pickle-based bundle without trusted=True.")
    path = Path(path)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("Decoder bundle hash does not match the trusted manifest.")
    bundle = joblib.load(path)
    if not isinstance(bundle, DecoderBundle) or bundle.format_version != BUNDLE_FORMAT:
        raise ValueError("Unsupported or invalid decoder bundle.")
    if bundle.package_versions.get("scikit_learn") != sklearn.__version__:
        raise ValueError("Decoder bundle was created with a different scikit-learn version.")
    return bundle

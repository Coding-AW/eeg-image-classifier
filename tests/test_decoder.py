import numpy as np
import pytest

from multimodal_eeg.classical_study import PreprocessingSpec
from multimodal_eeg.data import Modalities
from multimodal_eeg.decoder import CandidateBank, fit_decoder, load_bundle, save_bundle


def _data(seed: int, classes: int = 6, trials: int = 4) -> Modalities:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(classes), trials)
    prototypes = rng.normal(size=(classes, 8))
    semantics = prototypes[labels] + rng.normal(scale=0.03, size=(len(labels), 8))
    eeg = semantics @ rng.normal(size=(8, 12))
    return Modalities(eeg, semantics, None, labels)


def test_bundle_round_trip_and_alternate_candidates(tmp_path) -> None:
    reference = _data(1)
    external = _data(2)
    bundle = fit_decoder(
        "sub-test",
        "dinov2",
        reference,
        CandidateBank(external.image, external.labels),
        PreprocessingSpec(8, 4),
        10.0,
        17,
    )
    expected = bundle.predict(external.eeg)
    path = tmp_path / "decoder.joblib"
    manifest = save_bundle(bundle, path)
    assert manifest["candidate_classes"] == 6
    with pytest.raises(ValueError, match="trusted=True"):
        load_bundle(path)
    loaded = load_bundle(path, trusted=True, expected_sha256=manifest["sha256"])
    np.testing.assert_array_equal(expected, loaded.predict(external.eeg))

    alternate = CandidateBank(external.image, external.labels + 100)
    predictions = loaded.predict(external.eeg, alternate)
    assert set(predictions) <= set(alternate.labels)
    with pytest.raises(ValueError, match="hash"):
        load_bundle(path, trusted=True, expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="semantic source"):
        loaded.predict(
            external.eeg,
            CandidateBank(external.image, external.labels, semantic_source="cornet-s"),
        )
    with pytest.raises(ValueError, match="provenance"):
        loaded.predict(
            external.eeg,
            CandidateBank(
                external.image,
                external.labels,
                semantic_source="dinov2",
                feature_provenance={"checkpoint": "wrong"},
            ),
        )


def test_bundle_rejects_incompatible_or_nonfinite_inputs() -> None:
    reference = _data(3)
    external = _data(4)
    bundle = fit_decoder(
        "sub-test",
        "cornet-s",
        reference,
        CandidateBank(external.image, external.labels),
        PreprocessingSpec(),
        100.0,
        17,
    )
    with pytest.raises(ValueError, match="shape"):
        bundle.predict(np.ones((2, 11)))
    bad = external.eeg.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bundle.predict(bad)
    with pytest.raises(ValueError, match="Expected 8 semantic"):
        bundle.predict(external.eeg, CandidateBank(np.ones((3, 7)), np.arange(3)))

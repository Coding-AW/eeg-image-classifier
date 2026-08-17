import numpy as np

from multimodal_eeg.config import ExperimentConfig
from multimodal_eeg.data import Modalities
from multimodal_eeg.embedding_features import dinov2_variant_names
from multimodal_eeg.representation_probe import select_dinov2_variant


def test_probe_enumeration_is_exact_and_deterministic(monkeypatch) -> None:
    participants = ("sub-01", "sub-02")
    config = ExperimentConfig(subjects=participants)
    labels = np.array([0, 0, 1, 1])
    references = {
        name: {
            participant: Modalities(
                eeg=np.ones((4, 2)),
                image=np.full((4, 3), 2.0 if name == "layer-03-cls" else 1.0),
                text=None,
                labels=labels,
            )
            for participant in participants
        }
        for name in dinov2_variant_names()
    }

    def fake_scores(reference, *_args):
        score = float(reference.image[0, 0]) / 10
        return [score, score, score]

    monkeypatch.setattr("multimodal_eeg.representation_probe.reference_scores", fake_scores)
    first = select_dinov2_variant(references, config)
    second = select_dinov2_variant(references, config)
    assert first == second
    assert [item["variant"] for item in first["records"]] == list(dinov2_variant_names())
    assert first["selected_variant"] == "layer-03-cls"
    assert first["evaluation_features_loaded"] is False
    assert first["selection_protocol"] == "fixed raw semantics, baseline ridge, cosine retrieval"

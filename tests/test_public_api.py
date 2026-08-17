import multimodal_eeg


def test_public_api_contains_only_supported_research_interfaces() -> None:
    assert set(multimodal_eeg.__all__) == {
        "ExperimentConfig",
        "publish_nice_study",
        "run_aligned_comparison",
    }

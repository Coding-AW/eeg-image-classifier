import csv
from pathlib import Path

import numpy as np

from multimodal_eeg.embedding_features import (
    dinov2_variant_names,
    extract_dinov2_probe,
    extract_visual_features,
    load_visual_artifact,
)
from multimodal_eeg.manifest import MANIFEST_FIELDS


class FakeVisualBackend:
    checkpoint = "test/frozen-vision"
    revision = "revision-1"
    pooling = "pooler_or_cls"
    package_versions = {"fake": "1"}

    def __init__(self) -> None:
        self.calls = 0

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        self.calls += 1
        return np.asarray(
            [np.random.default_rng(sum(path.read_bytes())).normal(size=6) for path in paths]
        )


class FakeProbeBackend(FakeVisualBackend):
    def encode_variants(self, paths: list[Path]) -> dict[str, np.ndarray]:
        base = self.encode_images(paths)
        return {name: base + index for index, name in enumerate(dinov2_variant_names())}


def _manifest(path: Path, images: Path) -> None:
    rows = []
    for split in ("seen", "unseen"):
        for index in range(6):
            image = images / f"{split}-{index}.bin"
            image.write_bytes(bytes([index + (10 if split == "unseen" else 0)]))
            rows.append(
                {
                    "split": split,
                    "eeg_row_index": index,
                    "class_id": index // 2,
                    "stimulus_id": f"{split}-{index}",
                    "image_path": image.name,
                    "class_name": f"class {index // 2}",
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_generic_extraction_alignment_hashes_and_resume(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, images)
    output = tmp_path / "features"
    backend = FakeVisualBackend()
    first = extract_visual_features(manifest, images, output, backend, batch_size=4)
    call_count = backend.calls
    second = extract_visual_features(manifest, images, output, backend, batch_size=4, resume=True)
    assert first == second
    assert backend.calls == call_count
    artifact = load_visual_artifact(output, "seen")
    assert artifact.features.shape == (6, 6)
    np.testing.assert_array_equal(artifact.row_indices, np.arange(6))
    np.testing.assert_allclose(np.linalg.norm(artifact.features, axis=1), 1, atol=1e-6)


def test_dinov2_probe_is_exact_aligned_and_resumable(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, images)
    output = tmp_path / "probe"
    backend = FakeProbeBackend()
    first = extract_dinov2_probe(manifest, images, output, backend, batch_size=4)
    calls = backend.calls
    second = extract_dinov2_probe(manifest, images, output, backend, batch_size=4, resume=True)
    assert first == second
    assert backend.calls == calls
    assert tuple(first["variants"]) == dinov2_variant_names()
    for name in dinov2_variant_names():
        artifact = load_visual_artifact(output / name, "seen")
        assert artifact.metadata["encoder_frozen"] is True
        assert artifact.features.shape == (6, 6)

import csv
from pathlib import Path

import numpy as np

from multimodal_eeg.manifest import MANIFEST_FIELDS
from multimodal_eeg.nice_templates import (
    THINGS_ARCHIVE_SHA256,
    _concept_centers,
    build_template_manifest,
)


def test_concept_centers_average_each_class() -> None:
    labels = np.repeat(np.arange(200), 2)
    values = np.column_stack((labels, np.ones_like(labels))).astype(float)
    centers, classes = _concept_centers(values, labels)
    assert centers.shape == (200, 2)
    assert np.array_equal(classes, np.arange(200))
    assert np.isfinite(centers).all()
    order = np.random.default_rng(17).permutation(len(labels))
    reordered, reordered_classes = _concept_centers(values[order], labels[order])
    assert np.array_equal(classes, reordered_classes)
    assert np.allclose(centers, reordered)


def test_authoritative_things_checksum_is_locked() -> None:
    assert THINGS_ARCHIVE_SHA256 == (
        "fb5a7cac28a27ff1ff8b723f072ed4a8309bc807a5f481b001da93f4213f3d24"
    )


def test_manifest_excludes_renamed_eeg_images_by_hash(tmp_path: Path) -> None:
    full = tmp_path / "full" / "object_images"
    eeg = tmp_path / "eeg"
    manifest_rows = []
    concepts = []
    for class_id in range(200):
        name = f"concept_{class_id}"
        concepts.append(f"{class_id:05d}_{name}")
        directory = full / name
        directory.mkdir(parents=True)
        eliciting = eeg / f"eeg_{class_id}.jpg"
        eliciting.parent.mkdir(parents=True, exist_ok=True)
        eliciting.write_bytes(f"eliciting-{class_id}".encode())
        (directory / f"renamed_{class_id}.jpg").write_bytes(eliciting.read_bytes())
        (directory / f"independent_{class_id}.jpg").write_bytes(
            f"independent-{class_id}".encode()
        )
        manifest_rows.append(
            {
                "split": "unseen",
                "eeg_row_index": class_id,
                "class_id": class_id,
                "stimulus_id": f"stim-{class_id}",
                "image_path": eliciting.name,
                "class_name": name.replace("_", " "),
            }
        )
    metadata = tmp_path / "metadata.npy"
    np.save(metadata, {"test_img_concepts_THINGS": concepts})
    manifest = tmp_path / "stimulus.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    output = tmp_path / "template.csv"
    report = build_template_manifest(full.parent, metadata, manifest, eeg, output)
    assert report["images"] == 200
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all("/independent_" in row["image_id"] for row in rows)

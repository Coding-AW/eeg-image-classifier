import csv
from pathlib import Path

import pytest

from multimodal_eeg.nice_publication import _portable_image_path, _publish_template_manifest


def test_portable_image_path_removes_local_prefix() -> None:
    value = "C:/workspace/data/extracted/object_images/antelope/antelope_01.jpg"
    assert _portable_image_path(value) == "object_images/antelope/antelope_01.jpg"


def test_portable_image_path_rejects_unrelated_path() -> None:
    with pytest.raises(ValueError, match="outside object_images"):
        _portable_image_path("C:/workspace/private/image.jpg")


def test_published_manifest_has_no_absolute_paths(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "portable.csv"
    row = {
        "class_id": 0,
        "image_path": "C:/data/object_images/antelope/antelope_01.jpg",
        "sha256": "abc",
    }
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    _publish_template_manifest(source, destination)
    with destination.open(newline="", encoding="utf-8") as handle:
        published = list(csv.DictReader(handle))
    assert published[0]["image_path"] == "object_images/antelope/antelope_01.jpg"

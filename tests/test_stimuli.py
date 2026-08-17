import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from multimodal_eeg.manifest import StimulusRow
from multimodal_eeg.stimuli import (
    PINNED_FILES,
    RemoteFile,
    _checksum,
    _concept_from_filename,
    _resolve_extracted_image,
    _safe_extract,
    parse_nemar_manifest,
    stimulus_listing,
    verify_official_metadata_order,
)


def _manifest_payload() -> bytes:
    rows = [
        {
            "path": path,
            "size": size,
            "checksum_algorithm": algorithm,
            "checksum": checksum,
            "bytes_url": f"https://data.nemar.org/{path}",
        }
        for path, (size, algorithm, checksum) in PINNED_FILES.items()
    ]
    return json.dumps(rows).encode()


def test_pinned_manifest_accepts_only_exact_official_records() -> None:
    records = parse_nemar_manifest(_manifest_payload())
    assert set(records) == set(PINNED_FILES)
    listing = stimulus_listing(records)
    assert listing["dataset_id"] == "nm000232"
    assert listing["version"] == "v1.1.0"
    changed = json.loads(_manifest_payload())
    changed[0]["size"] += 1
    with pytest.raises(RuntimeError, match="changed"):
        parse_nemar_manifest(json.dumps(changed))


def test_concept_metadata_comes_from_official_path() -> None:
    assert _concept_from_filename(
        "training_images/00014_aloe/stim-train00131_image.jpg"
    ) == (13, "aloe")
    with pytest.raises(ValueError):
        _concept_from_filename("training_images/aloe.jpg")


def test_extracted_image_uses_official_original_filename(tmp_path: Path) -> None:
    folder = tmp_path / "training_images" / "00001_aardvark"
    folder.mkdir(parents=True)
    (folder / "aardvark_01b.jpg").write_bytes(b"image")
    row = {
        "stimulus_id": "stim-train00001",
        "filename": "training_images/00001_aardvark/stim-train00001_image.jpg",
        "description": 'THINGS image with original filename "aardvark_01b.jpg".',
    }
    assert _resolve_extracted_image(row, tmp_path) == (
        "training_images/00001_aardvark/aardvark_01b.jpg"
    )


def test_git_and_sha256_checksums(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"verified")
    assert _checksum(sample, "sha256") == hashlib.sha256(b"verified").hexdigest()
    expected_git = hashlib.sha1(b"blob 8\0verified").hexdigest()
    assert _checksum(sample, "git") == expected_git


def test_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_extract(archive, tmp_path / "output")


def test_remote_file_is_serializable() -> None:
    record = RemoteFile("stimuli/file", 3, "sha256", "abc", "https://example.test/file")
    assert record.path == "stimuli/file"


def test_verified_metadata_order_is_checked(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "image_metadata.npy"
    np.save(
        metadata,
        {
            "train_img_files": ["train.jpg"],
            "test_img_files": ["test.jpg"],
        },
        allow_pickle=True,
    )
    monkeypatch.setattr(
        "multimodal_eeg.stimuli._checksum",
        lambda *_args: PINNED_FILES["stimuli/image_metadata.npy"][2],
    )
    rows = [
        StimulusRow("seen", 0, 0, "train", "training_images/c/train.jpg", "train"),
        StimulusRow("unseen", 0, 0, "test", "test_images/c/test.jpg", "test"),
    ]
    report = verify_official_metadata_order(metadata, rows)
    assert report["metadata_checksum_verified"] is True

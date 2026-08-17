"""Safe retrieval of the BraVL version 3 ThingsEEG-Text feature archive."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

FEATURE_ARCHIVE_URL = "https://ndownloader.figshare.com/files/36977293"
FEATURE_ARCHIVE_DOI = "10.6084/m9.figshare.17024591"
FEATURE_ARCHIVE_DOI = "10.6084/m9.figshare.17024591"
FEATURE_ARCHIVE_DOI = "10.6084/m9.figshare.17024591"


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)


def download_feature_archive(destination: str | Path) -> Path:
    """Download and safely extract features unless they already exist."""

    destination = Path(destination)
    dataset = destination / "ThingsEEG-Text"
    if dataset.is_dir():
        return dataset
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "ThingsEEG-Text.zip"
    with urllib.request.urlopen(FEATURE_ARCHIVE_URL) as response, archive_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, destination)
    finally:
        archive_path.unlink(missing_ok=True)
    if not dataset.is_dir():
        raise RuntimeError("The archive did not create the expected ThingsEEG-Text directory.")
    return dataset

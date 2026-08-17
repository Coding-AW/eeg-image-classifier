"""Selective, verified access to the official NEMAR THINGS-EEG2 stimuli."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

from .manifest import MANIFEST_FIELDS, StimulusRow, validate_manifest

NEMAR_DATASET_ID = "nm000232"
NEMAR_VERSION = "v1.1.0"
NEMAR_MANIFEST_URL = (
    f"https://data.nemar.org/{NEMAR_DATASET_ID}/{NEMAR_VERSION}/manifest.json"
)
RESEARCH_TERMS_URL = (
    f"https://raw.githubusercontent.com/nemarDatasets/{NEMAR_DATASET_ID}/"
    f"{NEMAR_VERSION}/stimuli/LICENSE.txt"
)


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int
    checksum_algorithm: str
    checksum: str
    bytes_url: str


# Pinned from the official v1.1.0 NEMAR manifest. The small BIDS support files are
# included because they define licensing and authoritative stimulus identifiers.
PINNED_FILES = {
    "stimuli/LICENSE.txt": (3159, "git", "f4be66f0837de90faecdda6860e2232e6a8eb69d"),
    "stimuli/image_metadata.npy": (
        656579,
        "sha256",
        "681a75633c59a356a66ca19a277429c1d48a00a98405ff96e53ee0b9f00b397c",
    ),
    "stimuli/stimuli.json": (1617, "git", "0fe331e8e457663fb17ab8517cc6c9720e5d706b"),
    "stimuli/stimuli.tsv": (4934517, "git", "42641d3bf0cb4e814f2fd15fc550475d64ea9eaf"),
    "stimuli/test_images.zip": (
        8129948,
        "sha256",
        "57ed472cd88f68ccf88e9ecbadf3da91ac222d8548684d5680be95bcf47ec14a",
    ),
    "stimuli/training_images.zip": (
        655039265,
        "sha256",
        "dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a",
    ),
}


def parse_nemar_manifest(payload: bytes | str) -> dict[str, RemoteFile]:
    """Parse and pin the exact official files required by this repository."""

    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("The NEMAR manifest must be a JSON list.")
    available = {}
    for item in raw:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError("Malformed NEMAR manifest entry.")
        if item["path"] in PINNED_FILES:
            record = RemoteFile(
                path=str(item["path"]),
                size=int(item["size"]),
                checksum_algorithm=str(item["checksum_algorithm"]),
                checksum=str(item["checksum"]),
                bytes_url=str(item["bytes_url"]),
            )
            expected = PINNED_FILES[record.path]
            if (record.size, record.checksum_algorithm, record.checksum) != expected:
                raise RuntimeError(f"Pinned NEMAR file changed: {record.path}")
            available[record.path] = record
    missing = set(PINNED_FILES) - set(available)
    if missing:
        raise RuntimeError(f"Official NEMAR manifest is missing pinned files: {sorted(missing)}")
    return available


def fetch_nemar_manifest() -> dict[str, RemoteFile]:
    request = urllib.request.Request(NEMAR_MANIFEST_URL, headers={"User-Agent": "multimodal-eeg"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return parse_nemar_manifest(response.read())


def _checksum(path: Path, algorithm: str) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
        prefix = b""
    elif algorithm == "git":
        digest = hashlib.sha1()
        prefix = f"blob {path.stat().st_size}\0".encode()
    else:
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")
    digest.update(prefix)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, record: RemoteFile) -> None:
    if path.stat().st_size != record.size:
        raise RuntimeError(f"Downloaded size mismatch for {record.path}")
    if _checksum(path, record.checksum_algorithm) != record.checksum:
        raise RuntimeError(f"Downloaded checksum mismatch for {record.path}")


def _download(record: RemoteFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        _verify(destination, record)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "multimodal-eeg"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(record.bytes_url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    _verify(partial, record)
    partial.replace(destination)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 40_000:
            raise RuntimeError("Stimulus archive contains unexpectedly many entries.")
        if sum(member.file_size for member in members) > 2_000_000_000:
            raise RuntimeError("Stimulus archive expands beyond the declared safety limit.")
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe stimulus archive member: {member.filename}")
            target = (destination / Path(*relative.parts)).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe stimulus archive member: {member.filename}")
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(f"Symlink is prohibited in stimulus archive: {member.filename}")
        archive.extractall(destination)


def stimulus_listing(records: dict[str, RemoteFile]) -> dict[str, object]:
    return {
        "dataset_id": NEMAR_DATASET_ID,
        "version": NEMAR_VERSION,
        "manifest_url": NEMAR_MANIFEST_URL,
        "download_bytes": sum(record.size for record in records.values()),
        "files": [asdict(records[path]) for path in sorted(records)],
    }


def download_stimuli(
    destination: str | Path,
    acknowledge_research_terms: bool,
    extract: bool = True,
) -> dict[str, object]:
    """Download only pinned stimulus resources and record verified provenance."""

    if not acknowledge_research_terms:
        raise PermissionError(
            "Image download requires --acknowledge-research-image-terms. "
            f"Read {RESEARCH_TERMS_URL}"
        )
    records = fetch_nemar_manifest()
    root = Path(destination)
    for relative, record in sorted(records.items()):
        _download(record, root / relative)
    if extract:
        stimuli = root / "stimuli"
        for name in ("training_images.zip", "test_images.zip"):
            marker = stimuli / f".{name}.extracted"
            if not marker.is_file():
                _safe_extract(stimuli / name, stimuli)
                marker.write_text(records[f"stimuli/{name}"].checksum + "\n", encoding="utf-8")
    report = stimulus_listing(records)
    report.update(
        {
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "research_terms_acknowledged": True,
            "extracted": extract,
        }
    )
    (root / "stimulus-download-receipt.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _concept_from_filename(filename: str) -> tuple[int, str]:
    parts = PurePosixPath(filename).parts
    if len(parts) != 3:
        raise ValueError(f"Unexpected official stimulus filename: {filename}")
    concept = parts[1]
    number, separator, name = concept.partition("_")
    if not separator or not number.isdigit() or not name:
        raise ValueError(f"Invalid concept directory: {concept}")
    return int(number) - 1, name.replace("_", " ")


def read_stimulus_table(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Read the official BIDS stimulus table in declared stimulus-ID order."""

    result = {"seen": [], "unseen": []}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = row.get("stimulus_id", "")
            if identifier.startswith("stim-train"):
                split = "seen"
                number = identifier.removeprefix("stim-train")
            elif identifier.startswith("stim-test"):
                split = "unseen"
                number = identifier.removeprefix("stim-test")
            else:
                raise ValueError(f"Unknown official stimulus ID: {identifier}")
            if not number.isdigit() or row.get("present", "").lower() != "true":
                raise ValueError(f"Invalid or absent official stimulus: {identifier}")
            expected_prefix = "training_images/" if split == "seen" else "test_images/"
            if not row["filename"].startswith(expected_prefix):
                raise ValueError(f"Stimulus path disagrees with split: {identifier}")
            result[split].append(row)
    for split, prefix in (("seen", "stim-train"), ("unseen", "stim-test")):
        result[split].sort(key=lambda row: int(row["stimulus_id"].removeprefix(prefix)))
    return result


def _resolve_extracted_image(raw: dict[str, str], images_root: str | Path) -> str:
    """Resolve a ZIP image using only filenames declared by official metadata."""

    canonical = PurePosixPath(raw["filename"])
    canonical_path = Path(images_root).joinpath(*canonical.parts)
    if canonical_path.is_file():
        return canonical.as_posix()
    match = re.search(r'original filename "([^"/\\]+)"', raw.get("description", ""))
    if match is None:
        raise FileNotFoundError(
            f"Official metadata gives no extracted filename for {raw['stimulus_id']}"
        )
    original = match.group(1)
    if original in {".", ".."} or Path(original).name != original:
        raise ValueError(f"Unsafe original stimulus filename: {original}")
    extracted = canonical.parent / original
    extracted_path = Path(images_root).joinpath(*extracted.parts)
    if not extracted_path.is_file():
        raise FileNotFoundError(f"Official stimulus image is missing: {extracted_path}")
    return extracted.as_posix()


def verify_official_metadata_order(
    metadata_path: str | Path, rows: list[StimulusRow]
) -> dict[str, object]:
    """Verify manifest filenames against the checksummed official NumPy metadata."""

    metadata_path = Path(metadata_path)
    _, algorithm, expected = PINNED_FILES["stimuli/image_metadata.npy"]
    if _checksum(metadata_path, algorithm) != expected:
        raise RuntimeError("Official image metadata checksum does not match the pinned record.")
    metadata = np.load(metadata_path, allow_pickle=True).item()
    required = {"train_img_files", "test_img_files"}
    if not isinstance(metadata, dict) or not required <= metadata.keys():
        raise ValueError("Official image metadata has an unexpected structure.")
    seen = [PurePosixPath(row.image_path).name for row in rows if row.split == "seen"]
    unseen_rows = [row for row in rows if row.split == "unseen"]
    first_by_class = {}
    for row in unseen_rows:
        first_by_class.setdefault(row.class_id, PurePosixPath(row.image_path).name)
    unseen = [first_by_class[label] for label in sorted(first_by_class)]
    if seen != list(metadata["train_img_files"]):
        raise RuntimeError("Reference manifest order disagrees with official image metadata.")
    if unseen != list(metadata["test_img_files"]):
        raise RuntimeError("External manifest order disagrees with official image metadata.")
    return {
        "metadata_checksum_verified": True,
        "reference_filename_order_verified": True,
        "external_filename_order_verified": True,
    }


def build_stimulus_manifest(
    stimulus_table: str | Path,
    images_root: str | Path,
    output: str | Path,
    expected_labels: dict[str, NDArray[np.integer]],
) -> dict[str, object]:
    """Map authoritative stimulus IDs to every BraVL EEG row without filename inference."""

    table = read_stimulus_table(stimulus_table)
    if len(table["seen"]) != 16_540 or len(table["unseen"]) != 200:
        raise RuntimeError("Official stimulus counts do not match THINGS-EEG2.")
    seen_rows = []
    for index, raw in enumerate(table["seen"]):
        class_id, class_name = _concept_from_filename(raw["filename"])
        image_path = _resolve_extracted_image(raw, images_root)
        seen_rows.append(
            StimulusRow(
                "seen", index, class_id, raw["stimulus_id"], image_path, class_name
            )
        )
    unseen_by_class = {}
    for raw in table["unseen"]:
        class_id, class_name = _concept_from_filename(raw["filename"])
        image_path = _resolve_extracted_image(raw, images_root)
        unseen_by_class[class_id] = (raw, image_path, class_name)
    if set(unseen_by_class) != set(range(200)):
        raise RuntimeError("External stimulus classes must be exactly zero through 199.")
    unseen_rows = []
    for index, label in enumerate(np.asarray(expected_labels["unseen"]).reshape(-1)):
        raw, image_path, class_name = unseen_by_class[int(label)]
        unseen_rows.append(
            StimulusRow(
                "unseen",
                index,
                int(label),
                raw["stimulus_id"],
                image_path,
                class_name,
            )
        )
    rows = seen_rows + unseen_rows
    np.testing.assert_array_equal(
        np.asarray([row.class_id for row in seen_rows]), expected_labels["seen"]
    )
    seen_concepts = {row.class_name for row in seen_rows}
    unseen_concepts = {row.class_name for row in unseen_rows}
    if seen_concepts & unseen_concepts:
        raise RuntimeError("Reference and external concept names are not disjoint.")
    validation = validate_manifest(rows, images_root, expected_labels)
    metadata_validation = verify_official_metadata_order(
        Path(images_root) / "image_metadata.npy", rows
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return {
        **validation,
        "manifest": str(path),
        "source": f"NEMAR {NEMAR_DATASET_ID} {NEMAR_VERSION} stimuli.tsv",
        "authoritative_ordering": True,
        **metadata_validation,
    }

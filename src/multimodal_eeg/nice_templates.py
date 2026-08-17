"""Leakage-safe NICE-style concept-template construction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.decomposition import PCA

from .array_utils import normalize_rows
from .embedding_features import (
    TransformersDinoV2ProbeBackend,
    TransformersVisualBackend,
    dinov2_variant_names,
)
from .manifest import read_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THINGS_ARCHIVE_SHA256 = "fb5a7cac28a27ff1ff8b723f072ed4a8309bc807a5f481b001da93f4213f3d24"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safely_extract_things_archive(archive: Path, password_file: Path, output: Path) -> None:
    """Resume extraction of the verified archive without traversal or links."""

    if sha256_file(archive) != THINGS_ARCHIVE_SHA256:
        raise ValueError("The THINGS archive checksum does not match the OSF record.")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    password_line = password_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    if ":" not in password_line:
        raise ValueError("The THINGS password file has an unexpected format.")
    password = password_line.rsplit(":", 1)[1].strip().encode()
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
            destination = (output / Path(*relative.parts)).resolve()
            if output not in destination.parents and destination != output:
                raise ValueError(f"Archive member escapes output directory: {member.filename}")
        for member in members:
            relative = PurePosixPath(member.filename)
            destination = (output / Path(*relative.parts)).resolve()
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.is_file() and destination.stat().st_size == member.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            handle.extract(member, output, pwd=password)
        files = [member for member in members if not member.is_dir()]
        if any(
            not (output / Path(*PurePosixPath(member.filename).parts)).is_file()
            or (output / Path(*PurePosixPath(member.filename).parts)).stat().st_size
            != member.file_size
            for member in files
        ):
            raise RuntimeError("THINGS extraction is incomplete after the resumable pass.")
        _save_json_atomic(
            output / ".things-extraction.json",
            {
                "complete": True,
                "archive_sha256": THINGS_ARCHIVE_SHA256,
                "files": len(files),
            },
        )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _hash_paths_resumable(paths: list[Path], cache_path: Path) -> dict[Path, str]:
    """Hash files once and checkpoint progress without trusting stale cache entries."""

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    output: dict[Path, str] = {}
    changed = 0
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        key = resolved.as_posix()
        signature = f"{stat.st_size}:{stat.st_mtime_ns}"
        record = cache.get(key)
        if not isinstance(record, dict) or record.get("signature") != signature:
            record = {"signature": signature, "sha256": sha256_file(resolved)}
            cache[key] = record
            changed += 1
            if changed % 100 == 0:
                _save_json_atomic(cache_path, cache)
        output[resolved] = str(record["sha256"])
    if changed or not cache_path.is_file():
        _save_json_atomic(cache_path, cache)
    return output


def build_template_manifest(
    images_root: Path,
    eeg_metadata: Path,
    eeg_manifest: Path,
    eeg_images_root: Path,
    output: Path,
) -> dict[str, object]:
    """Use every non-EEG THINGS image from each of the 200 test concepts."""

    metadata = np.load(eeg_metadata, allow_pickle=True).item()
    things_concepts = [str(item) for item in metadata["test_img_concepts_THINGS"]]
    if len(things_concepts) != 200 or len(set(things_concepts)) != 200:
        raise ValueError("THINGS-EEG2 metadata must identify 200 unique test concepts.")
    concepts = [item.split("_", 1)[1] for item in things_concepts]
    eeg_rows = read_manifest(eeg_manifest)
    eeg_names = {Path(row.image_path).name for row in eeg_rows}
    eeg_paths: set[Path] = set()
    for row in eeg_rows:
        eeg_path = Path(row.image_path)
        if not eeg_path.is_absolute():
            eeg_path = eeg_images_root / eeg_path
        eeg_path = eeg_path.resolve()
        if not eeg_path.is_file():
            raise FileNotFoundError(f"EEG stimulus image is missing: {eeg_path}")
        eeg_paths.add(eeg_path)
    hash_cache = output.parent / "image-hash-cache.json"
    eeg_hashes = set(_hash_paths_resumable(sorted(eeg_paths), hash_cache).values())
    test_by_class = {
        row.class_id: Path(row.image_path).name for row in eeg_rows if row.split == "unseen"
    }
    class_names = {
        row.class_id: row.class_name for row in eeg_rows if row.split == "unseen"
    }
    if len(test_by_class) != 200:
        raise ValueError("The EEG manifest must identify one test image for every concept.")
    for class_id, concept in enumerate(concepts):
        if class_names[class_id].replace(" ", "_").casefold() != concept.casefold():
            raise ValueError(f"THINGS and EEG concept names disagree for class {class_id}.")

    concept_names = {str(item).casefold() for item in concepts}
    directories = {
        path.name.casefold(): path
        for path in images_root.rglob("*")
        if path.is_dir() and path.name.casefold() in concept_names
    }
    candidates_by_class: dict[int, list[Path]] = {}
    for class_id, concept in enumerate(concepts):
        directory = directories.get(str(concept).casefold())
        if directory is None:
            raise FileNotFoundError(f"No THINGS directory found for test concept {concept!r}.")
        candidates_by_class[class_id] = [
            path
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path.suffix.casefold() in IMAGE_SUFFIXES
            and path.name not in eeg_names
        ]
    candidate_hashes = _hash_paths_resumable(
        [path for paths in candidates_by_class.values() for path in paths], hash_cache
    )

    rows = []
    for class_id, concept in enumerate(concepts):
        candidates = []
        for path in candidates_by_class[class_id]:
            candidate_hash = candidate_hashes[path.resolve()]
            if candidate_hash in eeg_hashes:
                continue
            candidates.append((path, candidate_hash))
        if not candidates:
            raise ValueError(f"No non-EEG template image remains for {concept!r}.")
        for path, candidate_hash in candidates:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "things_concept_id": things_concepts[class_id],
                    "image_id": f"{things_concepts[class_id]}/{path.name}",
                    "image_path": path.resolve().as_posix(),
                    "sha256": candidate_hash,
                    "excluded_eeg_image": test_by_class[class_id],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    report = {
        "status": "passed",
        "source_archive_sha256": THINGS_ARCHIVE_SHA256,
        "eeg_metadata_sha256": sha256_file(eeg_metadata),
        "eeg_manifest_sha256": sha256_file(eeg_manifest),
        "classes": 200,
        "images": len(rows),
        "minimum_images_per_class": min(
            sum(row["class_id"] == class_id for row in rows) for class_id in range(200)
        ),
        "eeg_image_overlap": 0,
        "eeg_image_hashes_checked": len(eeg_hashes),
        "eeg_content_set_sha256": hashlib.sha256(
            "\n".join(sorted(eeg_hashes)).encode()
        ).hexdigest(),
        "template_content_set_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["sha256"]) for row in rows)).encode()
        ).hexdigest(),
        "manifest_sha256": sha256_file(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_template_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit_template_manifest(
    template_manifest: Path, eeg_manifest: Path, eeg_images_root: Path, output: Path
) -> dict[str, object]:
    """Independently recheck gallery membership against every EEG stimulus file."""

    rows = load_template_manifest(template_manifest)
    eeg_paths = set()
    for row in read_manifest(eeg_manifest):
        path = Path(row.image_path)
        eeg_paths.add((path if path.is_absolute() else eeg_images_root / path).resolve())
    eeg_names = {path.name for path in eeg_paths}
    eeg_hashes = {sha256_file(path) for path in sorted(eeg_paths)}
    class_counts = {class_id: 0 for class_id in range(200)}
    image_ids = set()
    template_hashes = set()
    for row in rows:
        class_id = int(row["class_id"])
        if class_id not in class_counts:
            raise ValueError("Template manifest contains an out-of-range class ID.")
        path = Path(row["image_path"])
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"Template content changed after manifest creation: {path}")
        if path.name in eeg_names or digest in eeg_hashes:
            raise ValueError(f"EEG stimulus leaked into concept gallery: {path}")
        if row["image_id"] in image_ids:
            raise ValueError(f"Duplicate immutable template image ID: {row['image_id']}")
        image_ids.add(row["image_id"])
        template_hashes.add(digest)
        class_counts[class_id] += 1
    if not class_counts or min(class_counts.values()) < 1:
        raise ValueError("Every evaluation class must have an independent template image.")
    report = {
        "status": "passed",
        "classes": len(class_counts),
        "template_images": len(rows),
        "unique_image_ids": len(image_ids),
        "minimum_images_per_class": min(class_counts.values()),
        "eeg_files_checked": len(eeg_paths),
        "filename_overlap": 0,
        "content_hash_overlap": len(template_hashes & eeg_hashes),
        "template_manifest_sha256": sha256_file(template_manifest),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_json_atomic(output, report)
    return report


@dataclass
class CornetSBackend:
    """Frozen ImageNet CORnet-S logits matching the archive's 1,000-wide source."""

    device: str = "cpu"
    cache_dir: Path = Path("artifacts/model-cache/torch")

    def __post_init__(self) -> None:
        import cornet
        import torch
        from torchvision.transforms import v2

        self.torch = torch
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(self.cache_dir.resolve()))
        self.model = cornet.cornet_s(pretrained=True, map_location=self.device)
        self.model = self.model.eval().requires_grad_(False).to(self.device)
        self.transform = v2.Compose(
            [
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.checkpoint = "dicarlolab/CORnet@d0cc17d4b34ad44dedb01683b70eafd15515adad"
        checkpoint_files = list(self.cache_dir.rglob("cornet_s-*.pth"))
        if len(checkpoint_files) != 1:
            raise FileNotFoundError("Expected one cached CORnet-S checkpoint file.")
        self.checkpoint_sha256 = sha256_file(checkpoint_files[0])

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(self.transform(source.convert("RGB")))
        batch = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            values = self.model(batch)
        return values.detach().cpu().numpy()


def _batched(paths: list[Path], encode, batch_size: int) -> np.ndarray:
    return np.concatenate(
        [encode(paths[start : start + batch_size]) for start in range(0, len(paths), batch_size)]
    ).astype(np.float64)


def _batched_resumable(
    paths: list[Path], encode, batch_size: int, cache: Path
) -> np.ndarray:
    """Encode deterministic batches and reuse only shape-valid completed batches."""

    cache.mkdir(parents=True, exist_ok=True)
    batches = []
    for start in range(0, len(paths), batch_size):
        selected = paths[start : start + batch_size]
        identity = hashlib.sha256(
            "\n".join(path.resolve().as_posix() for path in selected).encode()
        ).hexdigest()[:16]
        destination = cache / f"{start:06d}-{identity}.npy"
        values = np.load(destination, allow_pickle=False) if destination.is_file() else None
        if values is None or len(values) != len(selected) or values.ndim != 2:
            values = np.asarray(encode(selected))
            temporary = destination.with_suffix(".tmp.npy")
            np.save(temporary, values, allow_pickle=False)
            os.replace(temporary, destination)
        batches.append(values)
    return np.concatenate(batches).astype(np.float64)


def extract_cornet_shard(
    template_manifest: Path,
    eeg_manifest: Path,
    eeg_images_root: Path,
    output: Path,
    *,
    role: str,
    batch_size: int,
    shard: int,
    shards: int,
) -> dict[str, int]:
    """Populate one disjoint CORnet-S cache shard for parallel CPU extraction."""

    if role not in {"reference", "template"}:
        raise ValueError("CORnet shard role must be reference or template.")
    if shards < 1 or shard not in range(shards) or batch_size < 1:
        raise ValueError("Invalid CORnet shard assignment.")
    if role == "reference":
        rows = sorted(
            (row for row in read_manifest(eeg_manifest) if row.split == "seen"),
            key=lambda row: row.eeg_row_index,
        )
        paths = [(eeg_images_root / row.image_path).resolve() for row in rows]
    else:
        paths = [Path(row["image_path"]) for row in load_template_manifest(template_manifest)]
    cache = output / "batch-cache" / f"cornet-{role}"
    cache.mkdir(parents=True, exist_ok=True)
    backend = CornetSBackend()
    completed = 0
    for batch_index, start in enumerate(range(0, len(paths), batch_size)):
        if batch_index % shards != shard:
            continue
        selected = paths[start : start + batch_size]
        identity = hashlib.sha256(
            "\n".join(path.resolve().as_posix() for path in selected).encode()
        ).hexdigest()[:16]
        destination = cache / f"{start:06d}-{identity}.npy"
        if destination.is_file():
            values = np.load(destination, allow_pickle=False)
            if len(values) != len(selected) or values.ndim != 2:
                raise ValueError(f"Invalid CORnet cache batch: {destination}")
        else:
            values = np.asarray(backend.encode_images(selected))
            temporary = destination.with_suffix(".tmp.npy")
            np.save(temporary, values, allow_pickle=False)
            os.replace(temporary, destination)
        completed += len(selected)
    return {"shard": shard, "shards": shards, "images": completed}


def _concept_centers(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    if not np.array_equal(classes, np.arange(200)):
        raise ValueError("Template labels must cover classes 0 through 199 exactly.")
    per_image = normalize_rows(values.astype(np.float64))
    centers = np.stack([per_image[labels == label].mean(axis=0) for label in classes])
    return normalize_rows(centers), classes


def extract_template_features(
    template_manifest: Path,
    eeg_manifest: Path,
    eeg_images_root: Path,
    output: Path,
    *,
    batch_size: int = 16,
    transformer_batch_size: int = 16,
) -> dict[str, object]:
    """Extract compatible CORnet-S and all frozen DINOv2 template variants."""

    if batch_size < 1 or transformer_batch_size < 1:
        raise ValueError("Template extraction batch sizes must be positive.")

    template_rows = load_template_manifest(template_manifest)
    template_paths = [Path(row["image_path"]) for row in template_rows]
    template_labels = np.asarray([int(row["class_id"]) for row in template_rows])
    reference_rows = sorted(
        (row for row in read_manifest(eeg_manifest) if row.split == "seen"),
        key=lambda row: row.eeg_row_index,
    )
    reference_paths = [(eeg_images_root / row.image_path).resolve() for row in reference_rows]
    reference_labels = np.asarray([row.class_id for row in reference_rows])
    output.mkdir(parents=True, exist_ok=True)

    cornet = CornetSBackend()
    cache = output / "batch-cache"
    cornet_reference_raw = _batched_resumable(
        reference_paths, cornet.encode_images, batch_size, cache / "cornet-reference"
    )
    cornet_template_raw = _batched_resumable(
        template_paths, cornet.encode_images, batch_size, cache / "cornet-template"
    )
    pca = PCA(n_components=100, random_state=17).fit(cornet_reference_raw)
    cornet_reference = pca.transform(cornet_reference_raw)
    cornet_template = pca.transform(cornet_template_raw)
    cornet_centers, center_labels = _concept_centers(cornet_template, template_labels)
    cornet_output = output / "cornet-s.npz"
    np.savez_compressed(
        cornet_output,
        reference=normalize_rows(cornet_reference).astype(np.float32),
        reference_labels=reference_labels,
        templates=cornet_centers.astype(np.float32),
        template_labels=center_labels,
    )

    dino = TransformersDinoV2ProbeBackend("facebook/dinov2-base", local_files_only=True)
    variant_batches: dict[str, list[np.ndarray]] = {}
    dino_cache = cache / "dinov2-template"
    dino_cache.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(template_paths), transformer_batch_size):
        selected = template_paths[start : start + transformer_batch_size]
        identity = hashlib.sha256(
            "\n".join(path.resolve().as_posix() for path in selected).encode()
        ).hexdigest()[:16]
        destination = dino_cache / f"{start:06d}-{identity}.npz"
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as archive:
                variants = {name: np.asarray(archive[name]) for name in archive.files}
        else:
            variants = dino.encode_variants(selected)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **variants)
            os.replace(temporary, destination)
        if set(variants) != set(dinov2_variant_names()):
            raise ValueError("DINOv2 batch does not contain all predeclared variants.")
        for name, values in variants.items():
            if len(values) != len(selected) or values.ndim != 2:
                raise ValueError(f"Invalid cached DINOv2 batch for {name}.")
            variant_batches.setdefault(name, []).append(values)
    dino_hashes = {}
    for name, batches in variant_batches.items():
        centers, center_labels = _concept_centers(np.concatenate(batches), template_labels)
        destination = output / f"{name}.npz"
        np.savez_compressed(
            destination,
            templates=centers.astype(np.float32),
            template_labels=center_labels,
        )
        dino_hashes[name] = sha256_file(destination)
    metadata = {
        "complete": True,
        "template_manifest_sha256": sha256_file(template_manifest),
        "cornet_checkpoint": cornet.checkpoint,
        "cornet_checkpoint_sha256": cornet.checkpoint_sha256,
        "cornet_artifact_sha256": sha256_file(cornet_output),
        "cornet_pca_fit": "16,540 reference images only",
        "dinov2_checkpoint": dino.checkpoint,
        "dinov2_revision": dino.revision,
        "dinov2_artifact_sha256": dino_hashes,
        "template_images": len(template_rows),
        "cornet_batch_size": batch_size,
        "dinov2_batch_size": transformer_batch_size,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_official_nice_centers(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or len(values) != 200:
        raise ValueError("Official NICE centers must be a 200-row matrix.")
    return normalize_rows(values.astype(np.float64))


def validate_clip_templates(
    template_manifest: Path,
    official_centers: Path,
    output: Path,
    *,
    batch_size: int = 16,
) -> dict[str, object]:
    """Compare recreated centroids with NICE without using the result for selection."""

    rows = load_template_manifest(template_manifest)
    paths = [Path(row["image_path"]) for row in rows]
    labels = np.asarray([int(row["class_id"]) for row in rows])
    backend = TransformersVisualBackend(
        "openai/clip-vit-large-patch14",
        local_files_only=True,
        cache_dir=Path("artifacts/model-cache/huggingface"),
    )
    features = _batched_resumable(
        paths,
        backend.encode_images,
        batch_size,
        output / "batch-cache" / "clip-template",
    )
    recreated, classes = _concept_centers(features, labels)
    official = load_official_nice_centers(official_centers)
    similarities = recreated @ official.T
    diagonal = similarities[np.arange(200), np.arange(200)]
    nearest = np.argmax(similarities, axis=1)
    rows_out = [
        {
            "class_id": int(class_id),
            "same_class_cosine": float(diagonal[class_id]),
            "nearest_official_class": int(nearest[class_id]),
            "nearest_class_agrees": bool(nearest[class_id] == class_id),
        }
        for class_id in classes
    ]
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "clip-template-validation.csv", rows_out)
    report = {
        "diagnostic_only": True,
        "used_for_selection": False,
        "checkpoint": backend.checkpoint,
        "revision": backend.revision,
        "mean_same_class_cosine": float(diagonal.mean()),
        "nearest_template_agreement": float(np.mean(nearest == classes)),
        "official_centers_sha256": sha256_file(official_centers),
    }
    (output / "clip-template-validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report

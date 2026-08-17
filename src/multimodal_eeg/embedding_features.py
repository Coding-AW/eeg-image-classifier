"""Frozen visual embeddings aligned to the authoritative stimulus manifest."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .array_utils import normalize_rows
from .manifest import read_manifest, validate_manifest

MODEL_REGISTRY = {
    "dinov2": "facebook/dinov2-base",
}

DINO_LAYERS = (3, 6, 9, 12)
DINO_POOLING = ("cls", "mean-patch")


def dinov2_variant_names() -> tuple[str, ...]:
    """Return the complete, ordered frozen DINOv2 representation probe."""

    return tuple(
        f"layer-{layer:02d}-{pooling}" for layer in DINO_LAYERS for pooling in DINO_POOLING
    )


@dataclass(frozen=True)
class VisualFeatureArtifact:
    """Frozen image vectors in exact EEG-row order."""

    split: str
    row_indices: NDArray[np.int64]
    labels: NDArray[np.int64]
    stimulus_ids: NDArray[np.str_]
    features: NDArray[np.float64]
    metadata: dict[str, object]


class VisualBackend(Protocol):
    checkpoint: str
    revision: str
    pooling: str
    package_versions: dict[str, str]

    def encode_images(self, paths: list[Path]) -> NDArray[np.floating]: ...


class TransformersVisualBackend:
    """Hugging Face image-feature wrapper with an explicit pooling contract."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cpu",
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:  # pragma: no cover - optional dependency
            message = 'Install embedding support with: pip install -e ".[nice]"'
            raise RuntimeError(message) from error

        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            checkpoint, local_files_only=local_files_only
        ).eval().to(device)
        self._device = device
        self.checkpoint = checkpoint
        self.revision = str(getattr(self._model.config, "_commit_hash", None) or "unrecorded")
        self.pooling = "pooler_or_cls"
        self.image_processor = "checkpoint-default-slow"
        self.package_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }

    def encode_images(self, paths: list[Path]) -> NDArray[np.floating]:
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                output = self._model.get_image_features(**inputs)
            else:
                result = self._model(**inputs)
                output = getattr(result, "pooler_output", None)
                if output is None:
                    output = result.last_hidden_state[:, 0]
        return np.asarray(output.detach().cpu().numpy())


class TransformersDinoV2ProbeBackend(TransformersVisualBackend):
    """Extract selected hidden states without updating any DINOv2 weights."""

    pooling = "layer-probe:cls-or-mean-patch"

    def normalize_pooled(self, values):
        """Apply the checkpoint's final LayerNorm after either pooling rule."""

        return self._model.layernorm(values)

    def encode_variants(self, paths: list[Path]) -> dict[str, NDArray[np.floating]]:
        from PIL import Image

        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            result = self._model(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = result.hidden_states
        if hidden_states is None or len(hidden_states) <= max(DINO_LAYERS):
            raise ValueError("DINOv2 did not return every requested hidden state.")
        variants = {}
        for layer in DINO_LAYERS:
            state = hidden_states[layer]
            variants[f"layer-{layer:02d}-cls"] = self.normalize_pooled(state[:, 0])
            variants[f"layer-{layer:02d}-mean-patch"] = self.normalize_pooled(
                state[:, 1:].mean(dim=1)
            )
        return {
            name: np.asarray(value.detach().cpu().numpy()) for name, value in variants.items()
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(array: NDArray[np.generic]) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def extract_visual_features(
    manifest: str | Path,
    images_root: str | Path,
    output: str | Path,
    backend: VisualBackend,
    *,
    batch_size: int = 32,
    resume: bool = False,
) -> dict[str, object]:
    """Extract each unique image once and expand it to authoritative EEG-row order."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rows = read_manifest(manifest)
    validation = validate_manifest(rows, images_root)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    state_path = output_root / "extraction-state.json"
    batch_root = output_root / "batches"
    manifest_hash = _sha256(Path(manifest))
    expected = [output_root / f"{split}.npz" for split in validation["splits"]]
    if resume and metadata_path.is_file() and all(path.is_file() for path in expected):
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete") is True
            and existing.get("checkpoint") == backend.checkpoint
            and existing.get("revision") == backend.revision
            and existing.get("manifest_sha256") == manifest_hash
        ):
            return existing

    root = Path(images_root).resolve()
    unique_paths: dict[str, Path] = {}
    content_hashes: dict[str, str] = {}
    for row in rows:
        path = (root / row.image_path).resolve()
        digest = _sha256(path)
        previous = content_hashes.setdefault(row.stimulus_id, digest)
        if previous != digest:
            raise ValueError(f"Stimulus {row.stimulus_id} maps to different file contents.")
        unique_paths.setdefault(row.stimulus_id, path)

    identifiers = sorted(unique_paths)
    state = {
        "checkpoint": backend.checkpoint,
        "revision": backend.revision,
        "manifest_sha256": manifest_hash,
        "batch_size": batch_size,
        "unique_stimuli": len(identifiers),
    }
    if resume and state_path.is_file():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError("Existing extraction batches do not match this run.")
    else:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    batch_root.mkdir(parents=True, exist_ok=True)
    batches = []
    for start in range(0, len(identifiers), batch_size):
        selected = identifiers[start : start + batch_size]
        batch_path = batch_root / f"batch-{start // batch_size:05d}.npy"
        if resume and batch_path.is_file():
            batch = np.load(batch_path, allow_pickle=False)
            if len(batch) != len(selected):
                raise ValueError(f"Incomplete extraction checkpoint: {batch_path}")
        else:
            batch = np.asarray(backend.encode_images([unique_paths[item] for item in selected]))
            np.save(batch_path, batch, allow_pickle=False)
        batches.append(batch)
    unique_features = normalize_rows(np.concatenate(batches).astype(np.float64))
    if not np.isfinite(unique_features).all() or unique_features.ndim != 2:
        raise ValueError("The feature extractor returned invalid vectors.")
    lookup = dict(zip(identifiers, unique_features, strict=True))

    artifact_hashes = {}
    for split in validation["splits"]:
        selected_rows = (row for row in rows if row.split == split)
        ordered = sorted(selected_rows, key=lambda row: row.eeg_row_index)
        features = np.stack([lookup[row.stimulus_id] for row in ordered]).astype(np.float32)
        row_indices = np.asarray([row.eeg_row_index for row in ordered], dtype=np.int64)
        labels = np.asarray([row.class_id for row in ordered], dtype=np.int64)
        stimulus_ids = np.asarray([row.stimulus_id for row in ordered], dtype=np.str_)
        np.savez_compressed(
            output_root / f"{split}.npz",
            row_indices=row_indices,
            labels=labels,
            stimulus_ids=stimulus_ids,
            features=features,
        )
        artifact_hashes[split] = _array_hash(features)

    metadata = {
        "complete": True,
        "checkpoint": backend.checkpoint,
        "revision": backend.revision,
        "pooling": backend.pooling,
        "package_versions": backend.package_versions,
        "manifest_sha256": manifest_hash,
        "unique_stimuli": len(identifiers),
        "dimension": int(unique_features.shape[1]),
        "artifact_sha256": artifact_hashes,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def extract_dinov2_probe(
    manifest: str | Path,
    images_root: str | Path,
    output: str | Path,
    backend: TransformersDinoV2ProbeBackend,
    *,
    batch_size: int = 32,
    resume: bool = False,
) -> dict[str, object]:
    """Extract the fixed DINOv2 layer/pooling probe in one pass per image batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rows = read_manifest(manifest)
    validation = validate_manifest(rows, images_root)
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_hash = _sha256(Path(manifest))
    root = Path(images_root).resolve()
    unique_paths: dict[str, Path] = {}
    content_hashes: dict[str, str] = {}
    for row in rows:
        path = (root / row.image_path).resolve()
        if row.stimulus_id not in unique_paths:
            unique_paths[row.stimulus_id] = path
            content_hashes[row.stimulus_id] = _sha256(path)
        elif unique_paths[row.stimulus_id] != path and content_hashes[row.stimulus_id] != _sha256(
            path
        ):
            raise ValueError(f"Stimulus {row.stimulus_id} maps to different file contents.")
    identifiers = sorted(unique_paths)
    variants = dinov2_variant_names()
    state = {
        "checkpoint": backend.checkpoint,
        "revision": backend.revision,
        "manifest_sha256": manifest_hash,
        "batch_size": batch_size,
        "unique_stimuli": len(identifiers),
        "variants": list(variants),
        "encoder_frozen": True,
    }
    state_path = output_root / "extraction-state.json"
    metadata_path = output_root / "metadata.json"
    expected = [
        output_root / name / f"{split}.npz"
        for name in variants
        for split in validation["splits"]
    ]
    if resume and metadata_path.is_file() and all(path.is_file() for path in expected):
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True and existing.get("state") == state:
            return existing
    if resume and state_path.is_file():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError("Existing DINOv2 probe batches do not match this run.")
    else:
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    for name in variants:
        (output_root / name / "batches").mkdir(parents=True, exist_ok=True)
    for start in range(0, len(identifiers), batch_size):
        selected = identifiers[start : start + batch_size]
        batch_index = start // batch_size
        paths = {
            name: output_root / name / "batches" / f"batch-{batch_index:05d}.npy"
            for name in variants
        }
        if resume and all(path.is_file() for path in paths.values()):
            saved = [np.load(path, mmap_mode="r", allow_pickle=False) for path in paths.values()]
            widths = {value.shape[1] for value in saved if value.ndim == 2}
            incomplete = any(
                value.ndim != 2 or len(value) != len(selected) for value in saved
            ) or len(widths) != 1
            if incomplete:
                raise ValueError("Incomplete DINOv2 probe checkpoint.")
            if batch_index % 100 == 0:
                print(f"Verified existing DINOv2 batch {batch_index}", flush=True)
            continue
        encoded = backend.encode_variants([unique_paths[item] for item in selected])
        if tuple(encoded) != variants:
            raise ValueError("DINOv2 backend returned an unexpected representation probe.")
        for name, path in paths.items():
            values = np.asarray(encoded[name])
            if values.ndim != 2 or len(values) != len(selected) or not np.isfinite(values).all():
                raise ValueError(f"DINOv2 returned invalid vectors for {name}.")
            np.save(path, values, allow_pickle=False)
        if batch_index % 25 == 0:
            print(f"Extracted DINOv2 batch {batch_index}", flush=True)

    variant_metadata = {}
    for name in variants:
        batch_root = output_root / name / "batches"
        batch_paths = sorted(batch_root.glob("batch-*.npy"))
        unique = normalize_rows(
            np.concatenate([np.load(path, allow_pickle=False) for path in batch_paths]).astype(
                np.float64
            )
        )
        if len(unique) != len(identifiers):
            raise ValueError(f"DINOv2 probe is incomplete for {name}.")
        lookup = dict(zip(identifiers, unique, strict=True))
        hashes = {}
        for split in validation["splits"]:
            ordered = sorted(
                (row for row in rows if row.split == split), key=lambda row: row.eeg_row_index
            )
            features = np.stack([lookup[row.stimulus_id] for row in ordered]).astype(np.float32)
            arrays = {
                "row_indices": np.asarray([row.eeg_row_index for row in ordered], dtype=np.int64),
                "labels": np.asarray([row.class_id for row in ordered], dtype=np.int64),
                "stimulus_ids": np.asarray([row.stimulus_id for row in ordered], dtype=np.str_),
                "features": features,
            }
            np.savez_compressed(output_root / name / f"{split}.npz", **arrays)
            hashes[split] = _array_hash(features)
        layer, pooling = name.removeprefix("layer-").split("-", 1)
        item = {
            "complete": True,
            "checkpoint": backend.checkpoint,
            "revision": backend.revision,
            "layer": int(layer),
            "pooling": pooling,
            "encoder_frozen": True,
            "normalization": "model-final-layernorm-after-pooling-then-row-l2",
            "manifest_sha256": manifest_hash,
            "dimension": int(unique.shape[1]),
            "artifact_sha256": hashes,
            "package_versions": backend.package_versions,
            "image_processor": getattr(backend, "image_processor", "backend-defined"),
        }
        (output_root / name / "metadata.json").write_text(
            json.dumps(item, indent=2), encoding="utf-8"
        )
        variant_metadata[name] = item
    metadata = {"complete": True, "state": state, "variants": variant_metadata}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_visual_artifact(root: str | Path, split: str) -> VisualFeatureArtifact:
    """Load and validate a generic visual feature artifact."""

    if split not in {"seen", "unseen"}:
        raise ValueError("split must be seen or unseen.")
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        artifact = VisualFeatureArtifact(
            split,
            np.asarray(archive["row_indices"], dtype=np.int64),
            np.asarray(archive["labels"], dtype=np.int64),
            np.asarray(archive["stimulus_ids"], dtype=np.str_),
            np.asarray(archive["features"], dtype=np.float64),
            metadata,
        )
    count = len(artifact.labels)
    if artifact.features.ndim != 2 or any(
        len(value) != count
        for value in (artifact.row_indices, artifact.stimulus_ids, artifact.features)
    ):
        raise ValueError("Visual artifact arrays are not aligned.")
    if not np.array_equal(artifact.row_indices, np.arange(count)):
        raise ValueError("Visual artifact rows are not in authoritative EEG order.")
    if not np.isfinite(artifact.features).all():
        raise ValueError("Visual artifact contains non-finite vectors.")
    return artifact

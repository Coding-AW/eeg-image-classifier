"""Complete reference-only study with literature-standard averaged test EEG."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .array_utils import normalize_rows
from .averaging import (
    CONTROL_SEEDS,
    REPETITION_COUNTS,
    average_groups,
    file_sha256,
    independent_ranks,
    load_repeated_eeg,
)
from .classical_study import (
    PreprocessingSpec,
    calibrated_scores,
    embedding_eda,
    holm_adjust,
    paired_change_statistics,
    ranking_metrics,
    run_baseline_anchored_participant,
)
from .config import ExperimentConfig
from .data import Modalities, load_feature_split, normalize_labels
from .decoder import CandidateBank, fit_decoder, save_bundle
from .embedding_features import dinov2_variant_names, load_visual_artifact
from .manifest import read_manifest
from .representation_probe import select_dinov2_variant
from .statistics import aggregate_metrics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _effective_rank(values: np.ndarray) -> tuple[float, float]:
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False) ** 2
    probabilities = singular / max(singular.sum(), 1e-12)
    positive = probabilities > 0
    rank = float(np.exp(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    leading = float(probabilities[0]) if len(probabilities) else 0.0
    return rank, leading


def _eeg_eda(
    participant: str,
    reference: Modalities,
    raw: np.ndarray,
    labels: np.ndarray,
    full_average: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for split, values, classes in (
        ("reference", reference.eeg, reference.labels),
        ("evaluation_80_average", full_average, np.unique(labels)),
    ):
        rank, leading = _effective_rank(values)
        rows.append(
            {
                "participant": participant,
                "split": split,
                "rows": len(values),
                "classes": len(np.unique(classes)),
                "width": values.shape[1],
                "finite": bool(np.isfinite(values).all()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "effective_rank": rank,
                "leading_component_energy": leading,
                "within_stimulus_variance": (
                    "not_available"
                    if split == "reference"
                    else float(
                        np.mean(
                            [raw[labels == label].var(axis=0) for label in np.unique(labels)]
                        )
                    )
                ),
            }
        )
    return rows


def _averaging_eda(
    participant: str,
    raw: np.ndarray,
    labels: np.ndarray,
    trial_ids: np.ndarray,
    full: np.ndarray,
    seed: int,
) -> list[dict[str, object]]:
    truth = np.unique(labels)
    groups = [
        raw[labels == label][np.argsort(trial_ids[labels == label])]
        for label in truth
    ]
    first = np.stack([group[:40].mean(axis=0) for group in groups]).reshape(200, -1)
    last = np.stack([group[40:].mean(axis=0) for group in groups]).reshape(200, -1)
    rows = []
    for count in REPETITION_COUNTS:
        values, _ = average_groups(raw, labels, trial_ids, count, seed + count)
        similarity = np.sum(normalize_rows(values) * normalize_rows(full), axis=1)
        rows.append(
            {
                "participant": participant,
                "repetitions": count,
                "mean_similarity_to_80": float(similarity.mean()),
                "mean_feature_variance": float(values.var(axis=0, ddof=1).mean()),
                "first_last_40_similarity": float(
                    np.sum(normalize_rows(first) * normalize_rows(last), axis=1).mean()
                ),
            }
        )
    return rows


def _select_dinov2(dataset: Path, probe: Path, config: ExperimentConfig) -> dict[str, object]:
    references = {}
    for variant in dinov2_variant_names():
        artifact = load_visual_artifact(probe / variant, "seen")
        participants = {}
        for participant in config.subjects:
            base = load_feature_split(dataset, "seen", config, participant)
            if not np.array_equal(base.labels, artifact.labels):
                raise ValueError(f"{variant} reference labels disagree for {participant}.")
            participants[participant] = Modalities(base.eeg, artifact.features, None, base.labels)
        references[variant] = participants
    return select_dinov2_variant(references, config)


def _literature_protocol() -> list[dict[str, object]]:
    return [
        {
            "study": "current NICE-style averaged rerun",
            "dataset": "THINGS-EEG2 via BraVL",
            "participants": 10,
            "channels": 17,
            "time_window": "archive samples 27:60 at 100 Hz",
            "training_average": "archive-provided four-presentation image average",
            "test_average": 80,
            "candidates": "200 centroids of independently sourced non-EEG THINGS images",
            "visual_space": "CORnet-S or reference-selected DINOv2",
            "setting": "participant-dependent",
            "primary_retrieval": "cosine",
            "aggregation": "participant mean and sample SD",
        },
        {
            "study": "NICE (Song et al., ICLR 2024)",
            "dataset": "THINGS-EEG2",
            "participants": 10,
            "channels": 63,
            "time_window": "0:1000 ms at 250 Hz",
            "training_average": "four presentations per image",
            "test_average": 80,
            "candidates": "200 test-condition image centres",
            "visual_space": "CLIP",
            "setting": "participant-dependent",
            "primary_retrieval": "cosine/logit similarity",
            "aggregation": "participant mean; five training runs reported",
        },
    ]


def _final_reports(
    output: Path,
    metric_rows: list[dict[str, object]],
    curve_rows: list[dict[str, object]],
) -> None:
    """Write participant-paired statistics, a curve plot, and a plain-language report."""

    primary = [
        row
        for row in metric_rows
        if row["stage"] == "S3" and row["metric"] in {"top1_accuracy", "top5_accuracy"}
    ]
    comparisons = []
    for metric in ("top1_accuracy", "top5_accuracy"):
        by_source = {
            source: {
                str(row["participant"]): float(row["score"])
                for row in primary
                if row["semantic_source"] == source and row["metric"] == metric
            }
            for source in ("cornet-s", "dinov2")
        }
        participants = sorted(set(by_source["cornet-s"]) & set(by_source["dinov2"]))
        differences = np.asarray(
            [by_source["dinov2"][item] - by_source["cornet-s"][item] for item in participants]
        )
        comparisons.append(
            {
                "comparison": "dinov2_minus_cornet-s",
                "metric": metric,
                "participants": len(participants),
                **paired_change_statistics(differences),
            }
        )
    adjusted = holm_adjust([float(row["sign_flip_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_p"] = value
    _write_csv(output / "paired-comparisons.csv", comparisons)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for source in ("cornet-s", "dinov2"):
        source_rows = [row for row in curve_rows if row["semantic_source"] == source]
        counts = sorted({int(row["repetitions"]) for row in source_rows})
        means = [
            float(
                np.mean(
                    [
                        float(row["top1_mean"])
                        for row in source_rows
                        if int(row["repetitions"]) == count
                    ]
                )
            )
            for count in counts
        ]
        axis.plot(counts, means, marker="o", label=source)
    axis.axhline(1 / 200, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set(xlabel="EEG repetitions averaged", ylabel="Mean participant top-1 accuracy")
    axis.set_xticks(REPETITION_COUNTS)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "repetition-learning-curve.png", dpi=180)
    plt.close(figure)

    top1 = {
        source: np.asarray(
            [
                float(row["score"])
                for row in primary
                if row["semantic_source"] == source and row["metric"] == "top1_accuracy"
            ]
        )
        for source in ("cornet-s", "dinov2")
    }
    cornet_mean = 100 * top1["cornet-s"].mean()
    cornet_sd = 100 * top1["cornet-s"].std(ddof=1)
    dino_mean = 100 * top1["dinov2"].mean()
    dino_sd = 100 * top1["dinov2"].std(ddof=1)
    report = (
        "# Averaged-EEG study report\n\n"
        "## What was done\n\n"
        "The original pipeline was rerun without changing its model or tuning rules. "
        "The only primary evaluation change was to replace 16,000 separate test trials "
        "with 200 EEG vectors. Each vector is the arithmetic mean of the 80 recordings "
        "made while the participant viewed one known test image.\n\n"
        "All representation, preprocessing, Ridge and retrieval choices were made using "
        "the 1,654 reference categories only. The 200 test averages were used only after "
        "those choices were locked. Cosine retrieval is the primary literature-comparable "
        "score; CSLS is secondary because it examines the test batch as a whole.\n\n"
        "## Main result\n\n"
        f"Across ten participants, CORnet-S reached {cornet_mean:.2f}% mean top-1 "
        f"accuracy (SD {cornet_sd:.2f} points). DINOv2 reached {dino_mean:.2f}% "
        f"(SD {dino_sd:.2f} points). Random chance is 0.5%.\n\n"
        "This means concept-template retrieval after averaging 80 already-grouped repetitions. "
        "It does not mean that the system can classify a new single EEG trial at this "
        "accuracy.\n\n"
        "## Leakage status\n\n"
        "The local checks passed for reference-only fitting, row-order invariance, "
        "independent metric calculation and unchanged model bundles. Broken-pair and "
        "permutation controls stayed below the predeclared 5% ceiling. The result remains "
        "quarantined because the upstream BraVL archive was not recreated from raw EEG, "
        "and because forming each average requires knowing which 80 trials belong to the "
        "same stimulus.\n"
    )
    (output / "plain-language-report.md").write_text(report, encoding="utf-8")


def _candidate_ledger(participant_reports: dict[str, object]) -> list[dict[str, object]]:
    """Flatten every reference-fold S2-S4 candidate into an auditable table."""

    rows = []
    for participant, report in participant_reports.items():
        for source, source_report in report["sources"].items():
            selection = source_report["selection"]
            groups = (
                ("S2", selection["preprocessing_records"]),
                ("S3", selection["tuning_records"]),
                ("S4", selection["calibration_records"]),
            )
            for stage, records in groups:
                for index, record in enumerate(records):
                    rows.append(
                        {
                            "participant": participant,
                            "semantic_source": source,
                            "stage": stage,
                            "candidate_index": index,
                            "candidate": json.dumps(
                                {
                                    key: value
                                    for key, value in record.items()
                                    if key != "fold_scores"
                                },
                                sort_keys=True,
                            ),
                            "fold_1": record["fold_scores"][0],
                            "fold_2": record["fold_scores"][1],
                            "fold_3": record["fold_scores"][2],
                        }
                    )
    return rows


def run_averaged_study(
    dataset: str | Path,
    alignment_manifest: str | Path,
    dinov2_probe: str | Path,
    config_path: str | Path,
    output: str | Path,
    bundles: str | Path,
    *,
    averaging_rule: str,
    primary_repetitions: int,
    repetition_curve: tuple[int, ...],
    primary_retrieval: str,
    random_seed: int | None = None,
    template_features: str | Path,
) -> dict[str, object]:
    """Run S0-S4 with 80-trial EEG means and independent concept templates."""

    if averaging_rule != "arithmetic-within-stimulus":
        raise ValueError("Averaged study requires arithmetic-within-stimulus averaging.")
    if primary_repetitions != 80 or tuple(repetition_curve) != REPETITION_COUNTS:
        raise ValueError("The locked protocol requires 80 primary and 1 2 5 10 20 40 80 curve.")
    if primary_retrieval != "cosine":
        raise ValueError("Cosine is the locked literature-comparable primary retrieval rule.")

    dataset, probe, output, bundles = map(
        Path, (dataset, dinov2_probe, output, bundles)
    )
    output.mkdir(parents=True, exist_ok=True)
    bundles.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(config_path).read_text(encoding="utf-8"))
    experiment = protocol.get("experiment", protocol)
    tuple_fields = {
        "subjects",
        "ridge_alpha_candidates",
        "classical_semantic_dimensions",
        "classical_eeg_dimensions",
        "classical_remove_components",
        "classical_whitening",
        "classical_csls_neighbors",
    }
    experiment = {
        key: tuple(value) if key in tuple_fields else value for key, value in experiment.items()
    }
    config = ExperimentConfig(**experiment)
    if random_seed is not None:
        config = replace(config, random_seed=random_seed)
    selection_path = output / "dinov2-representation-selection.json"
    probe_metadata = sorted(probe.glob("*/metadata.json"))
    if len(probe_metadata) != 8:
        raise ValueError("DINOv2 probe must contain eight complete variant metadata files.")
    probe_metadata_sha256 = hashlib.sha256(
        "\n".join(file_sha256(path) for path in probe_metadata).encode()
    ).hexdigest()
    selection_identity = {
        "random_seed": config.random_seed,
        "config_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "probe": probe.resolve().as_posix(),
        "probe_metadata_sha256": probe_metadata_sha256,
    }
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("resume_identity") != selection_identity:
            raise ValueError("Saved DINOv2 selection belongs to a different locked run.")
    else:
        selection = _select_dinov2(dataset, probe, config)
        selection["resume_identity"] = selection_identity
        _write_json_atomic(selection_path, selection)
    selected_variant = str(selection["selected_variant"])

    manifest = read_manifest(alignment_manifest)
    seen_names = {row.class_name for row in manifest if row.split == "seen"}
    unseen = [row for row in manifest if row.split == "unseen"]
    unseen_names = {row.class_name for row in unseen}
    if len(unseen) != 16_000 or seen_names & unseen_names:
        raise ValueError("Manifest does not prove a 16,000-row class-disjoint evaluation split.")
    stimulus_sets = {}
    for row in unseen:
        stimulus_sets.setdefault(row.class_id, set()).add(row.stimulus_id)
    if set(stimulus_sets) != set(range(200)) or any(
        len(value) != 1 for value in stimulus_sets.values()
    ):
        raise ValueError("Every evaluation class must map to exactly one immutable stimulus ID.")
    ordered_unseen = sorted(unseen, key=lambda row: row.eeg_row_index)
    if np.asarray([row.eeg_row_index for row in ordered_unseen]).tolist() != list(range(16_000)):
        raise ValueError("Evaluation manifest must cover every physical EEG row exactly once.")
    stimulus_to_class: dict[str, int] = {}
    for row in ordered_unseen:
        previous = stimulus_to_class.setdefault(row.stimulus_id, row.class_id)
        if previous != row.class_id:
            raise ValueError("One immutable stimulus ID maps to multiple evaluation classes.")
    manifest_trial_labels = np.asarray([row.class_id for row in ordered_unseen], dtype=np.int64)
    immutable_trial_ids = np.asarray(
        [row.eeg_row_index for row in ordered_unseen], dtype=np.int64
    )
    stimulus_group_codes = np.asarray(
        [stimulus_to_class[row.stimulus_id] for row in ordered_unseen], dtype=np.int64
    )

    metric_rows: list[dict[str, object]] = []
    eeg_eda_rows: list[dict[str, object]] = []
    visual_eda_rows: list[dict[str, object]] = []
    averaging_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    half_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    bundle_rows: list[dict[str, object]] = []
    participant_reports = {}
    selected_dino_seen = load_visual_artifact(probe / selected_variant, "seen")
    template_root = Path(template_features)
    template_metadata = json.loads(
        (template_root / "metadata.json").read_text(encoding="utf-8")
    )
    template_report = json.loads(
        (template_root.parent / "template-image-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        template_metadata.get("complete") is not True
        or template_report.get("status") != "passed"
        or template_report.get("classes") != 200
        or template_report.get("eeg_image_overlap") != 0
        or template_metadata.get("template_manifest_sha256")
        != template_report.get("manifest_sha256")
    ):
        raise ValueError("NICE template provenance is incomplete or inconsistent.")
    cornet_template_artifact = np.load(template_root / "cornet-s.npz")
    dino_template_artifact = np.load(template_root / f"{selected_variant}.npz")
    for name, artifact in (
        ("cornet-s", cornet_template_artifact),
        ("dinov2", dino_template_artifact),
    ):
        if artifact["templates"].shape[0] != 200 or not np.isfinite(
            artifact["templates"]
        ).all():
            raise ValueError(f"Invalid NICE concept templates for {name}.")

    checkpoint_path = output / "participant-checkpoint.json"
    checkpoint_identity = {
        "random_seed": config.random_seed,
        "selected_dinov2_variant": selected_variant,
        "template_manifest_sha256": template_metadata["template_manifest_sha256"],
    }
    checkpoint_fields = {
        "metric_rows": metric_rows,
        "eeg_eda_rows": eeg_eda_rows,
        "visual_eda_rows": visual_eda_rows,
        "averaging_rows": averaging_rows,
        "curve_rows": curve_rows,
        "half_rows": half_rows,
        "control_rows": control_rows,
        "prediction_rows": prediction_rows,
        "provenance_rows": provenance_rows,
        "alignment_rows": alignment_rows,
        "bundle_rows": bundle_rows,
        "participant_reports": participant_reports,
    }
    if checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("identity") != checkpoint_identity:
            raise ValueError("Participant checkpoint belongs to a different locked NICE run.")
        for name in checkpoint_fields:
            checkpoint_fields[name].extend(saved[name]) if isinstance(
                checkpoint_fields[name], list
            ) else checkpoint_fields[name].update(saved[name])

    for participant_index, participant in enumerate(config.subjects):
        if participant in participant_reports:
            expected_bundles = [
                bundles / f"{participant}--{source}.joblib"
                for source in ("cornet-s", "dinov2")
            ]
            if not all(path.is_file() for path in expected_bundles):
                raise FileNotFoundError(
                    f"Checkpointed participant {participant} is missing a fitted bundle."
                )
            expected_hashes = {
                str(row["semantic_source"]): str(row["sha256"])
                for row in bundle_rows
                if row["participant"] == participant
            }
            if set(expected_hashes) != {"cornet-s", "dinov2"} or any(
                file_sha256(path) != expected_hashes[path.stem.split("--", 1)[1]]
                for path in expected_bundles
            ):
                raise ValueError(f"Checkpointed bundle hash changed for {participant}.")
            continue
        started = time.perf_counter()
        reference_cornet = load_feature_split(dataset, "seen", config, participant)
        brain = dataset / "brain_feature" / "17channels" / participant
        repeated_path = brain / "eeg_test_data.mat"
        unique_path = brain / "eeg_test_data_unique.mat"
        raw, archive_trial_labels = load_repeated_eeg(repeated_path, config)
        if not np.array_equal(archive_trial_labels, manifest_trial_labels):
            raise ValueError(f"MAT labels disagree with immutable manifest rows for {participant}.")
        averaged, truth = average_groups(
            raw, stimulus_group_codes, immutable_trial_ids, 80, config.random_seed
        )
        reordered, reordered_truth = average_groups(
            raw[::-1],
            stimulus_group_codes[::-1],
            immutable_trial_ids[::-1],
            80,
            config.random_seed,
        )
        # Full-count means are subset invariant; sorting and reversing cannot change them.
        if not np.array_equal(truth, reordered_truth) or not np.allclose(averaged, reordered):
            raise RuntimeError("The 80-repeat average changed after physical row reordering.")
        archive = np.asarray(loadmat(unique_path)["data"], dtype=np.float64)
        archive = archive[:, :, config.eeg_start_index : config.eeg_stop_index].reshape(200, -1)
        archive_labels = normalize_labels(loadmat(unique_path)["class_idx"])
        if not np.array_equal(truth, archive_labels):
            raise ValueError("Rebuilt and archive-average labels disagree.")
        cornet_candidates = cornet_template_artifact["templates"].astype(np.float64)
        cornet_labels = cornet_template_artifact["template_labels"].astype(np.int64)
        dino_candidates = dino_template_artifact["templates"].astype(np.float64)
        dino_labels = dino_template_artifact["template_labels"].astype(np.int64)
        reference_cornet = Modalities(
            reference_cornet.eeg,
            cornet_template_artifact["reference"].astype(np.float64),
            None,
            cornet_template_artifact["reference_labels"].astype(np.int64),
        )
        if not np.array_equal(truth, cornet_labels) or not np.array_equal(truth, dino_labels):
            raise ValueError("Candidate labels do not match averaged EEG queries.")

        external = {
            "cornet-s": Modalities(averaged, cornet_candidates, None, truth),
            "dinov2": Modalities(averaged, dino_candidates, None, truth),
        }
        reference = {
            "cornet-s": reference_cornet,
            "dinov2": Modalities(
                reference_cornet.eeg,
                selected_dino_seen.features,
                None,
                reference_cornet.labels,
            ),
        }
        rows, report, predictions = run_baseline_anchored_participant(
            participant, reference, external, config, full=True
        )
        metric_rows.extend(row.to_dict() for row in rows)
        eeg_eda_rows.extend(
            _eeg_eda(participant, reference_cornet, raw, stimulus_group_codes, averaged)
        )
        averaging_rows.extend(
            _averaging_eda(
                participant,
                raw,
                stimulus_group_codes,
                immutable_trial_ids,
                averaged,
                config.random_seed,
            )
        )
        for source, data in reference.items():
            visual_eda_rows.append(
                {
                    "participant": participant,
                    "semantic_source": source,
                    **embedding_eda(data.image, data.labels),
                }
            )

        for role, path in (
            ("reference_eeg", brain / "eeg_train_data_within.mat"),
            ("repeated_evaluation_eeg", repeated_path),
            ("archive_average_eeg", unique_path),
        ):
            provenance_rows.append(
                {
                    "participant": participant,
                    "role": role,
                    "path": path.as_posix(),
                    "sha256": file_sha256(path),
                }
            )
        alignment_rows.append(
            {
                "participant": participant,
                "reference_rows": len(reference_cornet.labels),
                "reference_classes": len(np.unique(reference_cornet.labels)),
                "evaluation_raw_rows": len(stimulus_group_codes),
                "evaluation_classes": len(truth),
                "repetitions_per_stimulus": 80,
                "averaged_width": averaged.shape[1],
                "archive_mean_abs_difference": float(np.mean(np.abs(archive - averaged))),
                "archive_max_abs_difference": float(np.max(np.abs(archive - averaged))),
                "averaged_sha256": _array_hash(averaged),
            }
        )

        for source_index, source in enumerate(("cornet-s", "dinov2")):
            source_selection = report["sources"][source]["selection"]
            spec = PreprocessingSpec(**source_selection["selected"]["spec"])
            alpha = float(source_selection["selected"]["alpha"])
            calibration = source_selection["calibration"]
            if source == "cornet-s":
                provenance = {
                    "source": "reextracted-cornet-s-reference-and-nice-concept-templates",
                    "template_manifest_sha256": template_metadata[
                        "template_manifest_sha256"
                    ],
                    "checkpoint": template_metadata["cornet_checkpoint"],
                }
            else:
                provenance = {
                    **selected_dino_seen.metadata,
                    "template_manifest_sha256": template_metadata[
                        "template_manifest_sha256"
                    ],
                }
            bundle = fit_decoder(
                participant,
                source,
                reference[source],
                CandidateBank(external[source].image, truth),
                spec,
                alpha,
                config.random_seed,
                retrieval=str(calibration["rule"]),
                neighbors=int(calibration["neighbors"]),
                configuration=config.to_dict(),
                feature_provenance=provenance,
            )
            bundle_path = bundles / f"{participant}--{source}.joblib"
            bundle_rows.append(save_bundle(bundle, bundle_path))
            before = file_sha256(bundle_path)
            query = normalize_rows(bundle.decoder.predict(bundle.transform.eeg(averaged)))
            cosine = query @ bundle.candidate_prototypes.T
            primary = ranking_metrics(cosine, truth, bundle.candidate_labels)
            if independent_ranks(cosine, truth, bundle.candidate_labels) != {
                key: primary[key]
                for key in ("top1_accuracy", "top5_accuracy", "mean_reciprocal_rank", "median_rank")
            }:
                raise RuntimeError("Independent rank implementation disagrees.")
            secondary_scores = calibrated_scores(cosine, bundle.retrieval, bundle.neighbors)
            secondary = ranking_metrics(secondary_scores, truth, bundle.candidate_labels)
            for retrieval_name, values in (
                ("cosine_primary", primary),
                ("selected_s4_transductive_secondary", secondary),
            ):
                for metric, score in values.items():
                    prediction_rows.append(
                        {
                            "participant": participant,
                            "semantic_source": source,
                            "retrieval": retrieval_name,
                            "metric": metric,
                            "score": score,
                        }
                    )
            predicted = bundle.candidate_labels[np.argmax(cosine, axis=1)]
            for query_index, (label, prediction) in enumerate(zip(truth, predicted, strict=True)):
                prediction_rows.append(
                    {
                        "participant": participant,
                        "semantic_source": source,
                        "retrieval": "cosine_prediction",
                        "metric": f"query_{query_index:03d}_truth_{label}",
                        "score": int(prediction),
                    }
                )

            for count in repetition_curve:
                scores = []
                for repeat in range(10 if count < 80 else 1):
                    sample, sample_truth = average_groups(
                        raw,
                        stimulus_group_codes,
                        immutable_trial_ids,
                        count,
                        10_000 + participant_index * 1000 + count * 10 + repeat,
                    )
                    sample_query = normalize_rows(
                        bundle.decoder.predict(bundle.transform.eeg(sample))
                    )
                    value = ranking_metrics(
                        sample_query @ bundle.candidate_prototypes.T,
                        sample_truth,
                        bundle.candidate_labels,
                    )["top1_accuracy"]
                    scores.append(value)
                curve_rows.append(
                    {
                        "participant": participant,
                        "semantic_source": source,
                        "repetitions": count,
                        "resamples": len(scores),
                        "top1_mean": float(np.mean(scores)),
                        "top1_sd": 0.0 if len(scores) == 1 else float(np.std(scores, ddof=1)),
                    }
                )
            groups = [
                raw[stimulus_group_codes == label][
                    np.argsort(immutable_trial_ids[stimulus_group_codes == label])
                ]
                for label in truth
            ]
            halves = (("first_40", slice(0, 40)), ("last_40", slice(40, 80)))
            for half_name, selection_slice in halves:
                half = np.stack(
                    [group[selection_slice].mean(axis=0) for group in groups]
                ).reshape(200, -1)
                half_query = normalize_rows(bundle.decoder.predict(bundle.transform.eeg(half)))
                half_rows.append(
                    {
                        "participant": participant,
                        "semantic_source": source,
                        "half": half_name,
                        "top1_accuracy": ranking_metrics(
                            half_query @ bundle.candidate_prototypes.T,
                            truth,
                            bundle.candidate_labels,
                        )["top1_accuracy"],
                    }
                )

            rng = np.random.default_rng(50_000 + participant_index * 10 + source_index)
            matrices = {
                "random_ranking": rng.normal(size=cosine.shape),
                "shuffled_eeg_rows": cosine[rng.permutation(200)],
                "permuted_candidate_rows": cosine[:, rng.permutation(200)],
                "permuted_labels": cosine,
                "permuted_stimulus_ids": cosine[rng.permutation(200)],
                "permuted_category_names": cosine,
            }
            for control, matrix in matrices.items():
                control_truth = (
                    rng.permutation(truth)
                    if control in {"permuted_labels", "permuted_category_names"}
                    else truth
                )
                control_rows.append(
                    {
                        "participant": participant,
                        "semantic_source": source,
                        "control": control,
                        "top1_accuracy": ranking_metrics(
                            matrix, control_truth, bundle.candidate_labels
                        )["top1_accuracy"],
                    }
                )
            for control_seed in CONTROL_SEEDS:
                shuffled = np.random.default_rng(control_seed + participant_index).permutation(
                    len(reference[source].labels)
                )
                from sklearn.linear_model import Ridge

                decoder = Ridge(alpha=alpha).fit(
                    bundle.transform.eeg(reference[source].eeg),
                    bundle.transform.semantics(reference[source].image[shuffled]),
                )
                broken_query = normalize_rows(decoder.predict(bundle.transform.eeg(averaged)))
                control_rows.append(
                    {
                        "participant": participant,
                        "semantic_source": source,
                        "control": f"shuffled_training_pairs_seed_{control_seed}",
                        "top1_accuracy": ranking_metrics(
                            broken_query @ bundle.candidate_prototypes.T,
                            truth,
                            bundle.candidate_labels,
                        )["top1_accuracy"],
                    }
                )
            if before != file_sha256(bundle_path):
                raise RuntimeError("Evaluation modified a fitted bundle.")
        report["runtime_seconds"] = time.perf_counter() - started
        participant_reports[participant] = report
        _write_json_atomic(
            checkpoint_path,
            {
                "identity": checkpoint_identity,
                **checkpoint_fields,
            },
        )

    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "eeg-eda.csv", eeg_eda_rows)
    _write_csv(output / "visual-eda.csv", visual_eda_rows)
    _write_csv(output / "averaging-eda.csv", averaging_rows)
    _write_csv(output / "repetition-curve.csv", curve_rows)
    _write_csv(output / "independent-halves.csv", half_rows)
    _write_csv(output / "negative-controls.csv", control_rows)
    _write_csv(output / "predictions-and-metrics.csv", prediction_rows)
    _write_csv(output / "provenance-ledger.csv", provenance_rows)
    _write_csv(output / "row-alignment.csv", alignment_rows)
    _write_csv(output / "bundle-manifest.csv", bundle_rows)
    _write_csv(
        output / "literature-protocol-comparison.csv",
        _literature_protocol(),
    )

    string_metrics = [{key: str(value) for key, value in row.items()} for row in metric_rows]
    _write_csv(output / "aggregate-metrics.csv", aggregate_metrics(string_metrics))
    selected_rows = [
        {
            "participant": participant,
            "semantic_source": source,
            "selected_spec": json.dumps(
                report["sources"][source]["selection"]["selected"]["spec"],
                sort_keys=True,
            ),
            "selected_alpha": report["sources"][source]["selection"]["selected"]["alpha"],
            "selected_retrieval": report["sources"][source]["selection"]["calibration"]["rule"],
            "selected_neighbors": report["sources"][source]["selection"]["calibration"][
                "neighbors"
            ],
        }
        for participant, report in participant_reports.items()
        for source in ("cornet-s", "dinov2")
    ]
    _write_csv(output / "selected-settings.csv", selected_rows)
    _write_csv(output / "s0-s4-candidate-ledger.csv", _candidate_ledger(participant_reports))
    (output / "summary.json").write_text(
        json.dumps(
            {
                "protocol": protocol,
                "config_hash": hashlib.sha256(
                    json.dumps(protocol, sort_keys=True).encode()
                ).hexdigest(),
                "selection": selection,
                "participants": participant_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    maximum_control = max(float(row["top1_accuracy"]) for row in control_rows)
    if maximum_control >= 0.05:
        raise RuntimeError(
            f"Negative control exceeded the locked 5% ceiling: {maximum_control:.3%}."
        )
    leakage = {
        "overall_status": "unresolved",
        "quarantined": True,
        "headline_use_permitted": False,
        "primary_claim": (
            "200-way NICE-style concept-template retrieval after averaging 80 known repetitions"
        ),
        "maximum_negative_control_top1": maximum_control,
        "boundaries": {
            "reference_only_selection": {"status": "passed"},
            "row_order_invariance": {"status": "passed"},
            "independent_cosine_metrics": {"status": "passed"},
            "upstream_bravl_preprocessing": {
                "status": "unresolved",
                "reason": "Raw archive construction is not reproduced locally.",
            },
            "ground_truth_repetition_grouping": {
                "status": "unresolved",
                "reason": "Known stimulus identity is required to form each 80-trial mean.",
            },
            "csls_secondary": {
                "status": "unresolved",
                "reason": "CSLS uses the complete unlabeled query batch and is secondary only.",
            },
        },
    }
    (output / "leakage-audit.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    _final_reports(output, metric_rows, curve_rows)
    return leakage

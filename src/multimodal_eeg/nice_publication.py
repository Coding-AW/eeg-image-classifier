"""Publish a compact, portable NICE-style research record from complete local runs."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PUBLIC_TABLES = (
    "participant-by-run-metrics.csv",
    "participant-averaged-metrics.csv",
    "five-run-aggregate-metrics.csv",
    "five-run-paired-comparisons.csv",
    "seed-level-negative-controls.csv",
    "seed-level-selected-settings.csv",
    "seed-level-repetition-curve.csv",
    "seed-level-independent-halves.csv",
    "strict-literature-comparison.csv",
    "context-only-literature-comparison.csv",
    "primary-source-protocol-ledger.csv",
)
SEED_EDA_TABLES = ("eeg-eda.csv", "visual-eda.csv", "averaging-eda.csv")
LOCKED_SEEDS = (17, 29, 43, 71, 101)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to publish an empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _portable_image_path(value: str) -> str:
    parts = Path(value).parts
    try:
        anchor = [item.casefold() for item in parts].index("object_images")
    except ValueError as error:
        raise ValueError(f"Template path is outside object_images: {value}") from error
    return Path(*parts[anchor:]).as_posix()


def _publish_template_manifest(source: Path, destination: Path) -> str:
    rows = _rows(source)
    portable = []
    for row in rows:
        selected = dict(row)
        selected["image_path"] = _portable_image_path(row["image_path"])
        portable.append(selected)
    _write_csv(destination, portable)
    if any(":" in row["image_path"] or row["image_path"].startswith("/") for row in portable):
        raise RuntimeError("Published template manifest contains an absolute path.")
    return _sha256(destination)


def _publish_seed_eda(comparison: Path, output: Path) -> None:
    """Combine all seed-level EDA rather than presenting a favourable single seed."""

    for name in SEED_EDA_TABLES:
        combined: list[dict[str, object]] = []
        for seed in LOCKED_SEEDS:
            source = comparison / "runs" / f"seed-{seed}" / name
            if not source.is_file():
                raise FileNotFoundError(f"Required seed EDA is missing: {source}")
            combined.extend({"seed": seed, **row} for row in _rows(source))
        _write_csv(output / f"seed-level-{name}", combined)


def _publish_selection_evidence(comparison: Path, output: Path) -> None:
    """Publish compact reference-only evidence behind every retained setting."""

    representation_rows: list[dict[str, object]] = []
    stage_rows: list[dict[str, object]] = []
    for seed in LOCKED_SEEDS:
        run = comparison / "runs" / f"seed-{seed}"
        representation = json.loads(
            (run / "dinov2-representation-selection.json").read_text(encoding="utf-8")
        )
        means = {row["variant"]: float(row["mean"]) for row in representation["records"]}
        selected = str(representation["selected_variant"])
        baseline = str(representation["baseline_variant"])
        representation_rows.append(
            {
                "seed": seed,
                "selection_data": representation["selection_split"],
                "selected_variant": selected,
                "selected_reference_mean": means[selected],
                "baseline_variant": baseline,
                "baseline_reference_mean": means[baseline],
                "reference_mean_change": means[selected] - means[baseline],
                "evaluation_features_loaded": representation["evaluation_features_loaded"],
            }
        )
        summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        for participant, report in sorted(summary["participants"].items()):
            for source, source_report in sorted(report["sources"].items()):
                selection = source_report["selection"]
                baseline_mean = float(np.mean(selection["baseline_fold_scores"]))
                preprocessing_mean = float(selection["selected_preprocessing"]["mean"])
                ridge_mean = float(selection["selected"]["mean"])
                stage_rows.append(
                    {
                        "seed": seed,
                        "participant": participant,
                        "semantic_source": source,
                        "s0_reference_mean": baseline_mean,
                        "s2_reference_mean": preprocessing_mean,
                        "s2_minus_s0": preprocessing_mean - baseline_mean,
                        "s2_retained": selection["retained_preprocessing"],
                        "s3_reference_mean": ridge_mean,
                        "s3_minus_s2": ridge_mean - preprocessing_mean,
                        "s3_retained": selection["retained_tuning"],
                    }
                )
    _write_csv(output / "dinov2-representation-selection.csv", representation_rows)
    _write_csv(output / "reference-fold-stage-evidence.csv", stage_rows)

    settings = _rows(comparison / "seed-level-selected-settings.csv")
    counts: Counter[tuple[str, str, str]] = Counter()
    totals: Counter[str] = Counter()
    for row in settings:
        source = row["semantic_source"]
        totals[source] += 1
        spec = json.loads(row["selected_spec"])
        for parameter in (
            "eeg_dimension",
            "semantic_dimension",
            "whiten",
            "remove_components",
        ):
            counts[(source, parameter, json.dumps(spec[parameter]))] += 1
        counts[(source, "ridge_alpha", row["selected_alpha"])] += 1
    frequency_rows = [
        {
            "semantic_source": source,
            "parameter": parameter,
            "selected_value": value,
            "models": count,
            "total_models": totals[source],
            "selection_fraction": count / totals[source],
            "selection_data": "category-disjoint reference folds only",
        }
        for (source, parameter, value), count in sorted(counts.items())
    ]
    _write_csv(output / "preprocessing-selection-frequency.csv", frequency_rows)


def _publish_control_summary(comparison: Path, output: Path) -> None:
    controls = _rows(comparison / "seed-level-negative-controls.csv")
    groups: dict[str, list[float]] = {}
    for row in controls:
        groups.setdefault(row["control"], []).append(float(row["top1_accuracy"]))
    rows = [
        {
            "control": name,
            "observations": len(values),
            "mean_top1": float(np.mean(values)),
            "minimum_top1": min(values),
            "maximum_top1": max(values),
            "analytical_chance": 0.005,
            "individual_fail_ceiling": 0.05,
            "passed": max(values) < 0.05,
        }
        for name, values in sorted(groups.items())
    ]
    _write_csv(output / "negative-control-summary.csv", rows)


def _validate_complete(comparison: Path) -> dict[str, object]:
    status = json.loads((comparison / "comparison-status.json").read_text(encoding="utf-8"))
    if status.get("seeds") != [17, 29, 43, 71, 101]:
        raise ValueError("All five locked NICE seeds are required for publication.")
    if float(status.get("maximum_control_top1", 1.0)) >= 0.05:
        raise ValueError("A negative control exceeded the locked 5% ceiling.")
    if float(status.get("mean_control_top1", 1.0)) >= 0.02:
        raise ValueError("Aggregate negative controls are not sufficiently close to chance.")
    participant = _rows(comparison / "participant-averaged-metrics.csv")
    keys = {(row["participant"], row["semantic_source"], row["metric"]) for row in participant}
    expected = {
        (f"sub-{number:02d}", source, metric)
        for number in range(1, 11)
        for source in ("cornet-s", "dinov2")
        for metric in ("top1_accuracy", "top5_accuracy")
    }
    if keys != expected or any(int(row["runs"]) != 5 for row in participant):
        raise ValueError("Participant-level five-run NICE results are incomplete.")
    return status


def _figures(comparison: Path, output: Path) -> None:
    rows = _rows(comparison / "participant-averaged-metrics.csv")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.hashsalt"] = "nice-style-eeg-study"
    participants = [f"sub-{number:02d}" for number in range(1, 11)]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(10)
    width = 0.38
    for offset, source in ((-width / 2, "cornet-s"), (width / 2, "dinov2")):
        lookup = {
            row["participant"]: 100 * float(row["mean_across_runs"])
            for row in rows
            if row["semantic_source"] == source and row["metric"] == "top1_accuracy"
        }
        axis.bar(x + offset, [lookup[item] for item in participants], width, label=source)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_xticks(x, participants, rotation=45)
    axis.set_ylabel("Top-1 accuracy (%)")
    axis.set_title("How consistent is top-1 performance across participants?")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output / "participant-performance.png",
        dpi=240,
        metadata={"Software": "eeg-image-classification"},
    )
    figure.savefig(
        output / "participant-performance.svg",
        dpi=240,
        metadata={"Date": None},
    )
    plt.close(figure)

    averaging = _rows(comparison / "runs" / "seed-17" / "averaging-eda.csv")
    # All five seeds use the same deterministic EEG subsets; one seed avoids duplicating EDA rows.
    counts = sorted({int(row["repetitions"]) for row in averaging})
    variance = [
        np.asarray(
            [
                float(row["mean_feature_variance"])
                for row in averaging
                if int(row["repetitions"]) == count
            ]
        )
        for count in counts
    ]
    similarity = [
        np.asarray(
            [
                float(row["mean_similarity_to_80"])
                for row in averaging
                if int(row["repetitions"]) == count
            ]
        )
        for count in counts
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].plot(counts, [values.mean() for values in variance], marker="o")
    axes[0].fill_between(
        counts,
        [values.mean() - values.std(ddof=1) for values in variance],
        [values.mean() + values.std(ddof=1) for values in variance],
        alpha=0.18,
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(counts, counts)
    axes[0].set_xlabel("EEG recordings averaged")
    axes[0].set_ylabel("Mean feature variance")
    axes[0].set_title("Does trial-to-trial noise fall?")
    axes[1].plot(counts, [values.mean() for values in similarity], marker="o", color="C2")
    axes[1].fill_between(
        counts,
        [values.mean() - values.std(ddof=1) for values in similarity],
        [values.mean() + values.std(ddof=1) for values in similarity],
        alpha=0.18,
        color="C2",
    )
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(counts, counts)
    axes[1].set_xlabel("EEG recordings averaged")
    axes[1].set_ylabel("Cosine similarity to 80-repeat average")
    axes[1].set_title("Does the response estimate stabilize?")
    figure.suptitle("What does EEG averaging change?")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"eeg-averaging-eda.{suffix}",
            dpi=240,
            metadata={"Software": "eeg-image-classification"}
            if suffix == "png"
            else {"Date": None},
        )
    plt.close(figure)

    visual = _rows(comparison / "runs" / "seed-17" / "visual-eda.csv")
    geometry = (
        ("effective_rank", "Effective rank", "Used feature directions"),
        ("common_direction_energy", "Shared-direction energy", "Common structure"),
        ("standardized_separation", "Standardized separation", "Class separation"),
        ("nearest_prototype_accuracy", "Nearest-prototype accuracy", "Reference retrieval"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, (field, label, title) in zip(axes.flat, geometry, strict=True):
        values = [
            np.mean(
                [
                    float(row[field])
                    for row in visual
                    if row["semantic_source"] == source
                ]
            )
            for source in ("cornet-s", "dinov2")
        ]
        axis.bar(["CORnet-S", "DINOv2"], values, color=("C0", "C1"))
        axis.set_ylabel(label)
        axis.set_title(title)
    figure.suptitle("How do the two visual target spaces differ?")
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"visual-space-eda.{suffix}",
            dpi=240,
            metadata={"Software": "eeg-image-classification"}
            if suffix == "png"
            else {"Date": None},
        )
    plt.close(figure)

    curve = _rows(comparison / "seed-level-repetition-curve.csv")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for source in ("cornet-s", "dinov2"):
        counts = sorted({int(row["repetitions"]) for row in curve})
        participant_values = []
        for count in counts:
            values = []
            for participant in participants:
                selected = [
                    float(row["top1_mean"])
                    for row in curve
                    if row["semantic_source"] == source
                    and row["participant"] == participant
                    and int(row["repetitions"]) == count
                ]
                values.append(100 * float(np.mean(selected)))
            participant_values.append(np.asarray(values))
        means = [float(values.mean()) for values in participant_values]
        standard_deviations = [float(values.std(ddof=1)) for values in participant_values]
        axis.plot(counts, means, marker="o", label=source)
        axis.fill_between(
            counts,
            np.asarray(means) - standard_deviations,
            np.asarray(means) + standard_deviations,
            alpha=0.15,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 5, 10, 20, 40, 80], [1, 2, 5, 10, 20, 40, 80])
    axis.set_xlabel("EEG repetitions averaged")
    axis.set_ylabel("Mean top-1 accuracy (%)")
    axis.set_title("How does averaging more recordings change performance?")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"repetition-curve.{suffix}",
            dpi=240,
            metadata={"Software": "eeg-image-classification"}
            if suffix == "png"
            else {"Date": None},
        )
    plt.close(figure)

    controls = _rows(comparison / "seed-level-negative-controls.csv")
    control_groups = {
        "random ranking": ("random_ranking",),
        "shuffled EEG": ("shuffled_eeg_rows", "permuted_stimulus_ids"),
        "permuted gallery": ("permuted_candidate_rows",),
        "permuted labels": ("permuted_labels", "permuted_category_names"),
        "shuffled training": tuple(
            f"shuffled_training_pairs_seed_{seed}" for seed in (101, 211, 307)
        ),
    }
    values = [
        100
        * np.mean(
            [float(row["top1_accuracy"]) for row in controls if row["control"] in names]
        )
        for names in control_groups.values()
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.bar(np.arange(len(values)), values, color="#5676a5")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.axhline(5.0, color="#b23a48", linestyle=":", linewidth=1.5, label="fail ceiling")
    axis.set_xticks(np.arange(len(values)), control_groups, rotation=25, ha="right")
    axis.set_ylabel("Mean top-1 accuracy (%)")
    axis.set_title("Do predictions survive when valid information is broken?")
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"leakage-controls.{suffix}",
            dpi=240,
            metadata={"Software": "eeg-image-classification"}
            if suffix == "png"
            else {"Date": None},
        )
    plt.close(figure)

    strict = _rows(comparison / "strict-literature-comparison.csv")
    context = _rows(comparison / "context-only-literature-comparison.csv")
    direct_labels = ["This study\nCORnet-S", "This study\nDINOv2", "NICE-GA", "Chance"]
    direct_top1 = [float(row["reported_top1_percent"]) for row in strict] + [0.5]
    direct_top5 = [float(row["reported_top5_percent"]) for row in strict] + [2.5]
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), width_ratios=(4, 1.35))
    direct_axis, context_axis = axes
    x = np.arange(len(direct_labels))
    direct_axis.bar(x - 0.19, direct_top1, 0.38, label="top-1")
    direct_axis.bar(x + 0.19, direct_top5, 0.38, label="top-5")
    direct_axis.set_xticks(x, direct_labels)
    direct_axis.set_ylabel("Accuracy (%)")
    direct_axis.set_title("Evaluation-aligned comparison")
    direct_axis.legend(frameon=False)
    context_labels = ["ATM\n(context)"]
    context_axis.bar(
        [-0.19], [float(context[0]["reported_top1_percent"])], 0.38, color="C0"
    )
    context_axis.bar(
        [0.19], [float(context[0]["reported_top5_percent"])], 0.38, color="C1"
    )
    context_axis.set_xticks([0], context_labels)
    context_axis.set_title("Protocol differs")
    maximum = max(
        [*direct_top1, *direct_top5, float(context[0]["reported_top1_percent"]),
         float(context[0]["reported_top5_percent"])]
    )
    direct_axis.set_ylim(0, maximum + 4)
    context_axis.set_ylim(0, maximum + 4)
    context_axis.tick_params(axis="y", labelleft=False)
    figure.tight_layout()
    for suffix in ("png", "svg"):
        figure.savefig(
            output / f"literature-comparison.{suffix}",
            dpi=240,
            metadata={"Software": "eeg-image-classification"}
            if suffix == "png"
            else {"Date": None},
        )
    plt.close(figure)


def publish_nice_study(
    comparison: Path,
    template_manifest: Path,
    template_report: Path,
    template_audit: Path,
    alignment_manifest: Path,
    output: Path,
    figures: Path,
) -> dict[str, object]:
    """Publish only a complete five-seed comparison and portable provenance."""

    status = _validate_complete(comparison)
    output.mkdir(parents=True, exist_ok=True)
    for name in PUBLIC_TABLES:
        source = comparison / name
        if not source.is_file():
            raise FileNotFoundError(f"Required NICE output is missing: {name}")
        shutil.copyfile(source, output / name)
    _publish_seed_eda(comparison, output)
    _publish_selection_evidence(comparison, output)
    _publish_control_summary(comparison, output)
    portable_hash = _publish_template_manifest(
        template_manifest, output / "template-image-manifest.csv"
    )
    alignment_rows = _rows(alignment_manifest)
    if len(alignment_rows) != 32_540:
        raise ValueError("Alignment manifest must contain 16,540 reference and 16,000 test rows.")
    if any(Path(row["image_path"]).is_absolute() for row in alignment_rows):
        raise ValueError("Alignment manifest contains an absolute image path.")
    _write_csv(output / "stimulus-alignment-manifest.csv", alignment_rows)
    alignment_hash = _sha256(output / "stimulus-alignment-manifest.csv")
    gallery = json.loads(template_report.read_text(encoding="utf-8"))
    if gallery.get("status") != "passed" or gallery.get("eeg_image_overlap") != 0:
        raise ValueError("Template leakage report did not pass.")
    independent_gallery_audit = json.loads(template_audit.read_text(encoding="utf-8"))
    if (
        independent_gallery_audit.get("status") != "passed"
        or independent_gallery_audit.get("filename_overlap") != 0
        or independent_gallery_audit.get("content_hash_overlap") != 0
        or independent_gallery_audit.get("template_manifest_sha256")
        != gallery.get("manifest_sha256")
    ):
        raise ValueError("Independent template leakage audit did not pass.")
    provenance = {
        "claim": "200-way NICE-style concept-template retrieval after 80-repeat averaging",
        "qualified": True,
        "qualification": "Upstream BraVL preprocessing provenance remains unresolved.",
        "comparison_status": status,
        "template_report": gallery,
        "independent_template_audit": independent_gallery_audit,
        "portable_template_manifest_sha256": portable_hash,
        "stimulus_alignment_manifest_sha256": alignment_hash,
        "published_files": {},
        "published_figures": {},
    }
    _figures(comparison, figures)
    for path in sorted(output.glob("*")):
        if path.is_file() and path.name not in {"provenance.json", "README.md"}:
            provenance["published_files"][path.name] = _sha256(path)
    for path in sorted(figures.glob("*")):
        if path.is_file():
            provenance["published_figures"][path.name] = _sha256(path)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    return provenance

"""Five-run, fail-closed averaged-EEG literature comparison."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .averaged_study import _write_csv, run_averaged_study
from .classical_study import holm_adjust, paired_change_statistics

LOCKED_SEEDS = (17, 29, 43, 71, 101)
PRIMARY_METRICS = ("top1_accuracy", "top5_accuracy")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nice_literature_ledger() -> list[dict[str, object]]:
    return [
        {
            "study": "Current NICE-style five-run study",
            "comparison_group": "direct NICE-style concept-template retrieval",
            "paper_url": "local protocol",
            "code_url": "local implementation",
            "dataset": "THINGS-EEG2 via BraVL",
            "participants": 10,
            "participant_dependent": "yes",
            "training_unit": "archive-provided four-repeat image mean",
            "test_unit": "mean of 80 repeated trials",
            "queries": 200,
            "gallery": "200 centroids of non-EEG THINGS images",
            "metric": "200-way cosine top-1 and top-5",
            "runs": 5,
            "eeg_channels": 17,
            "eeg_interval": "distributed-array samples 27-59 inclusive",
            "visual_encoder": "set per result row",
            "decoder": "linear Ridge regression",
            "participant_aggregation": (
                "five-run mean within participant, then mean across 10 participants"
            ),
            "reported_top1_percent": "computed after all five runs",
            "reported_top5_percent": "computed after all five runs",
            "eligibility": "direct",
            "reason": "Evaluation structure matches NICE; remaining model differences disclosed.",
            "evidence_location": "configuration, alignment report, and leakage audit",
        },
        {
            "study": "NICE-GA headline (Song et al., ICLR 2024)",
            "comparison_group": "direct NICE-style concept-template retrieval",
            "paper_url": "https://arxiv.org/abs/2308.13234",
            "code_url": "https://github.com/eeyhsong/NICE-EEG",
            "dataset": "THINGS-EEG2",
            "participants": 10,
            "participant_dependent": "yes",
            "training_unit": "mean of four repetitions per training condition",
            "test_unit": "mean of 80 repeated trials",
            "queries": 200,
            "gallery": "200 centres of THINGS images excluding EEG test-session images",
            "metric": "200-way retrieval",
            "runs": 5,
            "eeg_channels": 63,
            "eeg_interval": "0-1000 ms epoch as reported",
            "visual_encoder": "CLIP ViT-L/14",
            "decoder": "contrastive EEG encoder with graph attention",
            "participant_aggregation": "five-run result aggregated across 10 participants",
            "reported_top1_percent": 15.6,
            "reported_top5_percent": 42.8,
            "eligibility": "direct",
            "reason": "Target protocol: four/80 averaging and non-EEG concept templates.",
            "evidence_location": (
                "Song et al. paper Sections 3.1 and 4.1 and Table 1; official NICE-EEG "
                "README, EEG pre-processing step 2 and image-feature step 3; "
                "nice_stand.py evaluation loop"
            ),
            "visual_space": "CLIP ViT-L/14",
        },
        {
            "study": "ATM (Li et al., NeurIPS 2024)",
            "comparison_group": "context: ATM paper benchmark",
            "paper_url": "https://arxiv.org/abs/2403.07721",
            "code_url": "not established",
            "dataset": "THINGS-EEG2",
            "participants": 10,
            "participant_dependent": "yes",
            "training_unit": "four repetitions retained as separate trials",
            "test_unit": "mean of 80 repeated trials",
            "queries": 200,
            "gallery": "200 image embeddings",
            "metric": "200-way top-1 and top-5",
            "runs": "highest test accuracy during training",
            "eeg_channels": 63,
            "eeg_interval": "0-1000 ms epoch as reported",
            "visual_encoder": "CLIP ViT-L/14",
            "decoder": "attention-based temporal-spatial EEG encoder",
            "participant_aggregation": "reported across 10 participant-specific models",
            "reported_top1_percent": 28.64,
            "reported_top5_percent": 58.47,
            "eligibility": "context-only",
            "reason": (
                "Training unit and checkpoint-selection rule differ from the strict protocol."
            ),
            "evidence_location": (
                "Li et al. paper EEG-image retrieval section, Table 3, and Appendix B; "
                "the paper reports the highest test accuracy during training"
            ),
        },
    ]
def aggregate_aligned_runs(
    run_paths: dict[int, Path], output: Path
) -> dict[str, object]:
    """Aggregate run results only after every locked seed is complete."""

    if tuple(sorted(run_paths)) != tuple(sorted(LOCKED_SEEDS)):
        raise ValueError(f"Exactly the locked seeds {LOCKED_SEEDS} are required.")
    output.mkdir(parents=True, exist_ok=True)
    participant_run = []
    controls = []
    settings = []
    repetition_rows = []
    half_rows = []
    for seed in LOCKED_SEEDS:
        run = run_paths[seed]
        required = ("metrics.csv", "negative-controls.csv", "selected-settings.csv")
        if not all((run / name).is_file() for name in required):
            raise FileNotFoundError(f"Seed {seed} is incomplete: {run}")
        for row in _read_csv(run / "metrics.csv"):
            if row["stage"] == "S3" and row["metric"] in PRIMARY_METRICS:
                participant_run.append({"seed": seed, **row})
        controls.extend({"seed": seed, **row} for row in _read_csv(run / "negative-controls.csv"))
        settings.extend({"seed": seed, **row} for row in _read_csv(run / "selected-settings.csv"))
        repetition_rows.extend(
            {"seed": seed, **row} for row in _read_csv(run / "repetition-curve.csv")
        )
        half_rows.extend(
            {"seed": seed, **row} for row in _read_csv(run / "independent-halves.csv")
        )

    _write_csv(output / "participant-by-run-metrics.csv", participant_run)
    _write_csv(output / "seed-level-negative-controls.csv", controls)
    _write_csv(output / "seed-level-selected-settings.csv", settings)
    _write_csv(output / "seed-level-repetition-curve.csv", repetition_rows)
    _write_csv(output / "seed-level-independent-halves.csv", half_rows)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in participant_run:
        grouped[(row["participant"], row["semantic_source"], row["metric"])].append(
            float(row["score"])
        )
    participant_means = [
        {
            "participant": participant,
            "semantic_source": source,
            "metric": metric,
            "runs": len(values),
            "mean_across_runs": float(np.mean(values)),
            "run_sd": float(np.std(values, ddof=1)),
        }
        for (participant, source, metric), values in sorted(grouped.items())
    ]
    if any(row["runs"] != 5 for row in participant_means):
        raise RuntimeError("Every participant/source/metric must contain five runs.")
    _write_csv(output / "participant-averaged-metrics.csv", participant_means)

    aggregates = []
    for source in ("cornet-s", "dinov2"):
        for metric in PRIMARY_METRICS:
            values = np.asarray(
                [
                    row["mean_across_runs"]
                    for row in participant_means
                    if row["semantic_source"] == source and row["metric"] == metric
                ],
                dtype=np.float64,
            )
            aggregates.append(
                {
                    "semantic_source": source,
                    "metric": metric,
                    "participants": len(values),
                    "mean": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    # Two-sided 95% Student-t interval with 9 degrees of freedom.
                    "ci_low": float(values.mean() - 2.262157 * values.std(ddof=1) / np.sqrt(10)),
                    "ci_high": float(values.mean() + 2.262157 * values.std(ddof=1) / np.sqrt(10)),
                }
            )
    _write_csv(output / "five-run-aggregate-metrics.csv", aggregates)

    comparisons = []
    for metric in PRIMARY_METRICS:
        values = {
            source: {
                row["participant"]: row["mean_across_runs"]
                for row in participant_means
                if row["semantic_source"] == source and row["metric"] == metric
            }
            for source in ("cornet-s", "dinov2")
        }
        participants = sorted(set(values["cornet-s"]) & set(values["dinov2"]))
        difference = np.asarray(
            [values["dinov2"][item] - values["cornet-s"][item] for item in participants]
        )
        comparisons.append(
            {
                "comparison": "dinov2_minus_cornet-s",
                "metric": metric,
                "participants": len(participants),
                **paired_change_statistics(difference),
            }
        )
    adjusted = holm_adjust([float(row["sign_flip_p"]) for row in comparisons])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_p"] = value
    _write_csv(output / "five-run-paired-comparisons.csv", comparisons)

    ledger = _nice_literature_ledger()
    current = ledger[0]
    current_rows = []
    for source in ("cornet-s", "dinov2"):
        values = {
            row["metric"]: 100 * float(row["mean"])
            for row in aggregates
            if row["semantic_source"] == source
        }
        current_rows.append(
            {
                **current,
                "study": f"Current NICE-style study ({source})",
                "visual_space": source,
                "visual_encoder": source,
                "reported_top1_percent": values["top1_accuracy"],
                "reported_top5_percent": values["top5_accuracy"],
            }
        )
    ledger = [*current_rows, *ledger[1:]]
    _write_csv(output / "primary-source-protocol-ledger.csv", ledger)
    _write_csv(
        output / "strict-literature-comparison.csv",
        [row for row in ledger if row["eligibility"] in {"strict", "direct"}],
    )
    _write_csv(
        output / "context-only-literature-comparison.csv",
        [row for row in ledger if row["eligibility"] == "context-only"],
    )
    maximum_control = max(float(row["top1_accuracy"]) for row in controls)
    mean_control = float(np.mean([float(row["top1_accuracy"]) for row in controls]))
    controls_pass = maximum_control < 0.05 and mean_control < 0.02
    summary = {
        "status": "unresolved" if controls_pass else "failed",
        "quarantined": True,
        "seeds": list(LOCKED_SEEDS),
        "maximum_control_top1": maximum_control,
        "mean_control_top1": mean_control,
        "control_acceptance": "maximum <5% and aggregate mean <2%",
        "strict_external_comparators": 1,
        "reason": "NICE is the direct evaluation target; ATM remains context-only.",
    }
    (output / "comparison-status.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison_text = (
        "NICE is the direct evaluation target: both studies average four training "
        "repetitions, average 80 test repetitions, and rank 200 non-EEG concept templates. "
        "ATM remains context-only because it retains training repetitions separately and "
        "selects the highest test accuracy during training.\n\n"
    )
    report = (
        "# Strict averaged-EEG literature comparison\n\n"
        "Each test query is the arithmetic mean of 80 EEG recordings known to come from "
        "one test stimulus. Each participant therefore supplies 200 queries, and each "
        "query is ranked against 200 non-EEG concept templates using cosine. "
        "Settings are selected from reference categories only.\n\n"
        "The experiment is repeated with seeds 17, 29, 43, 71 and 101. Runs are averaged "
        "within each participant before the ten participant means are summarized. Images "
        "and runs are not treated as extra participants.\n\n"
        + comparison_text
        + "The result remains quarantined because upstream BraVL preprocessing cannot be "
        "reconstructed locally.\n"
    )
    (output / "plain-language-alignment-report.md").write_text(report, encoding="utf-8")
    return summary


def run_aligned_comparison(
    dataset: Path,
    alignment_manifest: Path,
    dinov2_probe: Path,
    config: Path,
    output: Path,
    bundles: Path,
    template_features: Path,
) -> dict[str, object]:
    """Resume the locked five runs and aggregate only when all are complete."""

    output.mkdir(parents=True, exist_ok=True)
    run_paths = {}
    for seed in LOCKED_SEEDS:
        run_output = output / "runs" / f"seed-{seed}"
        run_bundles = bundles / f"seed-{seed}"
        run_paths[seed] = run_output
        if not (run_output / "leakage-audit.json").is_file():
            status = {
                "status": "running",
                "active_seed": seed,
                "completed_seeds": [
                    item
                    for item, path in run_paths.items()
                    if (path / "leakage-audit.json").is_file()
                ],
                "locked_seeds": list(LOCKED_SEEDS),
            }
            (output / "run-status.json").write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            run_averaged_study(
                dataset,
                alignment_manifest,
                dinov2_probe,
                config,
                run_output,
                run_bundles,
                averaging_rule="arithmetic-within-stimulus",
                primary_repetitions=80,
                repetition_curve=(1, 2, 5, 10, 20, 40, 80),
                primary_retrieval="cosine",
                random_seed=seed,
                template_features=template_features,
            )
    result = aggregate_aligned_runs(run_paths, output)
    (output / "run-status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "active_seed": None,
                "completed_seeds": list(LOCKED_SEEDS),
                "locked_seeds": list(LOCKED_SEEDS),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return result

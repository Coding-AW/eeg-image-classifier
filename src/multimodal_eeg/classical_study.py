"""Baseline-anchored classical zero-shot study for frozen visual embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .array_utils import normalize_rows
from .config import ExperimentConfig
from .data import Modalities

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreprocessingSpec:
    eeg_dimension: int | None = None
    semantic_dimension: int | None = None
    whiten: bool = False
    remove_components: int = 0
    center_semantics: bool = False

    @property
    def name(self) -> str:
        return (
            f"eeg-{self.eeg_dimension or 'raw'}_sem-{self.semantic_dimension or 'raw'}_"
            f"white-{int(self.whiten)}_remove-{self.remove_components}_"
            f"center-{int(self.center_semantics)}"
        )


@dataclass(frozen=True)
class ClassicalMetric:
    participant: str
    stage: str
    predecessor: str
    baseline_id: str
    semantic_source: str
    preprocessing: str
    ridge_alpha: float
    retrieval: str
    candidate_count: int
    metric: str
    score: float
    change_from_predecessor: float
    change_from_baseline: float
    configuration_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MatchedPreprocessor:
    """Fold-fitted EEG and semantic PCA with auditable provenance."""

    def __init__(self, spec: PreprocessingSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed

    def fit(self, eeg: FloatArray, semantics: FloatArray, labels: IntArray):
        self.eeg_scaler_ = StandardScaler().fit(eeg)
        scaled_eeg = self.eeg_scaler_.transform(eeg)
        self.eeg_pca_ = self._pca(self.spec.eeg_dimension, False, scaled_eeg)
        self.semantic_mean_ = semantics.mean(axis=0)
        centered = semantics - self.semantic_mean_ if self.spec.center_semantics else semantics
        self.removal_pca_ = self._pca(self.spec.remove_components or None, False, centered)
        residual = self._remove(centered)
        self.semantic_pca_ = self._pca(self.spec.semantic_dimension, self.spec.whiten, residual)
        classes = np.unique(labels).astype(np.int64)
        self.provenance_ = {
            "fit_split": "reference_training_classes",
            "fit_sample_count": len(labels),
            "fit_class_hash": hashlib.sha256(classes.tobytes()).hexdigest()[:16],
            "seed": self.seed,
            "spec": asdict(self.spec),
        }
        return self

    def _pca(self, dimension: int | None, whiten: bool, values: FloatArray):
        if dimension is None:
            return None
        if dimension > min(values.shape):
            raise ValueError("PCA dimension exceeds the fitted matrix rank bound.")
        return PCA(n_components=dimension, whiten=whiten, random_state=self.seed).fit(values)

    def _remove(self, centered: FloatArray) -> FloatArray:
        if self.removal_pca_ is None:
            return centered
        return centered - self.removal_pca_.inverse_transform(
            self.removal_pca_.transform(centered)
        )

    def eeg(self, values: FloatArray) -> FloatArray:
        output = self.eeg_scaler_.transform(values)
        return output if self.eeg_pca_ is None else self.eeg_pca_.transform(output)

    def semantics(self, values: FloatArray) -> FloatArray:
        output = values - self.semantic_mean_ if self.spec.center_semantics else values
        output = self._remove(output)
        if self.semantic_pca_ is not None:
            output = self.semantic_pca_.transform(output)
        return normalize_rows(output)


def prototype_bank(features: FloatArray, labels: IntArray) -> tuple[IntArray, FloatArray]:
    """Return one equally weighted normalized mean per class."""

    classes = np.unique(labels).astype(np.int64)
    prototypes = np.stack(
        [normalize_rows(features[labels == label]).mean(axis=0) for label in classes]
    )
    return classes, normalize_rows(prototypes)


def calibrated_scores(scores: FloatArray, rule: str, neighbors: int = 10) -> FloatArray:
    """Apply an explicitly selected candidate-ranking rule."""

    if rule == "cosine":
        return scores
    if rule == "csls":
        k = min(neighbors, scores.shape[1] - 1)
        if k < 1:
            raise ValueError("CSLS requires at least two candidates.")
        query_density = np.partition(scores, -k, axis=1)[:, -k:].mean(axis=1, keepdims=True)
        # Candidate density is estimated from the query set; no query labels are used.
        candidate_density = np.partition(scores, -k, axis=0)[-k:].mean(axis=0, keepdims=True)
        return 2 * scores - query_density - candidate_density
    raise ValueError("retrieval rule must be cosine or csls.")


def ranking_metrics(scores: FloatArray, truth: IntArray, candidates: IntArray) -> dict[str, float]:
    """Metrics for the balanced forced-choice ranking task."""

    order = np.argsort(-scores, axis=1)
    ranked = candidates[order]
    matches = ranked == truth[:, None]
    if not np.all(matches.any(axis=1)):
        raise ValueError("Every query label must appear exactly once in the candidate bank.")
    ranks = np.argmax(matches, axis=1) + 1
    predictions = ranked[:, 0]
    candidate_index = {int(label): index for index, label in enumerate(candidates)}
    prediction_indices = np.asarray([candidate_index[int(label)] for label in predictions])
    counts = np.bincount(prediction_indices, minlength=len(candidates))
    return {
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "top5_accuracy": float(np.mean(ranks <= min(5, len(candidates)))),
        "macro_f1": float(f1_score(truth, predictions, labels=candidates, average="macro")),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "median_rank": float(np.median(ranks)),
        "prediction_frequency_skew": float(
            np.mean(((counts - counts.mean()) / max(counts.std(), 1e-12)) ** 3)
        ),
    }


def fit_and_score(
    train: Modalities,
    test: Modalities,
    spec: PreprocessingSpec,
    alpha: float,
    seed: int,
    retrieval: str = "cosine",
    neighbors: int = 10,
) -> tuple[dict[str, float], dict[str, object], IntArray]:
    """Fit the complete classical decoder without query-side image input."""

    transform = MatchedPreprocessor(spec, seed).fit(train.eeg, train.image, train.labels)
    train_eeg = transform.eeg(train.eeg)
    train_semantics = transform.semantics(train.image)
    decoder = Ridge(alpha=alpha).fit(train_eeg, train_semantics)
    query = normalize_rows(decoder.predict(transform.eeg(test.eeg)))
    candidates, raw_prototypes = prototype_bank(test.image, test.labels)
    prototypes = transform.semantics(raw_prototypes)
    scores = calibrated_scores(query @ prototypes.T, retrieval, neighbors)
    metrics = ranking_metrics(scores, test.labels, candidates)
    predictions = candidates[np.argmax(scores, axis=1)]
    provenance = {
        "transform": transform.provenance_,
        "decoder": {"fit_split": "reference", "alpha": alpha, "samples": len(train.labels)},
        "query_interface": ["eeg", "candidate_prototype_bank"],
    }
    return metrics, provenance, predictions


def class_disjoint_folds(labels: IntArray, folds: int, seed: int) -> list[IntArray]:
    """Deterministic folds whose class sets cannot overlap."""

    classes = np.unique(labels)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return [classes[validation] for _, validation in splitter.split(classes)]


def reference_scores(
    reference: Modalities,
    spec: PreprocessingSpec,
    alpha: float,
    folds: int,
    seed: int,
    retrieval: str = "cosine",
    neighbors: int = 10,
) -> list[float]:
    """Simulate ZSL using classes absent from each fitted fold."""

    values = []
    for held_classes in class_disjoint_folds(reference.labels, folds, seed):
        held = np.isin(reference.labels, held_classes)
        train = reference.take(np.flatnonzero(~held))
        validation = reference.take(np.flatnonzero(held))
        metrics, _, _ = fit_and_score(
            train, validation, spec, alpha, seed, retrieval, neighbors
        )
        values.append(metrics["top1_accuracy"])
    return values


def select_pipeline(reference: Modalities, config: ExperimentConfig) -> dict[str, object]:
    """Select S2/S3/S4 settings entirely through reference-class validation."""

    baseline_spec = PreprocessingSpec()
    baseline_folds = reference_scores(
        reference,
        baseline_spec,
        config.classical_baseline_alpha,
        config.tuning_folds,
        config.random_seed,
    )
    records = []
    dimensions = [
        value
        for value in config.classical_semantic_dimensions
        if value <= reference.image.shape[1]
    ]
    eeg_dimensions = [
        value
        for value in config.classical_eeg_dimensions
        if value is None or value <= reference.eeg.shape[1]
    ]
    for eeg_dimension, semantic_dimension, whiten, removed in product(
        eeg_dimensions,
        dimensions,
        config.classical_whitening,
        config.classical_remove_components,
    ):
        if removed + semantic_dimension > min(reference.image.shape):
            continue
        spec = PreprocessingSpec(eeg_dimension, semantic_dimension, whiten, removed, True)
        folds = reference_scores(
            reference,
            spec,
            config.classical_baseline_alpha,
            config.tuning_folds,
            config.random_seed,
        )
        records.append(
            {
                "spec": asdict(spec),
                "alpha": config.classical_baseline_alpha,
                "fold_scores": folds,
                "mean": float(np.mean(folds)),
                "beats_baseline_every_fold": bool(np.all(np.asarray(folds) > baseline_folds)),
            }
        )
    eligible = [row for row in records if row["beats_baseline_every_fold"]]
    if eligible:
        selected_s2 = max(eligible, key=lambda row: row["mean"])
        retained_preprocessing = True
    else:
        selected_s2 = {
            "spec": asdict(baseline_spec),
            "alpha": config.classical_baseline_alpha,
            "fold_scores": baseline_folds,
            "mean": float(np.mean(baseline_folds)),
            "beats_baseline_every_fold": False,
        }
        retained_preprocessing = False

    tuning_records = []
    for alpha in config.ridge_alpha_candidates:
        folds = reference_scores(
            reference,
            PreprocessingSpec(**selected_s2["spec"]),
            alpha,
            config.tuning_folds,
            config.random_seed,
        )
        tuning_records.append(
            {
                "alpha": alpha,
                "fold_scores": folds,
                "mean": float(np.mean(folds)),
                "beats_s2_every_fold": bool(
                    np.all(np.asarray(folds) > np.asarray(selected_s2["fold_scores"]))
                ),
            }
        )
    eligible_tuning = [row for row in tuning_records if row["beats_s2_every_fold"]]
    if eligible_tuning:
        selected_alpha = max(eligible_tuning, key=lambda row: row["mean"])
        retained_tuning = True
    else:
        selected_alpha = {
            "alpha": config.classical_baseline_alpha,
            "fold_scores": selected_s2["fold_scores"],
            "mean": selected_s2["mean"],
            "beats_s2_every_fold": False,
        }
        retained_tuning = False
    selected = {
        "spec": selected_s2["spec"],
        "alpha": selected_alpha["alpha"],
        "fold_scores": selected_alpha["fold_scores"],
        "mean": selected_alpha["mean"],
    }

    calibration = [{"rule": "cosine", "neighbors": 10}]
    for rule, neighbors in [
        ("csls", value) for value in config.classical_csls_neighbors
    ]:
        folds = reference_scores(
            reference,
            PreprocessingSpec(**selected["spec"]),
            float(selected["alpha"]),
            config.tuning_folds,
            config.random_seed,
            rule,
            neighbors,
        )
        calibration.append(
            {
                "rule": rule,
                "neighbors": neighbors,
                "fold_scores": folds,
                "mean": float(np.mean(folds)),
            }
        )
    cosine_scores = reference_scores(
        reference,
        PreprocessingSpec(**selected["spec"]),
        float(selected["alpha"]),
        config.tuning_folds,
        config.random_seed,
    )
    for item in calibration:
        item.setdefault("fold_scores", cosine_scores)
        item.setdefault("mean", float(np.mean(cosine_scores)))
        item["beats_cosine_every_fold"] = bool(
            item["rule"] == "cosine"
            or np.all(np.asarray(item["fold_scores"]) > np.asarray(cosine_scores))
        )
    retained = [
        item
        for item in calibration
        if item["rule"] != "cosine" and item["beats_cosine_every_fold"]
    ]
    chosen_calibration = (
        max(retained, key=lambda item: item["mean"]) if retained else calibration[0]
    )
    return {
        "baseline_fold_scores": baseline_folds,
        "preprocessing_records": records,
        "selected_preprocessing": selected_s2,
        "retained_preprocessing": retained_preprocessing,
        "tuning_records": tuning_records,
        "retained_tuning": retained_tuning,
        "selected": selected,
        "calibration_records": calibration,
        "calibration": chosen_calibration,
        "retained_calibration": bool(retained),
    }


def embedding_eda(features: FloatArray, labels: IntArray, seed: int = 17) -> dict[str, float]:
    """Deterministic geometry diagnostics used to justify preprocessing choices."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or len(features) != len(labels) or not np.isfinite(features).all():
        raise ValueError("EDA requires a finite row-aligned feature matrix.")
    norms = np.linalg.norm(features, axis=1)
    mean = features.mean(axis=0)
    centered = features - mean
    covariance = np.cov(centered, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)[::-1]
    probabilities = eigenvalues / max(eigenvalues.sum(), 1e-12)
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    effective_rank = float(np.exp(entropy))
    common_energy = float(np.dot(mean, mean) / max(np.mean(norms**2), 1e-12))

    rng = np.random.default_rng(seed)
    pair_count = min(50_000, len(features) * 3)
    left = rng.integers(0, len(features), pair_count)
    right = rng.integers(0, len(features), pair_count)
    distinct = left != right
    left, right = left[distinct], right[distinct]
    normalized = normalize_rows(features)
    cosine = np.sum(normalized[left] * normalized[right], axis=1)
    same = labels[left] == labels[right]
    within = cosine[same]
    between = cosine[~same]
    pooled = np.sqrt((within.var() + between.var()) / 2) if len(within) else np.nan
    classes, prototypes = prototype_bank(features, labels)
    source_scores = normalized @ prototypes.T
    correct_positions = np.searchsorted(classes, labels)
    correct = source_scores[np.arange(len(labels)), correct_positions]
    source_scores[np.arange(len(labels)), correct_positions] = -np.inf
    margins = correct - source_scores.max(axis=1)
    predictions = classes[np.argmax(normalized @ prototypes.T, axis=1)]
    counts = np.bincount(np.searchsorted(classes, predictions), minlength=len(classes))
    class_counts = np.unique(labels, return_counts=True)[1]
    return {
        "samples": float(len(features)),
        "classes": float(len(class_counts)),
        "trials_per_class_min": float(class_counts.min()),
        "trials_per_class_max": float(class_counts.max()),
        "dimensions": float(features.shape[1]),
        "constant_fraction": float(np.mean(np.var(features, axis=0) < 1e-12)),
        "total_variance": float(np.var(features, axis=0, ddof=1).sum()),
        "norm_mean": float(norms.mean()),
        "norm_sd": float(norms.std(ddof=1)),
        "common_direction_energy": common_energy,
        "effective_rank": effective_rank,
        "within_class_cosine": float(within.mean()),
        "between_class_cosine": float(between.mean()),
        "standardized_separation": float((within.mean() - between.mean()) / max(pooled, 1e-12)),
        "nearest_prototype_accuracy": float(np.mean(predictions == labels)),
        "prototype_margin_mean": float(margins.mean()),
        "prototype_hubness_skew": float(
            np.mean(((counts - counts.mean()) / max(counts.std(), 1e-12)) ** 3)
        ),
    }


def config_hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def paired_change_statistics(
    differences: NDArray[np.floating], seed: int = 17, bootstrap_samples: int = 10_000
) -> dict[str, float]:
    """Participant-paired interval, exact sign-flip test, and standardized effect."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Paired comparisons require at least two finite participant changes.")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
    observed = abs(values.mean())
    combinations = 1 << len(values)
    absolute_null = np.empty(combinations)
    for code in range(combinations):
        signs = np.asarray([1 if code & (1 << index) else -1 for index in range(len(values))])
        absolute_null[code] = abs(np.mean(signs * values))
    standard_deviation = values.std(ddof=1)
    return {
        "mean_change": float(values.mean()),
        "sample_sd": float(standard_deviation),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "sign_flip_p": float(np.mean(absolute_null >= observed - 1e-15)),
        "paired_effect_dz": float(values.mean() / max(standard_deviation, 1e-12)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values in their original order."""

    if any(not 0 <= value <= 1 for value in p_values):
        raise ValueError("p-values must lie between zero and one.")
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def run_baseline_anchored_participant(
    participant: str,
    reference_sources: dict[str, Modalities],
    external_sources: dict[str, Modalities],
    config: ExperimentConfig,
    *,
    full: bool = True,
) -> tuple[list[ClassicalMetric], dict[str, object], dict[str, IntArray]]:
    """Run S0–S4 while comparing each row with its predecessor and fixed S0."""

    if "cornet-s" not in reference_sources or "cornet-s" not in external_sources:
        raise ValueError("The permanent S0 baseline requires cornet-s.")
    if set(reference_sources) != set(external_sources):
        raise ValueError("Reference and external semantic sources must match.")
    baseline_metrics, baseline_provenance, baseline_predictions = fit_and_score(
        reference_sources["cornet-s"],
        external_sources["cornet-s"],
        PreprocessingSpec(),
        config.classical_baseline_alpha,
        config.random_seed,
    )
    baseline_reference = reference_sources["cornet-s"]
    baseline_external = external_sources["cornet-s"]
    rng = np.random.default_rng(config.random_seed)
    transform = MatchedPreprocessor(PreprocessingSpec(), config.random_seed).fit(
        baseline_reference.eeg, baseline_reference.image, baseline_reference.labels
    )
    candidates, raw_prototypes = prototype_bank(
        baseline_external.image, baseline_external.labels
    )
    prototypes = transform.semantics(raw_prototypes)
    random_scores = rng.normal(size=(len(baseline_external.labels), len(candidates)))
    shuffled = rng.permutation(len(baseline_reference.labels))
    shuffled_decoder = Ridge(alpha=config.classical_baseline_alpha).fit(
        transform.eeg(baseline_reference.eeg),
        transform.semantics(baseline_reference.image[shuffled]),
    )
    shuffled_query = normalize_rows(shuffled_decoder.predict(transform.eeg(baseline_external.eeg)))
    valid_query = normalize_rows(
        Ridge(alpha=config.classical_baseline_alpha)
        .fit(transform.eeg(baseline_reference.eeg), transform.semantics(baseline_reference.image))
        .predict(transform.eeg(baseline_external.eeg))
    )
    permuted_candidates = rng.permutation(candidates)
    controls = {
        "random_ranking": ranking_metrics(random_scores, baseline_external.labels, candidates),
        "shuffled_pairs": ranking_metrics(
            shuffled_query @ prototypes.T, baseline_external.labels, candidates
        ),
        "permuted_prototype_labels": ranking_metrics(
            valid_query @ prototypes.T, baseline_external.labels, permuted_candidates
        ),
    }
    rows: list[ClassicalMetric] = []
    predictions = {"S0:cornet-s": baseline_predictions}
    diagnostics: dict[str, object] = {
        "controls": controls,
        "baseline": {"metrics": baseline_metrics, "provenance": baseline_provenance},
        "sources": {},
    }

    def append_stage(
        stage: str,
        predecessor: str,
        source: str,
        spec: PreprocessingSpec,
        alpha: float,
        retrieval: str,
        metrics: dict[str, float],
        previous: dict[str, float],
    ) -> None:
        payload = {
            "stage": stage,
            "source": source,
            "spec": asdict(spec),
            "alpha": alpha,
            "retrieval": retrieval,
            "seed": config.random_seed,
        }
        for metric, score in metrics.items():
            rows.append(
                ClassicalMetric(
                    participant,
                    stage,
                    predecessor,
                    "S0-cornet-ridge",
                    source,
                    spec.name,
                    alpha,
                    retrieval,
                    len(np.unique(external_sources[source].labels)),
                    metric,
                    score,
                    score - previous.get(metric, np.nan),
                    score - baseline_metrics.get(metric, np.nan),
                    config_hash(payload),
                )
            )

    append_stage(
        "S0", "controls", "cornet-s", PreprocessingSpec(),
        config.classical_baseline_alpha, "cosine", baseline_metrics, baseline_metrics
    )
    for source in sorted(reference_sources):
        reference = reference_sources[source]
        external = external_sources[source]
        if not (
            np.array_equal(reference.labels, reference_sources["cornet-s"].labels)
            and np.array_equal(external.labels, external_sources["cornet-s"].labels)
        ):
            raise ValueError(f"{source} labels or rows do not match the fixed baseline.")
        s1_metrics, s1_provenance, s1_predictions = fit_and_score(
            reference,
            external,
            PreprocessingSpec(),
            config.classical_baseline_alpha,
            config.random_seed,
        )
        predictions[f"S1:{source}"] = s1_predictions
        if source != "cornet-s":
            append_stage(
                "S1", "S0", source, PreprocessingSpec(),
                config.classical_baseline_alpha, "cosine", s1_metrics, baseline_metrics
            )
        source_report: dict[str, object] = {
            "eda": embedding_eda(reference.image, reference.labels, config.random_seed),
            "s1": {"metrics": s1_metrics, "provenance": s1_provenance},
        }
        if not full:
            diagnostics["sources"][source] = source_report
            continue

        selection = select_pipeline(reference, config)
        s2_spec = PreprocessingSpec(**selection["selected_preprocessing"]["spec"])
        s2_metrics, _, s2_predictions = fit_and_score(
            reference,
            external,
            s2_spec,
            config.classical_baseline_alpha,
            config.random_seed,
        )
        predictions[f"S2:{source}"] = s2_predictions
        append_stage(
            "S2", "S1", source, s2_spec, config.classical_baseline_alpha,
            "cosine", s2_metrics, s1_metrics
        )

        alpha = float(selection["selected"]["alpha"])
        s3_metrics, _, s3_predictions = fit_and_score(
            reference, external, s2_spec, alpha, config.random_seed
        )
        predictions[f"S3:{source}"] = s3_predictions
        append_stage("S3", "S2", source, s2_spec, alpha, "cosine", s3_metrics, s2_metrics)

        calibration = selection["calibration"]
        retrieval = str(calibration["rule"])
        neighbors = int(calibration["neighbors"])
        s4_metrics, s4_provenance, s4_predictions = fit_and_score(
            reference, external, s2_spec, alpha, config.random_seed, retrieval, neighbors
        )
        predictions[f"S4:{source}"] = s4_predictions
        append_stage("S4", "S3", source, s2_spec, alpha, retrieval, s4_metrics, s3_metrics)
        source_report.update(
            {
                "selection": selection,
                "s2_metrics": s2_metrics,
                "s3_metrics": s3_metrics,
                "s4_metrics": s4_metrics,
                "s4_provenance": s4_provenance,
            }
        )
        diagnostics["sources"][source] = source_report
    diagnostics["prediction_hashes"] = {
        name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
        for name, values in predictions.items()
    }
    return rows, diagnostics, predictions

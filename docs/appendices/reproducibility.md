# Reproducibility and release record

## Reproduction stages

1. Acquire the BraVL-format EEG arrays and verify documented checksums.
2. Acquire THINGS-EEG2 stimulus metadata and build the immutable row/stimulus manifest.
3. Build the non-EEG concept gallery, excluding EEG images by filename and SHA-256.
4. Extract frozen CORnet-S, DINOv2, and diagnostic CLIP features.
5. Run five locked seeds with reference-only representation and decoder selection.
6. Evaluate the frozen models once on 200 ID-grouped 80-repeat averages.
7. Run negative controls, aggregate by participant, and publish deterministic outputs.

The public command surface is available with `multimodal-eeg --help`. The main commands are
`prepare-nice-templates`, `run-nice-averaged-comparison`, and `publish-nice-study`. Paths to raw
datasets, passwords, images, feature caches, and model bundles must remain local.

## Release gates

- Ten participants, two visual targets, and all five seeds are complete.
- Every test average contains exactly 80 rows from one immutable stimulus ID.
- The gallery contains exactly 200 templates and no EEG image filename/content overlap.
- Scaling, PCA, representation selection, and Ridge fitting use reference data only.
- All broken-information controls pass their thresholds.
- Independent cosine ranks agree exactly and model hashes remain unchanged.
- Participant-level aggregation and Student-t intervals reproduce committed tables.
- Two publication rebuilds produce identical tables and figures.
- Tests, lint, compilation, links, tracked-file size, secret scanning, and provenance checks pass.

## Committed and excluded material

Committed files contain source code, configuration, compact participant/seed summaries, portable
IDs/hashes, documentation, and figures. Raw EEG, THINGS images, proprietary archive material,
passwords, API keys, checkpoints, fitted bundles, caches, and row-level predictions are excluded.

[`provenance.json`](../results/nice/provenance.json) lists the claim, qualification, gallery audits,
publication file hashes, and figure hashes. The upstream BraVL preprocessing boundary remains
unresolved because raw-to-distributed reconstruction is outside the retained data basis.

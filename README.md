# EEG image classifier

This repository contains a leakage-controlled study of concept retrieval from averaged EEG.

For each participant, 80 EEG recordings of each test image are averaged into one cleaner query.
A linear decoder predicts a visual representation, then ranks 200 concept templates made only
from different, non-EEG images. This follows the evaluation structure used by NICE.

## Main result

| Visual representation | Top-1 | Top-5 |
| --- | ---: | ---: |
| CORnet-S | 15.81% | 41.76% |
| DINOv2 | 12.81% | 35.23% |
| Chance | 0.50% | 2.50% |

![Comparison with NICE](figures/nice/literature-comparison.png)

These results describe repeated-presentation concept retrieval, not ordinary real-time decoding.
Repository-controlled leakage checks passed. The earlier preprocessing used to create the
distributed BraVL arrays cannot be reconstructed here and remains an explicit limitation.

## Documentation

- [Complete plain-language report](docs/report.md)
- [Metric explanations](docs/appendices/metrics.md)
- [Leakage-control explanations](docs/appendices/leakage-controls.md)
- [Reproducibility and provenance](docs/appendices/reproducibility.md)
- [Published result files](docs/results/nice/README.md)
- [References](docs/references.bib)

## Install

Python 3.10–3.13 is supported.

```powershell
python -m pip install -e ".[dev]"
multimodal-eeg --help
```

Raw EEG, THINGS images, archive passwords, model weights, fitted bundles, caches, and row-level
predictions are intentionally excluded from GitHub.

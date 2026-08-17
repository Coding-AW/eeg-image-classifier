"""Small statistical summaries shared by the NICE-style study."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def aggregate_metrics(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    """Summarise participant-level scores within each analysis stage."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["semantic_source"], row["metric"])].append(
            float(row["score"])
        )
    output: list[dict[str, object]] = []
    for (stage, source, metric), values in sorted(grouped.items()):
        array = np.asarray(values)
        standard_error = array.std(ddof=1) / np.sqrt(len(array))
        output.append(
            {
                "stage": stage,
                "semantic_source": source,
                "metric": metric,
                "participants": len(array),
                "mean": float(array.mean()),
                "sample_sd": float(array.std(ddof=1)),
                "ci_low": float(array.mean() - 1.96 * standard_error),
                "ci_high": float(array.mean() + 1.96 * standard_error),
            }
        )
    return output

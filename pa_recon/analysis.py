"""Paired source-group bootstrap comparisons for exploratory experiments."""

from __future__ import annotations

import numpy as np


def paired_comparison(reference: dict, candidate: dict, metric: str,
                      seed: int = 20261011, resamples: int = 10000) -> dict:
    """Candidate minus reference; resample independent groups, never pixels.

    Intervals are descriptive percentile intervals, not multiplicity-adjusted
    significance tests. Macro RMSE is the mean of group RMSE, not pooled RMSE.
    """
    for key in ("evaluation_split", "split_csv_sha256", "input_scale", "target_scale", "test_corruption"):
        if key not in reference or reference[key] != candidate.get(key):
            raise ValueError(f"Paired evaluations must share {key}")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    def groups(report):
        rows = report["per_group_metrics"]
        result = {str(row["group_id"]): float(row[metric]) for row in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate group IDs")
        if not all(np.isfinite(v) for v in result.values()):
            raise ValueError("Metrics must be finite")
        return result

    left, right = groups(reference), groups(candidate)
    if left.keys() != right.keys() or len(left) < 2:
        raise ValueError("Need the same two or more source groups")
    delta = np.array([right[key] - left[key] for key in sorted(left)])
    draws = np.random.default_rng(seed).choice(delta, (resamples, len(delta)), replace=True).mean(axis=1)
    return {"metric": metric, "direction": "candidate minus reference; lower favors candidate",
            "group_count": len(delta), "mean_paired_delta": float(delta.mean()),
            "paired_bootstrap_95ci": np.quantile(draws, [.025, .975]).tolist(),
            "resamples": resamples, "seed": seed, "unit": "source group",
            "interpretation": "Exploratory percentile interval; no multiplicity correction or clinical inference"}

"""Statistical observables for percolation and market simulations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import stats

from .percolation import ClusterConfiguration

ClusterExclusion = Literal["largest", "spanning", "none"]


def finite_cluster_sizes(
    configuration: ClusterConfiguration,
    exclusion: ClusterExclusion = "largest",
) -> np.ndarray:
    """Return cluster sizes after applying a documented exclusion rule.

    ``largest`` removes exactly one largest cluster from each realization.
    ``spanning`` removes every spanning/wrapping cluster, while ``none``
    retains all clusters.
    """
    if exclusion == "none" or configuration.cluster_count == 0:
        return configuration.sizes.copy()
    keep = np.ones(configuration.cluster_count, dtype=bool)
    if exclusion == "largest":
        keep[configuration.largest_label - 1] = False
    elif exclusion == "spanning":
        for label in configuration.spanning_labels:
            keep[label - 1] = False
    else:
        raise ValueError("exclusion must be 'largest', 'spanning', or 'none'")
    return configuration.sizes[keep]


def susceptibility_from_sizes(sizes: np.ndarray) -> float:
    """Compute sum(s^2 n_s) / sum(s n_s), returning zero if empty."""
    sizes = np.asarray(sizes, dtype=np.int64)
    denominator = int(sizes.sum())
    if denominator == 0:
        return 0.0
    numerator = int(np.dot(sizes, sizes))
    return numerator / denominator


def cluster_histogram(sizes: np.ndarray, max_size: int | None = None) -> np.ndarray:
    """Count clusters by integer size; element zero is retained as zero."""
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        length = 1 if max_size is None else max_size + 1
        return np.zeros(length, dtype=np.int64)
    if np.any(sizes <= 0):
        raise ValueError("cluster sizes must be positive")
    minlength = int(sizes.max()) + 1 if max_size is None else max_size + 1
    return np.bincount(sizes, minlength=minlength).astype(np.int64)


def standardize_returns(returns: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize returns using the sample standard deviation (ddof=1)."""
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        raise ValueError("at least two returns are required")
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if standard_deviation == 0.0:
        raise ValueError("returns have zero variance")
    return (values - mean) / standard_deviation, mean, standard_deviation


def nonzero_absolute_standardized_returns(
    returns: np.ndarray, standardized: np.ndarray
) -> np.ndarray:
    """Tail sample: discard exact zero raw returns before taking absolute z."""
    raw = np.asarray(returns, dtype=float)
    z = np.asarray(standardized, dtype=float)
    if raw.shape != z.shape:
        raise ValueError("returns and standardized values must have the same shape")
    valid = np.isfinite(raw) & np.isfinite(z) & (raw != 0)
    return np.abs(z[valid])


def return_statistics(
    returns: np.ndarray,
    standardized: np.ndarray,
    extreme_thresholds: tuple[float, ...],
    active_cluster_counts: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Descriptive moments and standardized extreme-event frequencies."""
    returns = np.asarray(returns, dtype=float)
    z = np.asarray(standardized, dtype=float)
    if returns.shape != z.shape or returns.size < 2:
        raise ValueError("returns and standardized values must have equal nontrivial shape")
    result: dict[str, float | int] = {
        "observations": int(returns.size),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std(ddof=1)),
        "return_skewness": float(stats.skew(returns, bias=False)),
        "return_excess_kurtosis": float(stats.kurtosis(returns, fisher=True, bias=False)),
        "zero_return_fraction": float(np.mean(returns == 0)),
        "negative_crash_probability": float(np.mean(z < -5.0)),
    }
    if active_cluster_counts is not None:
        result["mean_active_cluster_count"] = float(
            np.asarray(active_cluster_counts).mean()
        )
    for threshold in extreme_thresholds:
        label = f"p_abs_z_gt_{threshold:g}"
        result[label] = float(np.mean(np.abs(z) > threshold))
    return result


def return_distribution_table(
    standardized: np.ndarray,
    pdf_limit: float = 12.0,
    pdf_bins: int = 240,
    ccdf_points: int = 120,
    *,
    raw_returns: np.ndarray | None = None,
) -> list[dict[str, float | int | str]]:
    """Build signed-return PDF and absolute-return CCDF plot data."""
    z = np.asarray(standardized, dtype=float)
    edges = np.linspace(-pdf_limit, pdf_limit, pdf_bins + 1)
    counts, _ = np.histogram(z, bins=edges)
    widths = np.diff(edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    rows: list[dict[str, float | int | str]] = [
        {
            "distribution": "signed_pdf",
            "x": float(center),
            "value": float(count / (z.size * width)),
            "count": int(count),
        }
        for center, width, count in zip(centers, widths, counts, strict=True)
        if count > 0
    ]
    absolute = np.abs(z)
    positive = (nonzero_absolute_standardized_returns(raw_returns, z)
                if raw_returns is not None else absolute[absolute > 0])
    if positive.size:
        lower = max(float(positive.min()), 0.05)
        upper = max(lower * 1.001, float(positive.max()))
        thresholds = np.geomspace(lower, upper, ccdf_points)
        rows.extend(
            {
                "distribution": "absolute_ccdf",
                "x": float(threshold),
                "value": float(np.mean(absolute >= threshold)),
                "count": int(np.count_nonzero(absolute >= threshold)),
            }
            for threshold in thresholds
        )
    return rows

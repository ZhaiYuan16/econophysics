"""Distribution statistics and tail fits from resampled configuration blocks."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Callable


def block_histogram(values: np.ndarray, block_ids: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Compress repeated discrete returns without losing their block membership."""
    support, columns = np.unique(np.asarray(values, dtype=float), return_inverse=True)
    matrix = sparse.coo_matrix(
        (np.ones(len(columns), dtype=np.int32), (block_ids, columns)),
        shape=(int(np.max(block_ids)) + 1, len(support)),
    ).tocsr()
    return support, matrix


def resampled_counts(matrix: sparse.csr_matrix, selected_blocks: np.ndarray) -> np.ndarray:
    multiplicities = np.bincount(selected_blocks, minlength=matrix.shape[0])
    return np.asarray(multiplicities @ matrix).ravel().astype(np.int64)


def select_network_configuration_blocks(
    rng: np.random.Generator, networks: int, configurations: int
) -> np.ndarray:
    """Draw networks, then configurations within each selected network."""
    selected_networks = rng.integers(0, networks, size=networks)
    return np.concatenate([
        network * configurations + rng.integers(0, configurations, size=configurations)
        for network in selected_networks
    ])


def weighted_quantile(values: np.ndarray, counts: np.ndarray, probability: float) -> float:
    """Match NumPy's linear sample quantile without expanding observations."""
    cumulative = np.cumsum(counts)
    n = int(cumulative[-1])
    position = (n - 1) * probability
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    lo = values[np.searchsorted(cumulative, lower + 1)]
    hi = values[np.searchsorted(cumulative, upper + 1)]
    return float(lo + (position - lower) * (hi - lo))


def weighted_pareto_fit(
    absolute_values: np.ndarray,
    counts: np.ndarray,
    quantiles: tuple[float, ...],
    minimum_count: int,
) -> dict[str, float | int]:
    """Reselect xmin by the original minimum-KS rule after each bootstrap draw."""
    order = np.argsort(absolute_values)
    x = absolute_values[order]
    weights = counts[order]
    keep = (x > 0) & (weights > 0)
    x, weights = x[keep], weights[keep]
    candidates = []
    for q in quantiles:
        xmin = weighted_quantile(x, weights, q)
        tail = x >= xmin
        n = int(weights[tail].sum())
        if n < minimum_count:
            continue
        log_sum = float(np.dot(weights[tail], np.log(x[tail] / xmin)))
        if log_sum <= 0:
            continue
        alpha = 1 + n / log_sum
        cumulative = np.cumsum(weights[tail])
        empirical_last = cumulative / n
        empirical_first = (cumulative - weights[tail] + 1) / n
        model = 1 - (x[tail] / xmin) ** (1 - alpha)
        ks = float(max(np.max(np.abs(empirical_first - model)),
                       np.max(np.abs(empirical_last - model))))
        candidates.append({"xmin_quantile": q, "xmin": xmin, "alpha_pdf": alpha,
                           "ks_distance": ks, "tail_count": n})
    if not candidates:
        raise ValueError("no valid Pareto tail candidate")
    return min(candidates, key=lambda candidate: candidate["ks_distance"])


def weighted_statistics(
    support: np.ndarray, counts: np.ndarray, quantiles: tuple[float, ...], minimum_count: int,
    exclude_zero_returns: bool = False,
) -> dict[str, float | int]:
    n = int(counts.sum())
    mean = float(np.dot(support, counts) / n)
    variance = float(np.dot((support - mean) ** 2, counts) / (n - 1))
    std = np.sqrt(variance)
    z = (support - mean) / std
    centered = support - mean
    m2 = float(np.dot(centered**2, counts) / n)
    m4 = float(np.dot(centered**4, counts) / n)
    g2 = m4 / m2**2 - 3
    kurtosis = (n - 1) * ((n + 1) * g2 + 6) / ((n - 2) * (n - 3))
    eligible = support != 0 if exclude_zero_returns else np.ones(len(support), dtype=bool)
    tail = weighted_pareto_fit(np.abs(z[eligible]), counts[eligible], quantiles, minimum_count)
    return {"return_mean": mean, "return_std": std, "return_excess_kurtosis": kurtosis,
            "zero_return_fraction": float(counts[support == 0].sum() / n),
            "p_abs_z_gt_5": float(counts[np.abs(z) > 5].sum() / n),
            "negative_crash_probability": float(counts[z < -5].sum() / n),
            **tail}


def selection_aware_tail_gof(
    support: np.ndarray, matrix: sparse.csr_matrix, fit: dict,
    quantiles: tuple[float, ...], minimum: int, rng: np.random.Generator,
    draws: int, selector: Callable[[np.random.Generator], np.ndarray],
    exclude_zero_returns: bool = False,
) -> float:
    """Semiparametric common-Pareto null with resampled block counts and xmin selection."""
    original = np.asarray(matrix.sum(axis=0)).ravel()
    mean = np.dot(support, original) / original.sum()
    std = np.sqrt(np.dot((support - mean) ** 2, original) / (original.sum() - 1))
    absolute = np.abs((support - mean) / std)
    eligible = support != 0 if exclude_zero_returns else np.ones(len(support), dtype=bool)
    xmin, alpha = float(fit["xmin"]), float(fit["alpha_pdf"])
    exceed = 0
    for _ in range(draws):
        counts = resampled_counts(matrix, selector(rng))
        body_mask = eligible & (absolute > 0) & (absolute < xmin)
        n_tail = int(counts[eligible & (absolute >= xmin)].sum())
        synthetic_tail = xmin * (1 - rng.random(n_tail)) ** (-1 / (alpha - 1))
        synthetic = weighted_pareto_fit(
            np.concatenate([absolute[body_mask], synthetic_tail]),
            np.concatenate([counts[body_mask], np.ones(n_tail, dtype=np.int64)]),
            quantiles, minimum,
        )
        exceed += synthetic["ks_distance"] >= fit["ks_distance"]
    return (exceed + 1) / (draws + 1)

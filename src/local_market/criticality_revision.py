"""Configuration-level measurements and inference for open square site percolation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

from .networks import sample_lattice_occupancy
from .percolation import identify_lattice_clusters
from .statistics import finite_cluster_sizes


RULES = ("largest_excluded", "all_spanning_excluded")
EVENTS = ("span_x", "span_y", "span_any", "span_both")


def ratio_grid(coarse: list[float], center: float, half_width: float,
               common_window: list[float], step: float) -> tuple[float, ...]:
    """Union of original grid, local peak window, and common crossing window."""
    if step <= 0 or half_width <= 0:
        raise ValueError("grid step and half width must be positive")
    def window(low: float, high: float) -> set[float]:
        return {round(float(x), 6) for x in np.arange(low, high + step / 2, step)}
    return tuple(sorted({round(x, 6) for x in coarse} |
                        window(center - half_width, center + half_width) |
                        window(*common_window)))


@dataclass
class ScanResult:
    samples: pd.DataFrame
    histograms: pd.DataFrame
    runtime_seconds: float
    pooled_counts: pd.DataFrame | None = None


def scan(lengths: list[int], ratios_by_length: dict[int, tuple[float, ...]],
         counts: dict[int, int], p_c: float, seed: int,
         histogram_output: Path | None = None) -> ScanResult:
    """Use each independent occupancy configuration for every observable."""
    start = perf_counter()
    rows = []
    hist = []
    pooled = Counter()
    stream = None
    writer = None
    if histogram_output is not None:
        existed = histogram_output.exists()
        stream = histogram_output.open("a", newline="", encoding="utf-8")
        writer = csv.writer(stream)
        if not existed:
            writer.writerow(("length", "p_ratio", "p", "configuration_id",
                             "exclusion", "cluster_size", "count"))
    for length in lengths:
        for ratio in ratios_by_length[length]:
            p = p_c * ratio
            for sample in range(counts[length]):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [seed, length, int(round(ratio * 1_000_000)), sample]))
                clusters = identify_lattice_clusters(
                    sample_lattice_occupancy(length, p, rng), "open")
                x, y = clusters.spans_horizontal, clusters.spans_vertical
                row = {"length": length, "p_ratio": ratio, "p": p,
                       "configuration_id": sample, "occupied_count": clusters.occupied_count,
                       "largest_size": clusters.largest_size,
                       "p_infinity": clusters.p_infinity,
                       "span_x": int(x), "span_y": int(y),
                       "span_any": int(x or y), "span_both": int(x and y)}
                sets = {"all": clusters.sizes,
                        "largest_excluded": finite_cluster_sizes(clusters, "largest"),
                        "all_spanning_excluded": finite_cluster_sizes(clusters, "spanning")}
                for rule in RULES:
                    sizes = sets[rule]
                    row[f"{rule}_m1"] = int(sizes.sum())
                    row[f"{rule}_m2"] = int(np.dot(sizes, sizes))
                rows.append(row)
                configuration_hist = []
                for rule, sizes in sets.items():
                    for size, count in Counter(sizes.tolist()).items():
                        record = (length, ratio, p, sample, rule, size, count)
                        configuration_hist.append(record)
                        pooled[(length, ratio, p, rule, size)] += count
                if writer is not None:
                    writer.writerows(configuration_hist)
                    if np.isclose(ratio, 1.0):
                        hist.extend(configuration_hist)
                else:
                    hist.extend(configuration_hist)
    if stream is not None:
        stream.close()
    histograms = pd.DataFrame(hist, columns=["length", "p_ratio", "p",
        "configuration_id", "exclusion", "cluster_size", "count"])
    pooled_frame = pd.DataFrame(
        [(length, ratio, p, rule, size, count)
         for (length, ratio, p, rule, size), count in pooled.items()],
        columns=["length", "p_ratio", "p", "exclusion", "cluster_size", "C_s"])
    return ScanResult(pd.DataFrame(rows), histograms, perf_counter() - start, pooled_frame)


def _ratio(second: np.ndarray, first: np.ndarray) -> float:
    denominator = first.sum()
    return float(second.sum() / denominator) if denominator else 0.0


def summarize(samples: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    """Bootstrap whole occupancy configurations, keeping their observables together."""
    rng = np.random.default_rng(seed)
    rows = []
    for (length, ratio, p), group in samples.groupby(["length", "p_ratio", "p"]):
        n = len(group)
        row = {"length": length, "p_ratio": ratio, "p": p, "samples": n}
        indices = rng.integers(0, n, size=(replicates, n))
        for column in ("p_infinity", *EVENTS):
            values = group[column].to_numpy(dtype=float)
            boots = values[indices].mean(axis=1)
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if n > 1 else 0.0
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = np.quantile(boots, [0.025, 0.975])
            if column in EVENTS:
                q = values.mean()
                row[f"{column}_binomial_se"] = float(np.sqrt(q * (1 - q) / n))
        for rule in RULES:
            first = group[f"{rule}_m1"].to_numpy(dtype=np.int64)
            second = group[f"{rule}_m2"].to_numpy(dtype=np.int64)
            numerator = second[indices].sum(axis=1)
            denominator = first[indices].sum(axis=1)
            boots = np.divide(numerator, denominator, out=np.zeros(replicates), where=denominator != 0)
            row[f"chi_{rule}"] = _ratio(second, first)
            row[f"chi_{rule}_m1"] = int(first.sum())
            row[f"chi_{rule}_m2"] = int(second.sum())
            row[f"chi_{rule}_ci_low"], row[f"chi_{rule}_ci_high"] = np.quantile(boots, [0.025, 0.975])
        rows.append(row)
    return pd.DataFrame(rows)


def _peak(p: np.ndarray, values: np.ndarray) -> tuple[float, float, float]:
    index = int(np.argmax(values))
    grid = float(p[index])
    interpolated = np.nan
    if 0 < index < len(p) - 1:
        a, b, _ = np.polyfit(p[index - 1:index + 2], values[index - 1:index + 2], 2)
        vertex = -b / (2 * a) if a < 0 else np.nan
        if np.isfinite(vertex) and p[index - 1] <= vertex <= p[index + 1]:
            interpolated = float(vertex)
    return grid, interpolated, float(values[index])


def pseudocritical(samples: pd.DataFrame, summary: pd.DataFrame,
                   replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for length, group in summary.groupby("length"):
        group = group.sort_values("p")
        p = group.p.to_numpy()
        subsets = [samples[(samples.length == length) & np.isclose(samples.p, value)] for value in p]
        for rule in RULES:
            column = f"chi_{rule}"
            grid, vertex, height = _peak(p, group[column].to_numpy())
            estimates = []
            for _ in range(replicates):
                curve = []
                for subset in subsets:
                    indices = rng.integers(0, len(subset), len(subset))
                    curve.append(_ratio(subset[f"{rule}_m2"].to_numpy()[indices],
                                        subset[f"{rule}_m1"].to_numpy()[indices]))
                estimates.append(_peak(p, np.asarray(curve))[0])
            low, high = np.quantile(estimates, [0.025, 0.975])
            rows.append({"length": length, "exclusion": rule, "p_grid_peak": grid,
                         "p_quadratic_peak": vertex, "chi_peak": height,
                         "p_ci_low": low, "p_ci_high": high,
                         "minimum_grid_spacing": float(np.min(np.diff(p))),
                         "peak_at_grid_boundary": bool(grid in (p[0], p[-1]))})
    return pd.DataFrame(rows)


def crossing(p: np.ndarray, a: np.ndarray, b: np.ndarray, p_c: float) -> float:
    """Linear interpolation in sign-changing brackets, closest to theory if multiple."""
    difference = a - b
    candidates = []
    for i in range(len(p) - 1):
        if difference[i] == 0:
            candidates.append(float(p[i]))
        elif difference[i] * difference[i + 1] < 0:
            candidates.append(float(p[i] - difference[i] * (p[i + 1] - p[i]) /
                                    (difference[i + 1] - difference[i])))
    if difference[-1] == 0:
        candidates.append(float(p[-1]))
    return min(candidates, key=lambda value: abs(value - p_c)) if candidates else np.nan


def crossings(samples: pd.DataFrame, lengths: list[int], common_ratios: list[float],
              p_c: float, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ratios = np.asarray(common_ratios)
    p = p_c * ratios
    for left, right in zip(lengths[:-1], lengths[1:]):
        data = {}
        for length in (left, right):
            data[length] = [samples[(samples.length == length) &
                 np.isclose(samples.p_ratio, ratio)]["span_x"].to_numpy(dtype=float)
                 for ratio in ratios]
        if any(len(values) == 0 for groups in data.values() for values in groups):
            continue
        means = {length: np.array([values.mean() for values in groups])
                 for length, groups in data.items()}
        point = crossing(p, means[left], means[right], p_c)
        boots = []
        for _ in range(replicates):
            curves = {length: np.array([rng.choice(values, len(values)).mean()
                         for values in groups]) for length, groups in data.items()}
            boots.append(crossing(p, curves[left], curves[right], p_c))
        finite = np.asarray(boots)[np.isfinite(boots)]
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
        rows.append({"length_left": left, "length_right": right, "p_crossing": point,
                     "p_ci_low": low, "p_ci_high": high,
                     "bootstrap_crossing_fraction": len(finite) / replicates,
                     "multiple_crossing_rule": "nearest theoretical p_c"})
    return pd.DataFrame(rows, columns=["length_left", "length_right", "p_crossing",
        "p_ci_low", "p_ci_high", "bootstrap_crossing_fraction", "multiple_crossing_rule"])


def pooled_histogram(histograms: pd.DataFrame, samples: pd.DataFrame,
                     pooled_counts: pd.DataFrame | None = None) -> pd.DataFrame:
    if pooled_counts is None:
        counts = histograms.groupby(["length", "p_ratio", "p", "exclusion", "cluster_size"],
                                    as_index=False)["count"].sum().rename(columns={"count": "C_s"})
    else:
        counts = pooled_counts.groupby(["length", "p_ratio", "p", "exclusion", "cluster_size"],
                                      as_index=False)["C_s"].sum()
    sizes = samples.groupby(["length", "p_ratio", "p"], as_index=False).size().rename(columns={"size": "M"})
    counts = counts.merge(sizes, on=["length", "p_ratio", "p"])
    counts["n_s"] = counts.C_s / (counts.M * counts.length**2)
    return counts


def finite_powerlaw(values: np.ndarray, xmin: int, xmax: int) -> tuple[float, float, int]:
    """Exact normalized likelihood on the finite integer support, plus KS."""
    tail = np.sort(values[(values >= xmin) & (values <= xmax)])
    if len(tail) < 30 or xmax <= xmin:
        return np.nan, np.nan, len(tail)
    support = np.arange(xmin, xmax + 1)
    logs = np.log(support)
    def nll(alpha: float) -> float:
        return float(alpha * np.log(tail).sum() + len(tail) * logsumexp(-alpha * logs))
    fit = minimize_scalar(nll, bounds=(0.01, 6), method="bounded")
    alpha = float(fit.x)
    model = np.exp(-alpha * logs - logsumexp(-alpha * logs))
    empirical = np.searchsorted(tail, support, side="right") / len(tail)
    return alpha, float(np.max(np.abs(empirical - np.cumsum(model)))), len(tail)


def truncated_powerlaw(values: np.ndarray, xmin: int) -> tuple[float, float, float]:
    """Discrete, normalized cutoff power law on the observed finite-system support."""
    tail = values[values >= xmin]
    if len(tail) < 30:
        return np.nan, np.nan, np.nan
    support = np.arange(xmin, int(values.max()) + 1)
    logs = np.log(support)
    def nll(params: np.ndarray) -> float:
        alpha, log_sc = params
        weights = -alpha * logs - support / np.exp(log_sc)
        return float(alpha * np.log(tail).sum() + tail.sum() / np.exp(log_sc) +
                     len(tail) * logsumexp(weights))
    fit = minimize(nll, [2.0, np.log(max(float(np.mean(tail)), 2.0))],
                   bounds=[(0.01, 6), (np.log(1), np.log(max(values.max() * 100, 2)))])
    return float(fit.x[0]), float(np.exp(fit.x[1])), float(fit.fun)


def finite_cutoff_powerlaw(values: np.ndarray, xmin: int, xmax: int) -> tuple[float, float, float]:
    """Two-parameter cutoff model normalized on the comparison interval."""
    tail = values[(values >= xmin) & (values <= xmax)]
    support = np.arange(xmin, xmax + 1)
    logs = np.log(support)
    def nll(params: np.ndarray) -> float:
        alpha, log_sc = params
        weights = -alpha * logs - support / np.exp(log_sc)
        return float(alpha * np.log(tail).sum() + tail.sum() / np.exp(log_sc) +
                     len(tail) * logsumexp(weights))
    fit = minimize(nll, [2.0, np.log(max(float(np.mean(tail)), 2.0))],
                   bounds=[(0.01, 6), (np.log(1), np.log(max(xmax * 100, 2)))])
    return float(fit.x[0]), float(np.exp(fit.x[1])), float(fit.fun)


def tail_fits(histograms: pd.DataFrame, ratio: float, thresholds: list[int],
              replicates: int, seed: int) -> pd.DataFrame:
    """Fit finite support and cutoff models, resampling whole configurations."""
    rng = np.random.default_rng(seed)
    rows = []
    selected = histograms[np.isclose(histograms.p_ratio, ratio)]
    for (length, rule), group in selected.groupby(["length", "exclusion"]):
        if rule == "all":
            continue
        by_config = {int(sample): part for sample, part in group.groupby("configuration_id")}
        ids = np.array(sorted(by_config))
        def expand(chosen: np.ndarray) -> np.ndarray:
            parts = [np.repeat(by_config[int(i)].cluster_size.to_numpy(dtype=int),
                               by_config[int(i)]["count"].to_numpy(dtype=int)) for i in chosen]
            return np.concatenate(parts) if parts else np.array([], dtype=int)
        values = expand(ids)
        if len(values) == 0:
            continue
        size_counts = np.bincount(values)
        for threshold in thresholds:
            candidates = np.flatnonzero(size_counts >= threshold)
            xmax = int(candidates.max()) if len(candidates) else 0
            options = []
            for xmin in (1, 2, 3, 5, 10, 20):
                alpha, ks, n = finite_powerlaw(values, xmin, xmax)
                if np.isfinite(alpha) and n >= 100:
                    options.append((ks, xmin, alpha, n))
            if not options:
                rows.append({"length": length, "p_ratio": ratio, "exclusion": rule,
                             "count_threshold": threshold, "s_max_fit": xmax,
                             "status": "insufficient_data"})
                continue
            ks, xmin, alpha, n = min(options)
            estimates = []
            for _ in range(replicates):
                boot = expand(rng.choice(ids, len(ids), replace=True))
                boot_counts = np.bincount(boot) if len(boot) else np.array([])
                eligible = np.flatnonzero(boot_counts >= threshold)
                upper = int(eligible.max()) if len(eligible) else 0
                possible = [(distance, lower, value) for lower in (1, 2, 3, 5, 10, 20)
                            for value, distance, count in [finite_powerlaw(boot, lower, upper)]
                            if np.isfinite(value) and count >= 100]
                if possible:
                    estimates.append(min(possible)[2])
            low, high = np.quantile(estimates, [0.025, 0.975]) if estimates else (np.nan, np.nan)
            cutoff_alpha, sc, cutoff_nll = truncated_powerlaw(values, xmin)
            fit_values = values[(values >= xmin) & (values <= xmax)]
            support = np.arange(xmin, xmax + 1)
            power_nll = float(alpha * np.log(fit_values).sum() +
                              len(fit_values) * logsumexp(-alpha * np.log(support)))
            exp_fit = minimize_scalar(lambda rate: float(rate * fit_values.sum() +
                len(fit_values) * logsumexp(-rate * support)), bounds=(1e-8, 2), method="bounded")
            finite_cutoff_alpha, finite_sc, finite_cutoff_nll = finite_cutoff_powerlaw(values, xmin, xmax)
            model_cdf = np.cumsum(np.exp(-alpha * np.log(support) -
                                          logsumexp(-alpha * np.log(support))))
            gof_exceedances = 0
            for _ in range(min(replicates, 200)):
                synthetic = rng.choice(support, size=n, p=np.diff(np.r_[0, model_cdf]))
                _, synthetic_ks, _ = finite_powerlaw(synthetic, xmin, xmax)
                gof_exceedances += int(synthetic_ks >= ks)
            gof_replicates = min(replicates, 200)
            rows.append({"length": length, "p_ratio": ratio, "exclusion": rule,
                         "count_threshold": threshold, "s_min": xmin, "s_max_fit": xmax,
                         "tail_count": n, "alpha_pdf": alpha, "alpha_ci_low": low,
                         "alpha_ci_high": high, "bootstrap_valid_fraction": len(estimates) / replicates,
                         "ks_distance": ks, "powerlaw_aic": 2 + 2 * power_nll,
                         "fixed_interval_gof_p": (gof_exceedances + 1) / (gof_replicates + 1),
                         "gof_replicates": gof_replicates,
                         "exponential_aic_same_interval": 2 + 2 * exp_fit.fun,
                         "cutoff_aic_same_interval": 4 + 2 * finite_cutoff_nll,
                         "cutoff_alpha_same_interval": finite_cutoff_alpha,
                         "cutoff_s_c_same_interval": finite_sc,
                         "cutoff_alpha_pdf": cutoff_alpha, "cutoff_s_c": sc,
                         "cutoff_aic_full_tail": 4 + 2 * cutoff_nll,
                         "status": "ok", "upper_rule": "largest observed s with C_s >= threshold"})
    return pd.DataFrame(rows)

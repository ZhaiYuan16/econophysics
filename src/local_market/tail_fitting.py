"""Tail estimators and distribution comparisons for absolute returns."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import powerlaw
from scipy.optimize import minimize_scalar
from scipy.special import zeta


@dataclass(frozen=True)
class TailFit:
    xmin: float
    alpha_pdf: float
    ks_distance: float
    tail_count: int
    tail_fraction: float


def discrete_powerlaw_mle(
    values: np.ndarray, xmin: int
) -> tuple[float, float, int]:
    """Numerical discrete power-law MLE and exact Hurwitz-zeta KS distance."""
    positive = positive_finite(values)
    integers = positive.astype(np.int64)
    if np.any(integers != positive) or xmin < 1:
        raise ValueError("discrete power-law data and xmin must be positive integers")
    tail = np.sort(integers[integers >= xmin])
    count = int(tail.size)
    if count < 2:
        return float("nan"), float("nan"), count
    log_sum = float(np.log(tail).sum())

    def negative_log_likelihood(alpha: float) -> float:
        return float(count * np.log(zeta(alpha, xmin)) + alpha * log_sum)

    optimization = minimize_scalar(
        negative_log_likelihood,
        bounds=(1.000001, 10.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not optimization.success:
        return float("nan"), float("nan"), count
    alpha = float(optimization.x)
    support, cumulative_counts = np.unique(tail, return_counts=True)
    empirical_cdf = np.cumsum(cumulative_counts, dtype=float) / count
    model_cdf = 1.0 - zeta(alpha, support + 1) / zeta(alpha, xmin)
    ks = float(np.max(np.abs(empirical_cdf - model_cdf)))
    return float(alpha), ks, count


def select_discrete_powerlaw_tail(
    values: np.ndarray, minimum_count: int
) -> TailFit:
    """Select integer xmin by minimum KS distance using ``powerlaw``."""
    positive = positive_finite(values)
    integers = positive.astype(np.int64)
    if np.any(integers != positive):
        raise ValueError("discrete power-law observations must be integers")
    if integers.size < minimum_count:
        raise ValueError("not enough observations for discrete tail fitting")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = powerlaw.Fit(
            integers, discrete=True, estimate_discrete=False, verbose=False
        )
    xmin = int(fitted.xmin)
    alpha, ks, count = discrete_powerlaw_mle(integers, xmin)
    if count < minimum_count:
        raise ValueError("selected xmin retains too few tail observations")
    return TailFit(xmin, alpha, ks, count, count / integers.size)


def bootstrap_discrete_powerlaw_alpha(
    values: np.ndarray,
    xmin: int,
    replicates: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Nonparametric fixed-xmin uncertainty for a discrete exponent."""
    integers = positive_finite(values).astype(np.int64)
    tail = integers[integers >= xmin]
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sample = rng.choice(tail, size=tail.size, replace=True)
        estimates[replicate] = discrete_powerlaw_mle(sample, xmin)[0]
    return (
        float(estimates.std(ddof=1)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _sample_discrete_powerlaw(
    alpha: float, xmin: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw from a Zipf law conditional on integer values >= xmin."""
    accepted: list[np.ndarray] = []
    remaining = count
    while remaining:
        draw = rng.zipf(alpha, max(remaining * 2, 1000))
        retained = draw[draw >= xmin][:remaining]
        if retained.size:
            accepted.append(retained)
            remaining -= retained.size
    return np.concatenate(accepted)


def discrete_powerlaw_fixed_xmin_gof_pvalue(
    fit: TailFit,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """Conditional parametric-bootstrap KS p-value for integer tails."""
    distances = np.empty(replicates, dtype=float)
    xmin = int(fit.xmin)
    for replicate in range(replicates):
        synthetic = _sample_discrete_powerlaw(
            fit.alpha_pdf, xmin, fit.tail_count, rng
        )
        distances[replicate] = discrete_powerlaw_mle(synthetic, xmin)[1]
    exceedances = int(np.count_nonzero(distances >= fit.ks_distance))
    return float((exceedances + 1) / (replicates + 1))


def compare_discrete_tail_distributions(
    values: np.ndarray, xmin: int
) -> dict[str, float]:
    """Compare a discrete power law with exponential and truncated forms."""
    integers = positive_finite(values).astype(np.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = powerlaw.Fit(
            integers,
            xmin=xmin,
            discrete=True,
            estimate_discrete=False,
            verbose=False,
        )
        ratio_exponential, p_exponential = fitted.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )
        ratio_truncated, p_truncated = fitted.distribution_compare(
            "power_law", "truncated_power_law"
        )
    return {
        "powerlaw_package_alpha_pdf": float(fitted.power_law.alpha),
        "loglikelihood_ratio_power_vs_exponential": float(ratio_exponential),
        "p_power_vs_exponential": float(p_exponential),
        "loglikelihood_ratio_power_vs_truncated": float(ratio_truncated),
        "p_power_vs_truncated": float(p_truncated),
    }


def positive_finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def pareto_mle(values: np.ndarray, xmin: float) -> tuple[float, float, int]:
    """Continuous Pareto PDF exponent and KS distance above fixed xmin."""
    positive = positive_finite(values)
    tail = np.sort(positive[positive >= xmin])
    count = int(tail.size)
    if count < 2:
        return float("nan"), float("nan"), count
    log_sum = float(np.log(tail / xmin).sum())
    if log_sum <= 0:
        return float("inf"), float("inf"), count
    alpha = 1.0 + count / log_sum
    empirical_cdf = np.arange(1, count + 1, dtype=float) / count
    model_cdf = 1.0 - (tail / xmin) ** (1.0 - alpha)
    ks = float(np.max(np.abs(empirical_cdf - model_cdf)))
    return float(alpha), ks, count


def select_pareto_tail(
    values: np.ndarray,
    quantiles: tuple[float, ...],
    minimum_count: int,
) -> tuple[TailFit, list[dict[str, float | int]]]:
    """Choose xmin from prespecified quantiles by minimum KS distance."""
    positive = positive_finite(values)
    if positive.size < minimum_count:
        raise ValueError("not enough positive observations for tail fitting")
    rows: list[dict[str, float | int]] = []
    seen: set[float] = set()
    for quantile in quantiles:
        xmin = float(np.quantile(positive, quantile))
        if xmin <= 0 or xmin in seen:
            continue
        seen.add(xmin)
        alpha, ks, count = pareto_mle(positive, xmin)
        if count < minimum_count or not np.isfinite(alpha):
            continue
        rows.append(
            {
                "xmin_quantile": quantile,
                "xmin": xmin,
                "alpha_pdf": alpha,
                "ks_distance": ks,
                "tail_count": count,
                "tail_fraction": count / positive.size,
            }
        )
    if not rows:
        raise ValueError("no xmin candidate retained enough tail observations")
    best = min(rows, key=lambda row: float(row["ks_distance"]))
    return (
        TailFit(
            xmin=float(best["xmin"]),
            alpha_pdf=float(best["alpha_pdf"]),
            ks_distance=float(best["ks_distance"]),
            tail_count=int(best["tail_count"]),
            tail_fraction=float(best["tail_fraction"]),
        ),
        rows,
    )


def pareto_fixed_xmin_gof_pvalue(
    fit: TailFit,
    replicates: int,
    rng: np.random.Generator,
) -> float:
    """Conditional parametric-bootstrap KS p-value at a fixed xmin."""
    synthetic_distances = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        uniform = rng.random(fit.tail_count)
        synthetic = fit.xmin * (1.0 - uniform) ** (-1.0 / (fit.alpha_pdf - 1.0))
        _, distance, _ = pareto_mle(synthetic, fit.xmin)
        synthetic_distances[replicate] = distance
    exceedances = int(np.count_nonzero(synthetic_distances >= fit.ks_distance))
    # The plus-one correction avoids reporting an impossible exact zero from
    # a finite Monte Carlo sample.
    return float((exceedances + 1) / (replicates + 1))


def compare_tail_distributions(
    values: np.ndarray, xmin: float
) -> dict[str, float]:
    """Compare a continuous power law with exponential and truncated alternatives."""
    positive = positive_finite(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(
            positive,
            xmin=xmin,
            discrete=False,
            verbose=False,
            parameter_ranges={"alpha": (1.000001, None)},
        )
        ratio_exponential, p_exponential = fit.distribution_compare(
            "power_law", "exponential", normalized_ratio=True
        )
        ratio_truncated, p_truncated = fit.distribution_compare(
            "power_law", "truncated_power_law"
        )
    return {
        "powerlaw_package_alpha_pdf": float(fit.power_law.alpha),
        "loglikelihood_ratio_power_vs_exponential": float(ratio_exponential),
        "p_power_vs_exponential": float(p_exponential),
        "loglikelihood_ratio_power_vs_truncated": float(ratio_truncated),
        "p_power_vs_truncated": float(p_truncated),
    }

"""Trading dynamics for the Cont-Bouchaud market model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .percolation import ClusterConfiguration


@dataclass(frozen=True)
class MarketSimulation:
    demands: np.ndarray
    returns: np.ndarray
    log_prices: np.ndarray
    occupied_counts: np.ndarray
    cluster_counts: np.ndarray


def cluster_action_probabilities(activity: float) -> tuple[float, float, float]:
    """Return probabilities for sell, inactive, and buy actions."""
    if not 0.0 <= activity <= 1.0:
        raise ValueError("activity must lie in [0, 1]")
    return activity / 2.0, 1.0 - activity, activity / 2.0


def sample_cluster_actions(
    cluster_count: int, activity: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample sell, inactive, and buy actions for independent clusters."""
    if cluster_count < 0:
        raise ValueError("cluster_count must be non-negative")
    cluster_action_probabilities(activity)
    draws = rng.random(cluster_count)
    actions = np.zeros(cluster_count, dtype=np.int8)
    actions[draws < activity / 2.0] = -1
    actions[draws >= 1.0 - activity / 2.0] = 1
    return actions


def excess_demand(
    configuration: ClusterConfiguration, actions: np.ndarray
) -> int:
    actions = np.asarray(actions)
    if actions.shape != configuration.sizes.shape:
        raise ValueError("one action is required for each cluster")
    if not np.isin(actions, (-1, 0, 1)).all():
        raise ValueError("actions must be -1, 0, or 1")
    return int(np.dot(configuration.sizes, actions))


def sample_excess_demands(
    cluster_sizes: np.ndarray,
    steps: int,
    activity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized excess-demand samples for one fixed cluster structure."""
    demands, _ = sample_demands_and_active_counts(cluster_sizes, steps, activity, rng)
    return demands


def sample_demands_and_active_counts(
    cluster_sizes: np.ndarray,
    steps: int,
    activity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample excess demand and the number of active clusters per step."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    cluster_action_probabilities(activity)
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    if sizes.ndim != 1 or np.any(sizes <= 0):
        if sizes.size == 0 and sizes.ndim == 1:
            zeros = np.zeros(steps, dtype=np.int64)
            return zeros, zeros.copy()
        raise ValueError("cluster_sizes must be a one-dimensional positive array")
    draws = rng.random((steps, sizes.size))
    actions = (draws >= 1.0 - activity / 2.0).astype(np.int8)
    actions -= (draws < activity / 2.0).astype(np.int8)
    demands = actions @ sizes
    active_counts = np.count_nonzero(actions, axis=1).astype(np.int64)
    return demands, active_counts


def _validate_market_inputs(steps: int, market_depth: float) -> None:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if market_depth <= 0.0:
        raise ValueError("market_depth must be positive")


def simulate_quenched(
    configuration: ClusterConfiguration,
    steps: int,
    activity: float,
    market_depth: float,
    rng: np.random.Generator,
    initial_log_price: float = 0.0,
) -> MarketSimulation:
    """Trade repeatedly on one fixed cluster configuration."""
    _validate_market_inputs(steps, market_depth)
    demands = sample_excess_demands(configuration.sizes, steps, activity, rng)
    returns = demands.astype(float) / market_depth
    log_prices = np.concatenate(
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns))
    )
    return MarketSimulation(
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=np.full(steps, configuration.occupied_count, dtype=np.int64),
        cluster_counts=np.full(steps, configuration.cluster_count, dtype=np.int64),
    )


def simulate_annealed(
    configuration_factory: Callable[[], ClusterConfiguration],
    steps: int,
    activity: float,
    market_depth: float,
    rng: np.random.Generator,
    initial_log_price: float = 0.0,
) -> MarketSimulation:
    """Regenerate the occupied configuration before every trading step."""
    _validate_market_inputs(steps, market_depth)
    demands = np.empty(steps, dtype=np.int64)
    occupied_counts = np.empty(steps, dtype=np.int64)
    cluster_counts = np.empty(steps, dtype=np.int64)
    for step in range(steps):
        configuration = configuration_factory()
        actions = sample_cluster_actions(configuration.cluster_count, activity, rng)
        demands[step] = excess_demand(configuration, actions)
        occupied_counts[step] = configuration.occupied_count
        cluster_counts[step] = configuration.cluster_count
    returns = demands.astype(float) / market_depth
    log_prices = np.concatenate(
        ([float(initial_log_price)], float(initial_log_price) + np.cumsum(returns))
    )
    return MarketSimulation(
        demands=demands,
        returns=returns,
        log_prices=log_prices,
        occupied_counts=occupied_counts,
        cluster_counts=cluster_counts,
    )

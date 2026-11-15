"""Optional driven-dissipative extension of the lattice CB model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from .cb_model import sample_demands_and_active_once
from .config import SOCConfig
from .percolation import identify_lattice_clusters


@dataclass(frozen=True)
class SOCSimulation:
    frame: pd.DataFrame
    final_occupancy: np.ndarray


def simulate_soc_path(
    config: SOCConfig,
    initial_occupancy: float,
    rng: np.random.Generator,
) -> SOCSimulation:
    """Run one slowly driven path with event-triggered node exit.

    Quiet steps add up to ``drive_nodes`` vacant sites. If the absolute excess
    demand exceeds the configured fraction of all potential investors, nodes
    in active clusters aligned with the demand sign exit independently with
    probability ``exit_probability``. Driving and dissipation never occur in
    the same step.
    """
    node_count = config.length**2
    occupied = rng.random((config.length, config.length)) < initial_occupancy
    rows: list[dict[str, object]] = []
    threshold = config.event_threshold_fraction * node_count
    for step in range(config.steps):
        clusters = identify_lattice_clusters(occupied, config.boundary)
        demand, actions = sample_demands_and_active_once(clusters, config.activity, rng)
        event = bool(abs(demand) >= threshold and demand != 0)
        removed_nodes = 0
        added_nodes = 0
        if event:
            direction = 1 if demand > 0 else -1
            aligned = np.flatnonzero(actions == direction) + 1
            if aligned.size:
                candidates = np.isin(clusters.labels, aligned)
                removals = candidates & (
                    rng.random(occupied.shape) < config.exit_probability
                )
                removed_nodes = int(np.count_nonzero(removals))
                occupied[removals] = False
        else:
            vacancies = np.flatnonzero(~occupied.ravel())
            if vacancies.size:
                chosen = rng.choice(
                    vacancies,
                    size=min(config.drive_nodes, vacancies.size),
                    replace=False,
                )
                occupied.ravel()[chosen] = True
                added_nodes = int(chosen.size)
        rows.append(
            {
                "step": step,
                "occupancy_before": clusters.occupied_count / node_count,
                "occupancy_after": float(occupied.mean()),
                "cluster_count": clusters.cluster_count,
                "largest_cluster_size": clusters.largest_size,
                "largest_occupied_fraction": clusters.largest_occupied_fraction,
                "demand": demand,
                "return": demand / config.market_depth,
                "event": event,
                "removed_nodes": removed_nodes,
                "added_nodes": added_nodes,
            }
        )
    return SOCSimulation(pd.DataFrame(rows), occupied.copy())

def run_soc_ensemble(
    config: SOCConfig, show_progress: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run paths from several initial states and summarize stationarity."""
    frames: list[pd.DataFrame] = []
    total = len(config.initial_occupancies) * config.replicates
    progress = tqdm(total=total, desc="SOC extension", unit="path", disable=not show_progress)
    for initial_index, initial in enumerate(config.initial_occupancies):
        for replicate in range(config.replicates):
            rng = np.random.default_rng(
                np.random.SeedSequence([config.seed, 600, initial_index, replicate])
            )
            result = simulate_soc_path(config, initial, rng)
            frame = result.frame.assign(
                initial_occupancy=initial,
                replicate=replicate,
                path_id=initial_index * config.replicates + replicate,
            )
            frame["event_sequence"] = np.where(
                frame["event"].to_numpy(dtype=bool),
                frame["event"].cumsum().to_numpy(dtype=np.int64),
                0,
            )
            frames.append(frame)
            progress.update(1)
    progress.close()
    paths = pd.concat(frames, ignore_index=True)
    stationary = paths[paths["step"] >= config.burn_in].copy()
    lower = config.critical_reference * (1.0 - config.critical_band_relative_width)
    upper = config.critical_reference * (1.0 + config.critical_band_relative_width)
    replicate_rows: list[dict[str, object]] = []
    for (initial, replicate), group in stationary.groupby(
        ["initial_occupancy", "replicate"], sort=True
    ):
        returns = group["return"].to_numpy(dtype=float)
        events = group["event"].to_numpy(dtype=bool)
        occupancy = group["occupancy_after"].to_numpy(dtype=float)
        midpoint = config.burn_in + (config.steps - config.burn_in) // 2
        early_mean = float(group.loc[group["step"] < midpoint, "occupancy_after"].mean())
        late_mean = float(group.loc[group["step"] >= midpoint, "occupancy_after"].mean())
        replicate_rows.append(
            {
                "initial_occupancy": initial,
                "replicate": int(replicate),
                "stationary_steps": len(group),
                "mean_occupancy": float(occupancy.mean()),
                "occupancy_std": float(occupancy.std(ddof=1)),
                "stationary_drift": late_mean - early_mean,
                "critical_band_fraction": float(
                    np.mean((occupancy >= lower) & (occupancy <= upper))
                ),
                "event_rate": float(events.mean()),
                "mean_event_removal": float(
                    group.loc[group["event"], "removed_nodes"].mean()
                ),
                "return_std": float(returns.std(ddof=1)),
                "return_excess_kurtosis": float(stats.kurtosis(returns, fisher=True)),
                "final_occupancy": float(group["occupancy_after"].iloc[-1]),
            }
        )
    replicates = pd.DataFrame(replicate_rows)
    summary = (
        replicates.groupby("initial_occupancy", as_index=False)
        .agg(
            replicates=("replicate", "count"),
            mean_occupancy=("mean_occupancy", "mean"),
            mean_occupancy_sd=("mean_occupancy", "std"),
            occupancy_std=("occupancy_std", "mean"),
            stationary_drift=("stationary_drift", "mean"),
            stationary_drift_sd=("stationary_drift", "std"),
            critical_band_fraction=("critical_band_fraction", "mean"),
            critical_band_fraction_sd=("critical_band_fraction", "std"),
            event_rate=("event_rate", "mean"),
            event_rate_sd=("event_rate", "std"),
            mean_event_removal=("mean_event_removal", "mean"),
            return_std=("return_std", "mean"),
            return_excess_kurtosis=("return_excess_kurtosis", "mean"),
            final_occupancy=("final_occupancy", "mean"),
        )
    )
    summary["critical_reference"] = config.critical_reference
    summary["critical_band_low"] = lower
    summary["critical_band_high"] = upper
    return paths, replicates, summary

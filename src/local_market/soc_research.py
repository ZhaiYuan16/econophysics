"""Independent SOC research dynamics; legacy soc.py remains the baseline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .cb_model import sample_demands_and_active_once
from .percolation import identify_lattice_clusters


@dataclass(frozen=True)
class ResearchConfig:
    length: int
    steps: int
    burn_in: int
    activity: float = .01
    theta: float = .08
    q: float = .25
    drive_nodes: int = 1
    mode: str = "baseline"
    max_rounds: int = 1000

    def __post_init__(self):
        if self.length < 2 or self.steps < 1 or not 0 <= self.burn_in < self.steps:
            raise ValueError("invalid lattice or time")
        if not 0 <= self.activity <= 1 or not 0 < self.theta < 1 or not 0 < self.q <= 1:
            raise ValueError("invalid action or dissipation parameter")
        if self.drive_nodes < 1 or self.max_rounds < 1 or self.mode not in {
            "baseline", "cascade", "no_dissipation", "random_dissipation"}:
            raise ValueError("invalid drive or mode")


def simulate(config: ResearchConfig, p0: float, seed: int, path_id: str):
    """Run one path; resample cluster actions after every cascade removal round."""
    rng = np.random.default_rng(seed)
    n = config.length**2
    occ = rng.random((config.length, config.length)) < p0
    rows, events, rounds = [], [], []
    last_event_step = None
    sequence = 0
    for step in range(config.steps):
        before = int(occ.sum())
        size = duration = first_demand = added = 0
        end_reason = "below_threshold"
        while True:
            clusters = identify_lattice_clusters(occ, "open")
            demand, actions = sample_demands_and_active_once(clusters, config.activity, rng)
            if duration == 0:
                first_demand = demand
            if abs(demand) < config.theta*n or demand == 0:
                end_reason = "below_threshold"
                break
            if config.mode == "no_dissipation":
                end_reason = "dissipation_disabled"
                break
            aligned = np.flatnonzero(actions == np.sign(demand)) + 1
            selected = np.isin(clusters.labels, aligned) & (rng.random(occ.shape) < config.q)
            count = int(selected.sum())
            if config.mode == "random_dissipation" and count:
                available = np.flatnonzero(occ.ravel())
                selected = np.zeros(occ.shape, dtype=bool)
                selected.ravel()[rng.choice(available, size=count, replace=False)] = True
            occ[selected] = False
            duration += 1
            size += count
            rounds.append(dict(path_id=path_id, step=step, round=duration,
                               demand=demand, removed_nodes=count,
                               occupancy_after=float(occ.mean())))
            if count == 0:
                end_reason = "zero_removal"
                break
            if config.mode != "cascade":
                end_reason = "single_round"
                break
            if duration >= config.max_rounds:
                end_reason = "max_rounds"
                break
        if duration == 0:
            vacancies = np.flatnonzero(~occ.ravel())
            if vacancies.size:
                chosen = rng.choice(vacancies, size=min(config.drive_nodes, vacancies.size), replace=False)
                occ.ravel()[chosen] = True
                added = len(chosen)
        if duration:
            sequence += 1
            events.append(dict(path_id=path_id, step=step, event_sequence=sequence,
                size=size, duration=duration,
                waiting_steps=np.nan if last_event_step is None else step-last_event_step,
                occupancy_before=before/n, occupancy_after=float(occ.mean()),
                trigger_demand=first_demand, trigger_reason="absolute_demand_threshold",
                end_reason=end_reason))
            last_event_step = step
        rows.append(dict(path_id=path_id, step=step, occupancy_before=before/n,
            occupancy_after=float(occ.mean()), demand=first_demand,
            added_nodes=added, removed_nodes=size, duration=duration,
            event_sequence=sequence if duration else 0))
    return (pd.DataFrame(rows),
            pd.DataFrame(events, columns=["path_id", "step", "event_sequence", "size",
                "duration", "waiting_steps", "occupancy_before", "occupancy_after",
                "trigger_demand", "trigger_reason", "end_reason"]),
            pd.DataFrame(rounds, columns=["path_id", "step", "round", "demand",
                "removed_nodes", "occupancy_after"]))

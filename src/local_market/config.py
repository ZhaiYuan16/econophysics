"""Configuration objects for reproducible baseline experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .percolation import Boundary
from .statistics import ClusterExclusion


@dataclass(frozen=True)
class BaselineConfig:
    length: int
    p: float
    activity: float
    market_depth: float
    boundary: Boundary
    mode: str
    steps: int
    seed: int

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("p must lie in [0, 1]")
        if not 0.0 <= self.activity <= 1.0:
            raise ValueError("activity must lie in [0, 1]")
        if self.market_depth <= 0:
            raise ValueError("market_depth must be positive")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        if self.mode not in ("quenched", "annealed"):
            raise ValueError("mode must be 'quenched' or 'annealed'")
        if self.steps < 0:
            raise ValueError("steps must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaselineConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CriticalityConfig:
    lengths: tuple[int, ...]
    p_c: float
    p_ratios: tuple[float, ...]
    samples_per_length: dict[int, int]
    boundary: Boundary
    finite_cluster_exclusion: ClusterExclusion
    histogram_p_ratios: tuple[float, ...]
    bootstrap_replicates: int
    seed: int

    def __post_init__(self) -> None:
        if not self.lengths or any(length <= 0 for length in self.lengths):
            raise ValueError("lengths must contain positive integers")
        if not 0.0 < self.p_c < 1.0:
            raise ValueError("p_c must lie in (0, 1)")
        if not self.p_ratios or any(
            not 0.0 < ratio * self.p_c <= 1.0 for ratio in self.p_ratios
        ):
            raise ValueError("all p ratios must produce probabilities in (0, 1]")
        if sorted(set(self.p_ratios)) != list(self.p_ratios):
            raise ValueError("p_ratios must be sorted and unique")
        if set(self.samples_per_length) != set(self.lengths):
            raise ValueError("samples_per_length must define every configured length")
        if any(samples <= 0 for samples in self.samples_per_length.values()):
            raise ValueError("sample counts must be positive")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("boundary must be 'open' or 'periodic'")
        if self.finite_cluster_exclusion not in ("largest", "spanning", "none"):
            raise ValueError("invalid finite-cluster exclusion rule")
        if not set(self.histogram_p_ratios).issubset(self.p_ratios):
            raise ValueError("histogram ratios must be included in p_ratios")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CriticalityConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        data["lengths"] = tuple(int(value) for value in data["lengths"])
        data["p_ratios"] = tuple(float(value) for value in data["p_ratios"])
        data["histogram_p_ratios"] = tuple(
            float(value) for value in data["histogram_p_ratios"]
        )
        data["samples_per_length"] = {
            int(length): int(samples)
            for length, samples in data["samples_per_length"].items()
        }
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MarketAnalysisConfig:
    length: int
    p_c: float
    p_ratios: tuple[float, ...]
    activities: tuple[float, ...]
    replicates: int
    configurations_per_replicate: int
    trades_per_configuration: int
    market_depth: float
    boundary: Boundary
    extreme_thresholds: tuple[float, ...]
    tail_xmin_quantiles: tuple[float, ...]
    tail_minimum_count: int
    bootstrap_replicates: int
    length_sensitivity_fractions: tuple[float, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not 0.0 < self.p_c < 1.0:
            raise ValueError("p_c must lie in (0, 1)")
        if not self.p_ratios or any(
            not 0.0 < ratio * self.p_c <= 1.0 for ratio in self.p_ratios
        ):
            raise ValueError("invalid p ratios")
        if not self.activities or any(not 0.0 <= value <= 1.0 for value in self.activities):
            raise ValueError("activities must lie in [0, 1]")
        if min(
            self.replicates,
            self.configurations_per_replicate,
            self.trades_per_configuration,
        ) <= 0:
            raise ValueError("replicate and sample counts must be positive")
        if self.market_depth <= 0:
            raise ValueError("market_depth must be positive")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("invalid boundary")
        if any(value <= 0 for value in self.extreme_thresholds):
            raise ValueError("extreme thresholds must be positive")
        if not {3.0, 5.0, 8.0}.issubset(self.extreme_thresholds):
            raise ValueError("extreme thresholds must include 3, 5, and 8")
        if any(not 0.0 < value < 1.0 for value in self.tail_xmin_quantiles):
            raise ValueError("tail xmin quantiles must lie in (0, 1)")
        if self.tail_minimum_count < 50 or self.bootstrap_replicates < 100:
            raise ValueError("tail and bootstrap sample counts are too small")
        if any(not 0.0 < value <= 1.0 for value in self.length_sensitivity_fractions):
            raise ValueError("length fractions must lie in (0, 1]")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MarketAnalysisConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        for key in (
            "p_ratios",
            "activities",
            "extreme_thresholds",
            "tail_xmin_quantiles",
            "length_sensitivity_fractions",
        ):
            data[key] = tuple(float(value) for value in data[key])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NetworkAnalysisConfig:
    node_count: int
    target_mean_degree: int
    topologies: tuple[str, ...]
    network_replicates: int
    occupancy_samples_per_p: int
    p_grids: dict[str, tuple[float, ...]]
    relative_p_values: tuple[float, ...]
    market_configurations_per_network: int
    trades_per_configuration: int
    activity: float
    market_depth: float
    ws_rewiring_probability: float
    path_sources: int
    extreme_thresholds: tuple[float, ...]
    tail_xmin_quantiles: tuple[float, ...]
    tail_minimum_count: int
    bootstrap_replicates: int
    seed: int

    def __post_init__(self) -> None:
        required = {"lattice", "er", "ws", "ba"}
        if set(self.topologies) != required:
            raise ValueError("topologies must contain lattice, er, ws, and ba")
        if self.node_count <= 0 or self.target_mean_degree <= 0:
            raise ValueError("node_count and target_mean_degree must be positive")
        if min(
            self.network_replicates,
            self.occupancy_samples_per_p,
            self.market_configurations_per_network,
            self.trades_per_configuration,
            self.path_sources,
        ) <= 0:
            raise ValueError("sample counts must be positive")
        if set(self.p_grids) != required:
            raise ValueError("p_grids must define every topology")
        for grid in self.p_grids.values():
            if sorted(set(grid)) != list(grid) or any(not 0.0 < p <= 1.0 for p in grid):
                raise ValueError("p grids must be sorted, unique, and within (0, 1]")
        if any(value <= 0 for value in self.relative_p_values):
            raise ValueError("relative p values must be positive")
        if not 0.0 <= self.activity <= 1.0 or self.market_depth <= 0:
            raise ValueError("invalid market activity or depth")
        if not 0.0 <= self.ws_rewiring_probability <= 1.0:
            raise ValueError("WS rewiring probability must lie in [0, 1]")
        if not {3.0, 5.0, 8.0}.issubset(self.extreme_thresholds):
            raise ValueError("extreme thresholds must include 3, 5, and 8")
        if any(not 0.0 < q < 1.0 for q in self.tail_xmin_quantiles):
            raise ValueError("tail xmin quantiles must lie in (0, 1)")
        if self.tail_minimum_count < 50 or self.bootstrap_replicates < 100:
            raise ValueError("tail settings are too small")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NetworkAnalysisConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        data["topologies"] = tuple(data["topologies"])
        data["p_grids"] = {
            name: tuple(float(value) for value in grid)
            for name, grid in data["p_grids"].items()
        }
        for key in (
            "relative_p_values",
            "extreme_thresholds",
            "tail_xmin_quantiles",
        ):
            data[key] = tuple(float(value) for value in data[key])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BAThresholdConfig:
    node_counts: tuple[int, ...]
    target_mean_degree: int
    network_replicates: int
    occupancy_samples_per_p: int
    p_grid: tuple[float, ...]
    bootstrap_replicates: int
    order_parameter_level: float
    seed: int

    def __post_init__(self) -> None:
        if len(self.node_counts) < 3 or any(value <= 0 for value in self.node_counts):
            raise ValueError("node_counts must contain at least three positive sizes")
        if sorted(set(self.node_counts)) != list(self.node_counts):
            raise ValueError("node_counts must be sorted and unique")
        if self.target_mean_degree <= 0:
            raise ValueError("target_mean_degree must be positive")
        if self.network_replicates < 4 or self.occupancy_samples_per_p < 10:
            raise ValueError("too few network or occupancy replicates")
        if len(self.p_grid) < 7 or any(not 0.0 < p < 1.0 for p in self.p_grid):
            raise ValueError("p_grid must contain at least seven probabilities")
        if sorted(set(self.p_grid)) != list(self.p_grid):
            raise ValueError("p_grid must be sorted and unique")
        spacing = [b - a for a, b in zip(self.p_grid, self.p_grid[1:])]
        if max(spacing) - min(spacing) > 1e-10:
            raise ValueError("p_grid must be uniformly spaced for slope estimation")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if not 0.0 < self.order_parameter_level < 1.0:
            raise ValueError("order_parameter_level must lie in (0, 1)")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BAThresholdConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        data["node_counts"] = tuple(int(value) for value in data["node_counts"])
        data["p_grid"] = tuple(float(value) for value in data["p_grid"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SOCConfig:
    length: int
    initial_occupancies: tuple[float, ...]
    activity: float
    market_depth: float
    boundary: Boundary
    steps: int
    burn_in: int
    replicates: int
    drive_nodes: int
    event_threshold_fraction: float
    exit_probability: float
    critical_reference: float
    critical_band_relative_width: float
    tail_minimum_count: int
    bootstrap_replicates: int
    seed: int

    def __post_init__(self) -> None:
        if self.length <= 1:
            raise ValueError("length must exceed one")
        if not self.initial_occupancies or any(
            not 0.0 <= value <= 1.0 for value in self.initial_occupancies
        ):
            raise ValueError("initial occupancies must lie in [0, 1]")
        if len(set(self.initial_occupancies)) != len(self.initial_occupancies):
            raise ValueError("initial occupancies must be unique")
        if not 0.0 <= self.activity <= 1.0 or self.market_depth <= 0:
            raise ValueError("invalid market activity or depth")
        if self.boundary not in ("open", "periodic"):
            raise ValueError("invalid boundary")
        if self.steps <= 0 or not 0 <= self.burn_in < self.steps:
            raise ValueError("burn_in must lie in [0, steps)")
        if self.replicates < 2 or self.drive_nodes <= 0:
            raise ValueError("replicates must be >=2 and drive_nodes positive")
        if not 0.0 < self.event_threshold_fraction < 1.0:
            raise ValueError("event threshold fraction must lie in (0, 1)")
        if not 0.0 < self.exit_probability <= 1.0:
            raise ValueError("exit probability must lie in (0, 1]")
        if not 0.0 < self.critical_reference < 1.0:
            raise ValueError("critical reference must lie in (0, 1)")
        if not 0.0 < self.critical_band_relative_width < 1.0:
            raise ValueError("critical band width must lie in (0, 1)")
        if self.tail_minimum_count < 50 or self.bootstrap_replicates < 100:
            raise ValueError("tail settings are too small")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SOCConfig":
        with Path(path).open(encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        data["initial_occupancies"] = tuple(
            float(value) for value in data["initial_occupancies"]
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

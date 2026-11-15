#!/usr/bin/env python3
"""Run the stage-1 three-point validation without a full parameter scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from local_market.cb_model import simulate_quenched
from local_market.networks import sample_lattice_occupancy
from local_market.percolation import identify_lattice_clusters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    length = int(config["length"])
    activity = float(config["activity"])
    market_depth = float(config["market_depth"])
    boundary = config["boundary"]
    configurations_per_p = int(config["configurations_per_p"])
    trades_per_configuration = int(config["trades_per_configuration"])
    rng = np.random.default_rng(int(config["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows: list[dict[str, object]] = []
    return_rows: list[dict[str, object]] = []
    for p in map(float, config["p_values"]):
        for sample in range(configurations_per_p):
            occupied = sample_lattice_occupancy(length, p, rng)
            clusters = identify_lattice_clusters(occupied, boundary)
            cluster_rows.append(
                {
                    "p": p,
                    "sample": sample,
                    "occupied_count": clusters.occupied_count,
                    "cluster_count": clusters.cluster_count,
                    "largest_size": clusters.largest_size,
                    "largest_occupied_fraction": clusters.largest_occupied_fraction,
                    "spans": clusters.spans,
                }
            )
            simulation = simulate_quenched(
                clusters,
                steps=trades_per_configuration,
                activity=activity,
                market_depth=market_depth,
                rng=rng,
            )
            return_rows.extend(
                {
                    "p": p,
                    "sample": sample,
                    "trade": trade,
                    "demand": int(demand),
                    "return": float(return_),
                }
                for trade, (demand, return_) in enumerate(
                    zip(simulation.demands, simulation.returns, strict=True)
                )
            )

    clusters_frame = pd.DataFrame(cluster_rows)
    returns_frame = pd.DataFrame(return_rows)
    summary = (
        clusters_frame.groupby("p", as_index=False)
        .agg(
            mean_occupied=("occupied_count", "mean"),
            mean_cluster_count=("cluster_count", "mean"),
            mean_largest_size=("largest_size", "mean"),
            mean_largest_occupied_fraction=("largest_occupied_fraction", "mean"),
            spanning_frequency=("spans", "mean"),
        )
        .merge(
            returns_frame.groupby("p", as_index=False).agg(
                return_mean=("return", "mean"),
                return_std=("return", "std"),
                nonzero_return_fraction=("return", lambda values: (values != 0).mean()),
            ),
            on="p",
        )
    )

    clusters_path = args.output_dir / "validation_small_clusters.csv"
    returns_path = args.output_dir / "validation_small_returns.csv"
    summary_path = args.output_dir / "validation_small_summary.csv"
    metadata_path = args.output_dir / "validation_small_metadata.json"
    clusters_frame.to_csv(clusters_path, index=False)
    returns_frame.to_csv(returns_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()

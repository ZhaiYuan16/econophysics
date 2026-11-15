#!/usr/bin/env python3
"""Run the optional driven-dissipative CB extension."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from local_market.config import SOCConfig
from local_market.plotting import plot_soc_dynamics
from local_market.soc import run_soc_ensemble
from local_market.tail_fitting import (
    compare_discrete_tail_distributions,
    discrete_powerlaw_mle,
    discrete_powerlaw_fixed_xmin_gof_pvalue,
    select_discrete_powerlaw_tail,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SOCConfig.from_yaml(args.config)
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    paths, replicates, summary = run_soc_ensemble(
        config, show_progress=not args.quiet
    )
    output_paths = {
        "paths": tables / "soc_paths.csv.gz",
        "replicates": tables / "soc_replicates.csv",
        "summary": tables / "soc_summary.csv",
        "event_tail": tables / "soc_event_tail_fit.csv",
        "figure": figures / "soc_dynamics.png",
    }
    paths.to_csv(output_paths["paths"], index=False, compression="gzip")
    replicates.to_csv(output_paths["replicates"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    event_sizes = paths.loc[
        (paths["step"] >= config.burn_in) & (paths["removed_nodes"] > 0),
        "removed_nodes",
    ].to_numpy(dtype=np.int64)
    event_fit = select_discrete_powerlaw_tail(
        event_sizes, config.tail_minimum_count
    )
    tail_rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, 601])
    )
    stationary_events = paths[(paths["step"] >= config.burn_in) & (paths["removed_nodes"] > 0)]
    path_sizes = [
        [stationary_events.loc[stationary_events.path_id == path_id,
                               "removed_nodes"].to_numpy(dtype=np.int64)
         for path_id in sorted(initial_group.path_id.unique())]
        for _, initial_group in paths.groupby("initial_occupancy")
    ]
    alpha_draws = np.empty(config.bootstrap_replicates, dtype=float)
    for index in range(config.bootstrap_replicates):
        sample = np.concatenate([
            paths_for_initial[path]
            for paths_for_initial in path_sizes
            for path in tail_rng.integers(0, len(paths_for_initial), size=len(paths_for_initial))
        ])
        alpha_draws[index] = discrete_powerlaw_mle(sample, int(event_fit.xmin))[0]
    alpha_se = float(alpha_draws.std(ddof=1))
    alpha_low, alpha_high = np.quantile(alpha_draws, [.025, .975])
    gof = discrete_powerlaw_fixed_xmin_gof_pvalue(
        event_fit, config.bootstrap_replicates, tail_rng
    )
    event_tail = pd.DataFrame(
        [
            {
                "event_definition": "nodes removed after an extreme-demand event",
                "xmin": int(event_fit.xmin),
                "alpha_pdf": event_fit.alpha_pdf,
                "ks_distance": event_fit.ks_distance,
                "tail_count": event_fit.tail_count,
                "tail_fraction": event_fit.tail_fraction,
                "alpha_bootstrap_se": alpha_se,
                "alpha_ci_low": alpha_low,
                "alpha_ci_high": alpha_high,
                "fixed_xmin_gof_p": gof,
            }
            | compare_discrete_tail_distributions(
                event_sizes, int(event_fit.xmin)
            )
        ]
    )
    event_tail.to_csv(output_paths["event_tail"], index=False)
    plot_soc_dynamics(
        paths,
        summary,
        config.burn_in,
        config.critical_reference,
        config.critical_band_relative_width,
        output_paths["figure"],
    )
    metadata = {
        "config": config.to_dict(),
        "mechanism": {
            "drive": "quiet steps add up to drive_nodes randomly selected vacant sites",
            "event": "absolute excess demand exceeds event_threshold_fraction times N",
            "dissipation": (
                "on an event, nodes in active clusters aligned with net demand "
                "exit independently with exit_probability"
            ),
        },
        "interpretation": (
            "optional extension separate from the externally tuned CB experiments; "
            "finite-size attraction to a critical band is evidence, not proof, of SOC"
        ),
        "event_uncertainty": (
            "event-size alpha interval resamples complete evolution paths at fixed xmin; "
            "the retained fixed_xmin_gof_p uses an iid-event parametric null and is "
            "not a path-aware significance test"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }
    (tables / "soc_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(event_tail.to_string(index=False))
    print(f"saved optional SOC extension under {args.output_root}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Scan square-lattice site percolation around the critical point."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from local_market.config import CriticalityConfig
from local_market.experiments import estimate_finite_size_scaling, run_criticality_scan
from local_market.plotting import (
    plot_cluster_distributions,
    plot_critical_observables,
    plot_finite_size_scaling,
    plot_pseudocritical_estimates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CriticalityConfig.from_yaml(args.config)
    tables = args.output_root / "tables"
    figures = args.output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    samples, summary, histogram, estimates = run_criticality_scan(
        config, show_progress=not args.quiet
    )
    scaling = estimate_finite_size_scaling(summary)
    paths = {
        "samples": tables / "criticality_samples.csv",
        "summary": tables / "criticality_summary.csv",
        "histogram": tables / "criticality_cluster_histogram.csv",
        "estimates": tables / "criticality_pseudocritical.csv",
        "scaling": tables / "criticality_scaling.csv",
    }
    samples.to_csv(paths["samples"], index=False)
    summary.to_csv(paths["summary"], index=False)
    histogram.to_csv(paths["histogram"], index=False)
    estimates.to_csv(paths["estimates"], index=False)
    scaling.to_csv(paths["scaling"], index=False)

    plot_critical_observables(
        summary, config.p_c, figures / "criticality_observables.png"
    )
    plot_cluster_distributions(
        histogram,
        config.histogram_p_ratios,
        config.finite_cluster_exclusion,
        figures / "criticality_cluster_distributions.png",
    )
    plot_pseudocritical_estimates(
        estimates, config.p_c, figures / "criticality_pseudocritical.png"
    )
    plot_finite_size_scaling(
        summary,
        scaling,
        1.0,
        figures / "criticality_finite_size_scaling.png",
    )

    metadata = {
        "config": config.to_dict(),
        "definitions": {
            "p_infinity": "mean(largest_size / total_lattice_sites), zero if empty",
            "susceptibility_pooled": (
                "sum over realizations and retained clusters of s^2 divided by "
                "the corresponding sum of s"
            ),
            "finite_cluster_exclusion": config.finite_cluster_exclusion,
            "pseudocritical": "discrete scan-grid locations; no interpolation",
            "cluster_histogram": "n_s is cluster count per lattice site and realization",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    (tables / "criticality_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(estimates.to_string(index=False))
    print(scaling.to_string(index=False))
    print(f"saved stage-2 outputs under {args.output_root}")


if __name__ == "__main__":
    main()

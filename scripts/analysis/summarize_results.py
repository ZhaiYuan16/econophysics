#!/usr/bin/env python3
"""Validate stage outputs and create one machine-readable final summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    tables = args.output_root / "tables"
    required = {
        "criticality": tables / "criticality_pseudocritical.csv",
        "scaling": tables / "criticality_scaling.csv",
        "cluster_tail": tables / "criticality_cluster_tail_fits.csv",
        "ba_criticality": tables / "ba_criticality_criteria.csv",
        "soc": tables / "soc_summary.csv",
        "soc_tail": tables / "soc_event_tail_fit.csv",
        "market": tables / "market_summary.csv",
        "market_tail": tables / "market_tail_fits.csv",
        "network": tables / "network_metric_summary.csv",
        "network_market": tables / "network_market_summary.csv",
        "network_tail": tables / "network_tail_fits.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required result files: " + ", ".join(missing))

    frames = {name: pd.read_csv(path) for name, path in required.items()}
    if any(frame.empty for frame in frames.values()):
        raise ValueError("a required result table is empty")
    if any(frame.isna().any().any() for frame in frames.values()):
        raise ValueError("a required result table contains missing values")

    rows: list[dict[str, object]] = []

    def add(
        section: str,
        group: str,
        metric: str,
        value: float,
        unit: str,
        source_key: str,
    ) -> None:
        rows.append(
            {
                "section": section,
                "group": group,
                "metric": metric,
                "value": float(value),
                "unit": unit,
                "source": str(required[source_key].relative_to(args.output_root)),
            }
        )

    for row in frames["criticality"].itertuples(index=False):
        group = f"L={int(row.length)}"
        add("criticality", group, "p_chi_peak", row.p_chi_peak, "probability", "criticality")
        add("criticality", group, "chi_peak", row.chi_peak, "nodes", "criticality")
        add(
            "criticality",
            group,
            "p_max_order_parameter_slope",
            row.p_max_order_parameter_slope,
            "probability",
            "criticality",
        )
    for row in frames["scaling"].itertuples(index=False):
        add(
            "finite_size_scaling",
            str(row.observable),
            "fitted_exponent",
            row.fitted_exponent,
            "dimensionless",
            "scaling",
        )

    for row in frames["cluster_tail"].itertuples(index=False):
        group = f"L={int(row.length)}"
        for metric, unit in (
            ("xmin", "nodes"),
            ("alpha_pdf", "dimensionless"),
            ("alpha_ci_low", "dimensionless"),
            ("alpha_ci_high", "dimensionless"),
            ("fixed_xmin_gof_p", "probability"),
        ):
            add("critical_cluster_tail", group, metric, getattr(row, metric), unit, "cluster_tail")

    for row in frames["ba_criticality"].itertuples(index=False):
        group = f"N={int(row.node_count)}"
        for metric in (
            "p_chi_peak",
            "p_chi_peak_ci_low",
            "p_chi_peak_ci_high",
            "p_smoothed_slope_peak",
            "p_order_parameter_level",
            "molloy_reed_pc",
        ):
            add("ba_threshold_sensitivity", group, metric, getattr(row, metric), "probability", "ba_criticality")

    for row in frames["soc"].itertuples(index=False):
        group = f"initial_p={row.initial_occupancy:g}"
        for metric, unit in (
            ("mean_occupancy", "fraction"),
            ("stationary_drift", "fraction"),
            ("critical_band_fraction", "fraction"),
            ("event_rate", "events_per_step"),
            ("mean_event_removal", "nodes"),
            ("return_excess_kurtosis", "dimensionless"),
        ):
            add("soc_extension", group, metric, getattr(row, metric), unit, "soc")
    soc_tail = frames["soc_tail"].iloc[0]
    for metric in (
        "xmin",
        "alpha_pdf",
        "alpha_ci_low",
        "alpha_ci_high",
        "fixed_xmin_gof_p",
        "loglikelihood_ratio_power_vs_exponential",
    ):
        add("soc_event_tail", "all_initial_states", metric, soc_tail[metric], "dimensionless", "soc_tail")

    market = frames["market"]
    market_pc = market[np.isclose(market["p_ratio"], 1.0)]
    for row in market_pc.itertuples(index=False):
        group = f"a={row.activity:g}"
        for metric, unit in (
            ("return_excess_kurtosis", "dimensionless"),
            ("zero_return_fraction", "probability"),
            ("p_abs_z_gt_5", "probability"),
            ("negative_crash_probability", "probability"),
            ("mean_active_cluster_count", "clusters"),
        ):
            add("market_at_pc", group, metric, getattr(row, metric), unit, "market")

    market_tail = frames["market_tail"]
    selected_market_tail = market_tail[
        np.isclose(market_tail["p_ratio"], 1.0)
        & np.isclose(market_tail["activity"], 0.01)
    ].iloc[0]
    for metric in ("xmin", "alpha_pdf", "alpha_ci_low", "alpha_ci_high", "pareto_fixed_xmin_gof_p"):
        add("market_tail_at_pc", "a=0.01", metric, selected_market_tail[metric], "dimensionless", "market_tail")

    for row in frames["network"].itertuples(index=False):
        group = str(row.topology)
        for metric, unit in (
            ("mean_degree", "edges_per_node"),
            ("clustering_coefficient", "dimensionless"),
            ("largest_connected_component_fraction", "fraction"),
            ("average_shortest_path_length_approx", "edges"),
            ("effective_pc", "probability"),
        ):
            add("network_structure", group, metric, getattr(row, metric), unit, "network")

    network_market = frames["network_market"]
    network_pc = network_market[np.isclose(network_market["relative_p"], 1.0)]
    for row in network_pc.itertuples(index=False):
        group = str(row.topology)
        for metric, unit in (
            ("return_excess_kurtosis", "dimensionless"),
            ("p_abs_z_gt_5", "probability"),
            ("p_abs_z_gt_8", "probability"),
            ("negative_crash_probability", "probability"),
        ):
            add("network_market_at_pc", group, metric, getattr(row, metric), unit, "network_market")

    network_tail = frames["network_tail"]
    network_tail_pc = network_tail[np.isclose(network_tail["relative_p"], 1.0)]
    for row in network_tail_pc.itertuples(index=False):
        group = str(row.topology)
        for metric in ("alpha_pdf", "alpha_ci_low", "alpha_ci_high", "pareto_fixed_xmin_gof_p"):
            add("network_tail_at_pc", group, metric, getattr(row, metric), "dimensionless", "network_tail")

    summary = pd.DataFrame(rows)
    summary_path = tables / "final_summary.csv"
    summary.to_csv(summary_path, index=False)

    artifact_paths = sorted(
        path
        for directory in (tables, args.output_root / "figures")
        if directory.exists()
        for path in directory.iterdir()
        if path.is_file() and path.name not in {"reproduction_manifest.json"}
    )
    manifest = {
        "generated_by": "scripts/summarize_results.py",
        "python": platform.python_version(),
        "validation": {
            "required_tables_present": True,
            "required_tables_nonempty": True,
            "required_tables_without_missing_values": True,
        },
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
    }
    manifest_path = tables / "reproduction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"validated {len(required)} required tables")
    print(f"wrote {len(summary)} summary rows to {summary_path}")
    print(f"wrote {len(artifact_paths)} artifact hashes to {manifest_path}")


if __name__ == "__main__":
    main()

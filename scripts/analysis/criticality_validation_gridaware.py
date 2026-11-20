"""Propagate discrete peak-cell uncertainty into WLS extrapolation intervals."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("results/criticality_revision_20261113/full/tables")
OUT = Path("results/criticality_validation_20261113")
PC = 0.592746
LENGTHS = (32, 64, 128, 256, 512)


def jitter_peak_cells(draws: np.ndarray, grid: np.ndarray, rng) -> np.ndarray:
    """Uniformly locate a grid maximum within its neighbouring midpoint cell."""
    indices = np.searchsorted(grid, draws)
    if not np.all(np.isclose(grid[indices], draws)):
        raise ValueError("bootstrap peak outside the stored scan grid")
    low = np.where(indices > 0, (grid[np.maximum(indices - 1, 0)] + draws) / 2,
                   draws - (grid[1] - grid[0]) / 2)
    high = np.where(indices < len(grid) - 1,
                    (grid[np.minimum(indices + 1, len(grid) - 1)] + draws) / 2,
                    draws + (grid[-1] - grid[-2]) / 2)
    return rng.uniform(low, high)


def main():
    destination = OUT / "pseudocritical_extrapolation_gridaware.csv"
    if destination.exists():
        raise FileExistsError(destination)
    peaks = pd.read_csv(SOURCE / "pseudocritical.csv")
    summary = pd.read_csv(SOURCE / "summary.csv")
    boots = pd.read_csv(OUT / "bootstrap_peak_replicates.csv")
    original = pd.read_csv(OUT / "pseudocritical_extrapolation.csv")
    rng = np.random.default_rng(20261113 + 811)
    rows = []
    for rule in ("largest_excluded", "all_spanning_excluded"):
        grid_values = []
        for length in LENGTHS:
            grid = np.sort(summary[summary.length == length].p.to_numpy())
            draws = boots[f"L{length}_{rule}"].to_numpy()
            grid_values.append((grid, draws))
        for subset, indices in (("all_5", np.arange(5)), ("large_3", np.arange(2, 5))):
            lengths = np.array(LENGTHS)[indices]
            x = lengths.astype(float)**(-.75)
            observed = np.array([peaks[(peaks.length == length) &
                (peaks.exclusion == rule)].p_grid_peak.iloc[0] for length in lengths])
            spread = np.array([grid_values[i][1].std(ddof=1) for i in indices])
            grid_floor = np.array([np.diff(grid_values[i][0]).min() / np.sqrt(12)
                for i in indices])
            weight = 1 / (spread**2 + grid_floor**2)
            X = np.column_stack([np.ones(len(x)), x])
            operator = np.linalg.solve(X.T @ (weight[:, None] * X), X.T * weight)
            sampled_rows = rng.integers(0, len(boots), 5000)
            jittered = np.column_stack([
                jitter_peak_cells(grid_values[i][1][sampled_rows], grid_values[i][0], rng)
                for i in indices])
            coefficients = (operator @ jittered.T).T
            intervals = np.quantile(coefficients, [.025, .975], axis=0)
            old = original[(original.exclusion == rule) & (original.subset == subset)].iloc[0]
            rows.append({"exclusion": rule, "subset": subset,
                "point_intercept": (operator @ observed)[0],
                "gridaware_intercept_ci_low": intervals[0, 0],
                "gridaware_intercept_ci_high": intervals[1, 0],
                "point_slope": (operator @ observed)[1],
                "gridaware_slope_ci_low": intervals[0, 1],
                "gridaware_slope_ci_high": intervals[1, 1],
                "config_only_intercept_ci_low": old.intercept_ci_low,
                "config_only_intercept_ci_high": old.intercept_ci_high,
                "theory_pc": PC, "draws": len(sampled_rows),
                "method": "configuration peak bootstrap + uniform neighboring grid midpoint cell"})
    pd.DataFrame(rows).to_csv(destination, index=False)
    (OUT / "gridaware_metadata.json").write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_simulations": False, "point_estimator": "stored discrete scan-grid peak",
        "grid_uncertainty": "uniform within adjacent scan-point midpoints"}, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())
                if p.is_file() and not p.name.startswith("sha256_manifest")}
    (OUT / "sha256_manifest_v2.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one small CB baseline experiment and save machine-readable output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from local_market.config import BaselineConfig
from local_market.experiments import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig.from_yaml(args.config)
    simulation = run_baseline(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"baseline_{config.mode}_L{config.length}_seed{config.seed}"
    frame = pd.DataFrame(
        {
            "step": range(config.steps),
            "demand": simulation.demands,
            "return": simulation.returns,
            "log_price": simulation.log_prices[1:],
            "occupied_count": simulation.occupied_counts,
            "cluster_count": simulation.cluster_counts,
        }
    )
    csv_path = args.output_dir / f"{stem}.csv"
    metadata_path = args.output_dir / f"{stem}.json"
    frame.to_csv(csv_path, index=False)
    metadata = config.to_dict() | {
        "output_csv": str(csv_path),
        "return_mean": float(simulation.returns.mean()) if config.steps else None,
        "return_std": float(simulation.returns.std(ddof=1)) if config.steps > 1 else None,
        "nonzero_return_fraction": (
            float((simulation.returns != 0).mean()) if config.steps else None
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved {csv_path}")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()

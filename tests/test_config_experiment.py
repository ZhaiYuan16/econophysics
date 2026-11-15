from pathlib import Path

import numpy as np

from local_market.config import BaselineConfig
from local_market.experiments import run_baseline


def test_yaml_configuration_loads_without_parameter_changes(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "length: 8\np: 0.4\nactivity: 0.1\nmarket_depth: 64\n"
        "boundary: open\nmode: quenched\nsteps: 12\nseed: 17\n",
        encoding="utf-8",
    )
    config = BaselineConfig.from_yaml(path)
    assert config.to_dict() == {
        "length": 8,
        "p": 0.4,
        "activity": 0.1,
        "market_depth": 64,
        "boundary": "open",
        "mode": "quenched",
        "steps": 12,
        "seed": 17,
    }


def test_configured_experiment_is_reproducible() -> None:
    config = BaselineConfig(8, 0.5, 0.1, 64.0, "open", "annealed", 30, 99)
    first = run_baseline(config)
    second = run_baseline(config)
    np.testing.assert_array_equal(first.demands, second.demands)
    np.testing.assert_array_equal(first.occupied_counts, second.occupied_counts)

import numpy as np

from local_market.config import SOCConfig
from local_market.soc import simulate_soc_path, run_soc_ensemble


def _config(**overrides: object) -> SOCConfig:
    values = {
        "length": 4,
        "initial_occupancies": (0.0, 1.0),
        "activity": 0.0,
        "market_depth": 16.0,
        "boundary": "open",
        "steps": 3,
        "burn_in": 1,
        "replicates": 2,
        "drive_nodes": 2,
        "event_threshold_fraction": 0.1,
        "exit_probability": 1.0,
        "critical_reference": 0.6,
        "critical_band_relative_width": 0.1,
        "tail_minimum_count": 50,
        "bootstrap_replicates": 100,
        "seed": 7,
    }
    values.update(overrides)
    return SOCConfig(**values)


def test_quiet_soc_drive_adds_configured_nodes() -> None:
    result = simulate_soc_path(_config(), 0.0, np.random.default_rng(5))
    np.testing.assert_allclose(result.frame["occupancy_after"], [2 / 16, 4 / 16, 6 / 16])
    np.testing.assert_array_equal(result.frame["added_nodes"], [2, 2, 2])
    assert not result.frame["event"].any()


def test_extreme_single_cluster_is_dissipated() -> None:
    config = _config(activity=1.0, steps=1, burn_in=0, drive_nodes=1)
    result = simulate_soc_path(config, 1.0, np.random.default_rng(8))
    assert bool(result.frame.loc[0, "event"])
    assert result.frame.loc[0, "removed_nodes"] == 16
    assert result.frame.loc[0, "occupancy_after"] == 0.0


def test_soc_path_is_reproducible() -> None:
    config = _config(activity=0.2, steps=20, burn_in=5)
    first = simulate_soc_path(config, 0.5, np.random.default_rng(11)).frame
    second = simulate_soc_path(config, 0.5, np.random.default_rng(11)).frame
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_soc_paths_keep_identity_time_and_event_order() -> None:
    config = _config(activity=1.0, steps=20, burn_in=5)
    paths, _, _ = run_soc_ensemble(config, show_progress=False)
    assert paths.path_id.nunique() == len(config.initial_occupancies) * config.replicates
    for _, path in paths.groupby("path_id"):
        np.testing.assert_array_equal(path.step, np.arange(config.steps))
        events = path[path.event]
        np.testing.assert_array_equal(events.event_sequence, np.arange(1, len(events) + 1))
        assert (path.loc[~path.event, "event_sequence"] == 0).all()

import numpy as np
import pandas as pd
import pytest
from scipy.signal import fftconvolve
from scipy.stats import chi2

from local_market.cb_model import sample_excess_demands, simulate_quenched
from local_market.percolation import identify_lattice_clusters
from local_market import plotting


def _lag_one(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / np.dot(x, x))


def _lag50_ljung_box_p(x: np.ndarray) -> float:
    centered = x - x.mean()
    covariance = fftconvolve(centered, centered[::-1], mode="full")[len(x)-1:len(x)+50]
    rho = covariance[1:] / covariance[0]
    n = len(x)
    statistic = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, 51)))
    return float(chi2.sf(statistic, 50))


def test_conditional_moments_and_independent_actions() -> None:
    sizes = np.array([1, 2, 5, 13, 29])
    activity, depth = 0.3, 50.0
    expected = activity * np.sum(sizes**2) / depth**2
    returns = sample_excess_demands(sizes, 100_000, activity, np.random.default_rng(412)) / depth
    assert abs(returns.mean()) < 4 * np.sqrt(expected / len(returns))
    assert abs(returns.var(ddof=1) / expected - 1) < 0.025
    assert abs(_lag_one(returns)) < 4 / np.sqrt(len(returns))
    assert abs(_lag_one(np.abs(returns))) < 4 / np.sqrt(len(returns))
    assert _lag50_ljung_box_p(returns) > .01
    assert _lag50_ljung_box_p(np.abs(returns)) > .01


def test_distinct_configurations_have_distinct_conditional_variances() -> None:
    rng = np.random.default_rng(413)
    small = sample_excess_demands(np.array([1, 1, 1]), 50_000, .5, rng)
    large = sample_excess_demands(np.array([10, 10, 10]), 50_000, .5, rng)
    assert 90 < large.var(ddof=1) / small.var(ddof=1) < 110


def test_ensemble_plot_has_no_cumulative_price_or_time_axis(tmp_path, monkeypatch) -> None:
    captured = {}
    def capture(fig, path):
        captured["axes"] = [ax.get_xlabel() for ax in fig.axes]
        captured["titles"] = [ax.get_title() for ax in fig.axes]
    monkeypatch.setattr(plotting, "_finish_figure", capture)
    data = pd.DataFrame({"activity": [.1, .1], "p_ratio": [1., 1.],
                         "replicate": [0, 0], "step": [0, 1], "return": [.1, -.1]})
    plotting.plot_market_ensemble_samples(data, tmp_path / "ensemble.png")
    assert len(captured["axes"]) == 1
    assert "not time" in captured["axes"][0]
    assert all("Cumulative" not in title for title in captured["titles"])
    with pytest.raises(TypeError):
        plotting.plot_market_time_series({.1: data}, tmp_path / "time.png")


def test_cumulative_log_return_only_for_ordered_trajectory() -> None:
    config = identify_lattice_clusters(np.array([[1, 1], [0, 1]], dtype=bool))
    simulation = simulate_quenched(config, 100, .5, 4., np.random.default_rng(19))
    np.testing.assert_allclose(np.diff(simulation.log_prices), simulation.returns)

import pandas as pd

from local_market.config import CriticalityConfig
from local_market.experiments import estimate_finite_size_scaling, run_criticality_scan


def test_tiny_criticality_scan_is_reproducible_and_well_formed() -> None:
    config = CriticalityConfig(
        lengths=(8, 12),
        p_c=0.592746,
        p_ratios=(0.9, 1.0, 1.1),
        samples_per_length={8: 4, 12: 3},
        boundary="open",
        finite_cluster_exclusion="largest",
        histogram_p_ratios=(1.0,),
        bootstrap_replicates=100,
        seed=44,
    )
    first = run_criticality_scan(config, show_progress=False)
    second = run_criticality_scan(config, show_progress=False)
    for first_frame, second_frame in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_frame, second_frame)

    samples, summary, histogram, estimates = first
    assert len(samples) == (4 + 3) * 3
    assert len(summary) == 2 * 3
    assert set(estimates["length"]) == {8, 12}
    assert (summary["p_infinity"].between(0, 1)).all()
    assert (summary["susceptibility_pooled"] >= 0).all()
    assert (
        summary["susceptibility_ci_low"]
        <= summary["susceptibility_pooled"]
    ).all()
    assert (
        summary["susceptibility_pooled"]
        <= summary["susceptibility_ci_high"]
    ).all()
    assert (histogram["cluster_size"] > 0).all()


def test_scaling_fit_recovers_known_power() -> None:
    lengths = pd.Series([8, 16, 32, 64], dtype=float)
    summary = pd.DataFrame(
        {
            "length": lengths,
            "p_ratio": 1.0,
            "p_infinity": 2.0 * lengths ** (-5 / 48),
            "susceptibility_pooled": 0.25 * lengths ** (43 / 24),
        }
    )
    scaling = estimate_finite_size_scaling(summary)
    assert abs(scaling.loc[0, "fitted_exponent"] + 5 / 48) < 1e-12
    assert abs(scaling.loc[1, "fitted_exponent"] - 43 / 24) < 1e-12

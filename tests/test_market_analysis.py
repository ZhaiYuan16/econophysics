import numpy as np

from local_market.config import MarketAnalysisConfig
from local_market.experiments import run_market_ensemble, summarize_market_returns
from local_market.statistics import return_statistics, standardize_returns
from local_market.tail_fitting import (
    bootstrap_discrete_powerlaw_alpha,
    compare_tail_distributions,
    discrete_powerlaw_fixed_xmin_gof_pvalue,
    discrete_powerlaw_mle,
    pareto_mle,
    pareto_fixed_xmin_gof_pvalue,
    select_pareto_tail,
    select_discrete_powerlaw_tail,
)


def test_standardization_and_extreme_statistics() -> None:
    returns = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    z, mean, standard_deviation = standardize_returns(returns)
    assert abs(mean) < 1e-12
    assert np.isclose(z.std(ddof=1), 1.0)
    result = return_statistics(returns, z, (3.0, 5.0, 8.0))
    assert result["observations"] == 5
    assert result["zero_return_fraction"] == 0.2
    assert standard_deviation > 0


def test_pareto_mle_recovers_synthetic_pdf_exponent() -> None:
    rng = np.random.default_rng(71)
    values = rng.pareto(2.5, 50_000) + 1.0  # PDF exponent alpha=3.5
    alpha, ks, count = pareto_mle(values, 1.0)
    assert abs(alpha - 3.5) < 0.08
    assert ks < 0.02
    assert count == values.size
    selected, candidates = select_pareto_tail(
        values, (0.5, 0.7, 0.8, 0.9), minimum_count=1000
    )
    assert abs(selected.alpha_pdf - 3.5) < 0.12
    assert len(candidates) == 4


def test_distribution_comparison_does_not_cap_alpha_at_three() -> None:
    values = np.random.default_rng(9).pareto(5.0, 5_000) + 1.0
    comparison = compare_tail_distributions(values, 1.0)
    assert comparison["powerlaw_package_alpha_pdf"] > 4.5


def test_fixed_xmin_goodness_of_fit_is_a_probability() -> None:
    values = np.random.default_rng(11).pareto(2.0, 3_000) + 1.0
    fit, _ = select_pareto_tail(values, (0.5,), minimum_count=500)
    pvalue = pareto_fixed_xmin_gof_pvalue(
        fit, 100, np.random.default_rng(12)
    )
    assert 0.0 <= pvalue <= 1.0


def test_discrete_powerlaw_tools_recover_zipf_exponent() -> None:
    rng = np.random.default_rng(121)
    values = rng.zipf(2.2, 100_000)
    alpha, ks, count = discrete_powerlaw_mle(values, 5)
    assert abs(alpha - 2.2) < 0.08
    assert ks < 0.03
    assert count == np.count_nonzero(values >= 5)
    selected = select_discrete_powerlaw_tail(values, minimum_count=500)
    assert isinstance(selected.xmin, int)
    assert selected.tail_count >= 500
    from local_market.tail_fitting import TailFit

    fit = TailFit(5, alpha, ks, count, count / values.size)
    standard_error, low, high = bootstrap_discrete_powerlaw_alpha(
        values, 5, 100, np.random.default_rng(122)
    )
    assert standard_error > 0
    assert low < alpha < high
    pvalue = discrete_powerlaw_fixed_xmin_gof_pvalue(
        fit, 100, np.random.default_rng(123)
    )
    assert 0.0 <= pvalue <= 1.0


def test_tiny_market_ensemble_has_expected_shape_and_reproducibility() -> None:
    config = MarketAnalysisConfig(
        length=12,
        p_c=0.592746,
        p_ratios=(1.0,),
        activities=(0.5, 1.0),
        replicates=2,
        configurations_per_replicate=3,
        trades_per_configuration=20,
        market_depth=144.0,
        boundary="open",
        extreme_thresholds=(3.0, 5.0, 8.0),
        tail_xmin_quantiles=(0.7, 0.8),
        tail_minimum_count=50,
        bootstrap_replicates=100,
        length_sensitivity_fractions=(0.5, 1.0),
        seed=91,
    )
    first_returns, first_structures = run_market_ensemble(config, show_progress=False)
    second_returns, second_structures = run_market_ensemble(config, show_progress=False)
    assert len(first_returns) == 2 * 2 * 3 * 20
    assert len(first_structures) == 2 * 3
    np.testing.assert_array_equal(first_returns["demand"], second_returns["demand"])
    np.testing.assert_array_equal(
        first_structures["largest_size"], second_structures["largest_size"]
    )
    standardized, summary, replicates = summarize_market_returns(first_returns, config)
    assert standardized["z"].notna().all()
    assert len(summary) == 2
    assert len(replicates) == 4

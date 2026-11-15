import numpy as np
from local_market.soc_research import ResearchConfig, simulate
from local_market.soc_tail import fit_tail, select_xmin


def test_no_dissipation_accumulates_occupancy():
    cfg = ResearchConfig(4, 20, 5, activity=1, mode="no_dissipation")
    trace, events, rounds = simulate(cfg, 0, 12, "p")
    assert (trace.added_nodes >= 0).all()
    assert trace.occupancy_after.is_monotonic_increasing
    assert len(events) == len(rounds) == 0


def test_cascade_conserves_removed_sites_and_event_order():
    cfg = ResearchConfig(8, 100, 10, activity=1, theta=.02, q=.25, mode="cascade")
    trace, events, rounds = simulate(cfg, .9, 4, "a")
    assert len(events) > 0
    assert (events["size"] <= 64*events.occupancy_before).all()
    assert (events["size"] == events.apply(
        lambda e: rounds.loc[rounds.step == e.step, "removed_nodes"].sum(), axis=1)).all()
    assert (events.duration == events.apply(
        lambda e: len(rounds.loc[rounds.step == e.step]), axis=1)).all()
    np.testing.assert_array_equal(events.event_sequence, np.arange(1, len(events)+1))
    assert (events.step.diff().dropna() > 0).all()
    assert (trace.loc[trace.duration > 0, "added_nodes"] == 0).all()
    assert set(events.end_reason).issubset({"below_threshold", "zero_removal", "max_rounds"})


def test_tail_selection_requires_dynamic_range():
    xmin, _, status = select_xmin(np.full(120, 3), upper=100)
    assert xmin is None and status == "dynamic_range_or_tail_insufficient"


def test_seed_and_waiting_times_are_path_local():
    cfg = ResearchConfig(8, 80, 10, activity=1, theta=.02, q=.25, mode="cascade")
    first = simulate(cfg, .9, 44, "first")
    again = simulate(cfg, .9, 44, "first")
    for left, right in zip(first, again):
        assert left.equals(right)
    events = first[1]
    assert np.isnan(events.waiting_steps.iloc[0])
    np.testing.assert_array_equal(events.waiting_steps.iloc[1:], events.step.diff().iloc[1:])


def test_discrete_models_share_finite_support():
    x = np.array([2, 2, 3, 5, 8, 10])
    for model in ("power", "cutoff", "exponential"):
        fit = fit_tail(x, 2, 10, model)
        assert fit["xmin"] == 2 and fit["upper"] == 10
        assert np.isfinite(fit["log_likelihood"])
        assert 0 <= fit["ks"] <= 1

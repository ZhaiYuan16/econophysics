import numpy as np

from local_market.statistics import (
    nonzero_absolute_standardized_returns,
    return_distribution_table,
    standardize_returns,
)


def test_zero_raw_returns_are_excluded_only_from_tail_sample() -> None:
    raw = np.array([0., 0., 0., 1., 2., -1.])
    z, _, _ = standardize_returns(raw)
    assert np.all(z[raw == 0] != 0)
    tail = nonzero_absolute_standardized_returns(raw, z)
    np.testing.assert_array_equal(tail, np.abs(z[raw != 0]))

    rows = return_distribution_table(z, raw_returns=raw)
    ccdf = [row for row in rows if row["distribution"] == "absolute_ccdf"]
    assert all(row["value"] == row["count"] / len(raw) for row in ccdf)
    assert max(row["count"] for row in ccdf) <= len(raw)

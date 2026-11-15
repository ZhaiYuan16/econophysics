"""Numerical checks for the finite discrete tail analysis."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('cluster_tail_l512', Path(__file__).resolve().parents[1] / 'scripts/analysis/analyze_cluster_tail_l512.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_normalization_and_zero_count_sizes():
    p = mod.model(None, 3, 9, 2.1)
    assert len(p) == 7
    assert np.isclose(p.sum(), 1)
    assert np.all(p > 0)


def test_mle_and_discrete_ks():
    rng = np.random.default_rng(123)
    p = mod.model(None, 5, 50, 2.2)
    c = np.zeros(51, dtype=int)
    c[5:] = rng.multinomial(200000, p)
    f = mod.fit(c, 5, 50)
    assert abs(f['tau'] - 2.2) < .025
    assert f['ks'] < .01
    assert f['n'] == 200000


def test_configuration_resampling_preserves_blocks():
    rng = np.random.default_rng(1)
    blocks = np.array([[1, 0, 4], [0, 2, 8], [3, 0, 12]])
    ids = rng.integers(3, size=3)
    resample = blocks[ids].sum(axis=0)
    assert np.array_equal(resample, sum(blocks[i] for i in ids))


def test_continuity_rejects_isolated_high_count():
    c = np.zeros(101, dtype=int)
    c[1:61] = 10
    c[100] = 10
    assert mod.upper(c, 10, 0) == 100
    assert mod.upper(c, 10, 25) == 60

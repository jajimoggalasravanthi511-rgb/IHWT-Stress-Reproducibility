import pandas as pd

from src.statistics import cluster_bootstrap_ci, holm_adjust


def test_bootstrap_ci_order():
    s = pd.Series([0.8, 0.82, 0.85, 0.87, 0.9])
    mean, lo, hi = cluster_bootstrap_ci(s, iterations=500, seed=1)
    assert lo <= mean <= hi


def test_holm_is_bounded():
    a = holm_adjust([0.001, 0.03, 0.2])
    assert all(0 <= x <= 1 for x in a)

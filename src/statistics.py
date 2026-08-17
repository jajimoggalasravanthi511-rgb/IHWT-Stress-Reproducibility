from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def cluster_bootstrap_ci(values_by_subject: pd.Series, iterations: int = 5000, confidence: float = 0.95, seed: int = 1234) -> tuple[float, float, float]:
    vals = values_by_subject.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = np.mean(rng.choice(vals, size=vals.size, replace=True))
    alpha = (1.0 - confidence) / 2.0
    return float(np.mean(vals)), float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def paired_statistical_tests(subject_metric: pd.DataFrame, reference: str, value_col: str = "value") -> pd.DataFrame:
    """subject_metric columns: subject_id, method, value."""
    pivot = subject_metric.pivot_table(index="subject_id", columns="method", values=value_col, aggfunc="mean").dropna(axis=0)
    rows = []
    methods = list(pivot.columns)
    if len(methods) >= 3 and len(pivot) >= 3:
        stat, p = friedmanchisquare(*[pivot[m].to_numpy() for m in methods])
        rows.append({"test": "Friedman", "comparison": "all_methods", "statistic": float(stat), "p_value": float(p)})
    raw_pair_rows = []
    for m in methods:
        if m == reference:
            continue
        d = pivot[[reference, m]].dropna()
        if len(d) < 2:
            continue
        try:
            stat, p = wilcoxon(d[reference], d[m], zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = 0.0, 1.0
        raw_pair_rows.append({"test": "Wilcoxon", "comparison": f"{reference} vs {m}", "statistic": float(stat), "p_value": float(p)})
    if raw_pair_rows:
        adj = holm_adjust([r["p_value"] for r in raw_pair_rows])
        for r, a in zip(raw_pair_rows, adj):
            r["p_holm"] = a
        rows.extend(raw_pair_rows)
    return pd.DataFrame(rows)

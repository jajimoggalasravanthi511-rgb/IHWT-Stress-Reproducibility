from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_signal
from .features import extract_features
from .preprocess import preprocess_signal


def benchmark_first_record(manifest_path: str, cfg: dict, methods: list[str]) -> pd.DataFrame:
    """Measured preprocessing and feature-extraction timing on the first manifest record."""
    df = pd.read_csv(manifest_path)
    if df.empty:
        return pd.DataFrame()
    r = df.iloc[0]
    seconds = float(cfg["data"]["window_seconds"])
    repeats = int(cfg["runtime"]["repeats"])
    rows = []
    for modality in cfg["data"]["modalities"]:
        fs = float(r[f"fs_{modality}"])
        x = load_signal(str(r[f"{modality}_path"]))[: int(round(seconds * fs))]
        if x.size < int(round(seconds * fs)):
            continue
        prep_times = []
        processed = None
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            processed, q = preprocess_signal(x, fs, modality, cfg["preprocessing"])
            prep_times.append((time.perf_counter_ns() - t0) / 1e6)
        rows.append({
            "stage": "preprocessing", "method": "-", "modality": modality,
            "mean_ms": float(np.mean(prep_times)), "std_ms": float(np.std(prep_times, ddof=1)),
            "p95_ms": float(np.quantile(prep_times, 0.95)), "n": repeats,
        })
        if not q.get("valid", False):
            continue
        for method in methods:
            if method == "emd":
                # EMD may be intentionally optional; benchmark only if installed.
                try:
                    import PyEMD  # noqa: F401
                except Exception:
                    continue
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                extract_features(processed, fs, method, cfg["ihwt"])
                times.append((time.perf_counter_ns() - t0) / 1e6)
            rows.append({
                "stage": "feature_extraction", "method": method, "modality": modality,
                "mean_ms": float(np.mean(times)), "std_ms": float(np.std(times, ddof=1)),
                "p95_ms": float(np.quantile(times, 0.95)), "n": repeats,
            })
    return pd.DataFrame(rows)


def benchmark_scaling(cfg: dict, methods: list[str]) -> pd.DataFrame:
    """Empirical transform scaling on deterministic synthetic signals; not classification evidence."""
    rng = np.random.default_rng(123)
    rows = []
    repeats = max(3, min(10, int(cfg["runtime"]["repeats"])))
    for n in cfg["runtime"]["signal_lengths"]:
        t = np.arange(int(n)) / 256.0
        x = np.sin(2*np.pi*1.3*t) + 0.15*np.sin(2*np.pi*18*t) + 0.02*rng.normal(size=int(n))
        for method in methods:
            if method == "emd":
                try:
                    import PyEMD  # noqa: F401
                except Exception:
                    continue
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter_ns()
                extract_features(x, 256.0, method, cfg["ihwt"])
                times.append((time.perf_counter_ns() - t0) / 1e6)
            rows.append({"method": method, "n_samples": int(n), "mean_ms": float(np.mean(times)), "std_ms": float(np.std(times, ddof=1)), "repeats": repeats})
    return pd.DataFrame(rows)

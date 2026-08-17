from __future__ import annotations

import numpy as np

# Orthonormal analysis low-pass coefficients. The implementation is kept
# self-contained so the central transform is not hidden behind a third-party API.
_FILTERS = {
    "haar": np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=float),
    "db4": np.array([
        -0.010597401785069032,
         0.0328830116668852,
         0.030841381835560764,
        -0.18703481171888114,
        -0.027983769416859854,
         0.6308807679298587,
         0.7148465705529154,
         0.2303778133088965,
    ], dtype=float),
}


def analysis_filters(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in _FILTERS:
        raise ValueError(f"Unsupported wavelet '{name}'. Supported: {sorted(_FILTERS)}")
    lo = _FILTERS[name]
    # Quadrature-mirror high-pass analysis filter.
    hi = lo[::-1].copy()
    hi[::2] *= -1.0
    return lo, hi


def dwt_once(x: np.ndarray, wavelet: str = "db4") -> tuple[np.ndarray, np.ndarray]:
    """Single-level decimated orthogonal filter bank with symmetric extension."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 4:
        raise ValueError("At least four samples are required for a DWT node.")
    lo, hi = analysis_filters(wavelet)
    pad = max(len(lo) - 1, 1)
    xp = np.pad(x, (pad, pad), mode="symmetric")
    a = np.convolve(xp, lo[::-1], mode="valid")[::2]
    d = np.convolve(xp, hi[::-1], mode="valid")[::2]
    n = min(a.size, d.size)
    return a[:n], d[:n]


def full_dwt(x: np.ndarray, max_level: int, wavelet: str = "db4") -> dict[str, np.ndarray]:
    """Classical approximation-only DWT; returns details d1..dJ and aJ."""
    coeffs: dict[str, np.ndarray] = {}
    cur = np.asarray(x, dtype=float)
    for level in range(1, max_level + 1):
        if cur.size < 8:
            break
        a, d = dwt_once(cur, wavelet)
        coeffs[f"d{level}"] = d
        cur = a
    coeffs[f"a{len([k for k in coeffs if k.startswith('d')])}"] = cur
    return coeffs


def full_wavelet_packet(x: np.ndarray, max_level: int, wavelet: str = "db4") -> dict[str, np.ndarray]:
    """Full binary wavelet packet tree up to max_level."""
    nodes = {"": np.asarray(x, dtype=float)}
    leaves: dict[str, np.ndarray] = {}
    for _ in range(max_level):
        nxt: dict[str, np.ndarray] = {}
        for path, c in nodes.items():
            if c.size < 8:
                leaves[path] = c
                continue
            a, d = dwt_once(c, wavelet)
            nxt[path + "a"] = a
            nxt[path + "d"] = d
        nodes = nxt
        if not nodes:
            break
    leaves.update(nodes)
    return leaves

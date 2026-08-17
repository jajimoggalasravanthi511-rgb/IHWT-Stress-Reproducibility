from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

from .ihwt import IHWTConfig, InterpolativeHeuristicWaveletTransform, normalized_shannon_entropy
from .wavelets import full_dwt, full_wavelet_packet


def _stats(c: np.ndarray, prefix: str) -> tuple[list[float], list[str]]:
    c = np.asarray(c, dtype=float).ravel()
    if c.size == 0:
        vals = [0.0] * 6
    else:
        vals = [
            float(np.mean(c * c)),
            normalized_shannon_entropy(c),
            float(np.sqrt(np.mean(c * c))),
            float(np.std(c)),
            float(skew(c, bias=False)) if c.size > 3 else 0.0,
            float(kurtosis(c, fisher=True, bias=False)) if c.size > 4 else 0.0,
        ]
        vals = [0.0 if not np.isfinite(v) else v for v in vals]
    names = [f"{prefix}_{s}" for s in ("energy", "entropy", "rms", "std", "skew", "kurtosis")]
    return vals, names


def fft_features(x: np.ndarray, fs: float) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x, dtype=float)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    p = spec / (spec.sum() + 1e-12)
    centroid = float(np.sum(freqs * p))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(p.size, 2)))
    q = np.cumsum(p)
    roll95 = float(freqs[min(np.searchsorted(q, 0.95), freqs.size - 1)])
    vals, names = _stats(x, "time")
    vals += [centroid, spread, entropy, roll95]
    names += ["fft_centroid", "fft_spread", "fft_entropy", "fft_rolloff95"]
    return np.asarray(vals), names


def dwt_features(x: np.ndarray, max_level: int, wavelet: str) -> tuple[np.ndarray, list[str]]:
    coeffs = full_dwt(x, max_level, wavelet)
    vals: list[float] = []
    names: list[str] = []
    for key in sorted(coeffs):
        v, n = _stats(coeffs[key], f"dwt_{key}")
        vals.extend(v); names.extend(n)
    return np.asarray(vals), names


def wpt_features(x: np.ndarray, max_level: int, wavelet: str) -> tuple[np.ndarray, list[str]]:
    leaves = full_wavelet_packet(x, max_level, wavelet)
    vals: list[float] = []
    names: list[str] = []
    # A fixed full tree is expected for normal 30-s windows; sorted paths keep order deterministic.
    for key in sorted(leaves):
        v, n = _stats(leaves[key], f"wpt_{key or 'root'}")
        vals.extend(v); names.extend(n)
    return np.asarray(vals), names


def emd_features(x: np.ndarray, max_imfs: int = 6) -> tuple[np.ndarray, list[str]]:
    try:
        from PyEMD import EMD
    except Exception as e:
        raise RuntimeError("EMD comparator requires the optional 'EMD-signal' package.") from e
    imfs = EMD().emd(np.asarray(x, dtype=float), max_imf=max_imfs)
    vals: list[float] = []
    names: list[str] = []
    for i in range(max_imfs):
        c = imfs[i] if i < len(imfs) else np.zeros(8)
        v, n = _stats(c, f"emd_imf{i+1}")
        vals.extend(v); names.extend(n)
    return np.asarray(vals), names


def extract_features(x: np.ndarray, fs: float, method: str, ihwt_cfg: dict) -> tuple[np.ndarray, list[str]]:
    method = method.lower()
    if method == "ihwt":
        cfg = IHWTConfig(**ihwt_cfg)
        return InterpolativeHeuristicWaveletTransform(cfg).fixed_feature_vector(x)
    if method == "dwt":
        return dwt_features(x, int(ihwt_cfg["max_level"]), str(ihwt_cfg["wavelet"]))
    if method == "wpt":
        return wpt_features(x, int(ihwt_cfg["max_level"]), str(ihwt_cfg["wavelet"]))
    if method == "fft":
        return fft_features(x, fs)
    if method == "emd":
        return emd_features(x)
    raise ValueError(f"Unknown feature method: {method}")

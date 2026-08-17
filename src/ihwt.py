from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict

import numpy as np

from .wavelets import dwt_once


@dataclass(frozen=True)
class IHWTConfig:
    wavelet: str = "db4"
    max_level: int = 4
    entropy_threshold: float = 0.55
    interpolation_lambda: float = 0.65
    min_node_samples: int = 64
    eps: float = 1e-12


def normalized_shannon_entropy(c: np.ndarray, eps: float = 1e-12) -> float:
    """Energy-normalized Shannon entropy in [0, 1] for a non-empty coefficient vector."""
    c = np.asarray(c, dtype=float).ravel()
    if c.size <= 1:
        return 0.0
    e = np.square(c)
    total = float(e.sum())
    if total <= eps:
        return 0.0
    p = e / total
    h = -float(np.sum(p * np.log(p + eps)))
    return float(np.clip(h / np.log(c.size), 0.0, 1.0))


def linear_lattice_interpolant(c: np.ndarray) -> np.ndarray:
    """
    Interpolate odd-index coefficients from the even-index lattice.

    The operation provides a deterministic local reference. IHWT blends this
    interpolant with the observed coefficient sequence. High-information nodes
    retain more of the original coefficients; low-information nodes receive
    more interpolation, reducing sensitivity to isolated noise.
    """
    c = np.asarray(c, dtype=float).ravel()
    if c.size < 3:
        return c.copy()
    idx = np.arange(c.size)
    anchors = idx[::2]
    vals = c[::2]
    if anchors[-1] != idx[-1]:
        anchors = np.r_[anchors, idx[-1]]
        vals = np.r_[vals, c[-1]]
    return np.interp(idx, anchors, vals)


def node_statistics(c: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    c = np.asarray(c, dtype=float).ravel()
    if c.size == 0:
        return {k: 0.0 for k in ("energy", "entropy", "rms", "std", "skew", "kurtosis")}
    mean = float(np.mean(c))
    std = float(np.std(c))
    z = (c - mean) / (std + eps)
    return {
        "energy": float(np.mean(c * c)),
        "entropy": normalized_shannon_entropy(c, eps),
        "rms": float(np.sqrt(np.mean(c * c))),
        "std": std,
        "skew": float(np.mean(z ** 3)),
        "kurtosis": float(np.mean(z ** 4) - 3.0),
    }


def all_paths(max_level: int) -> list[str]:
    return ["".join(bits) for level in range(1, max_level + 1) for bits in product("ad", repeat=level)]


class InterpolativeHeuristicWaveletTransform:
    """
    Entropy-guided selective wavelet-packet decomposition.

    At each split, child b receives normalized entropy H_b and relative energy
    R_b. The heuristic score is q_b = lambda*H_b + (1-lambda)*R_b. A child is
    recursively decomposed only when q_b >= tau and finite-depth constraints
    are satisfied. The coefficient blend is

        c_tilde = q_b*c + (1-q_b)*I(c),

    where I(c) is a local linear-lattice interpolant. Thus information-rich
    nodes preserve transient coefficients while low-score nodes are regularized.

    This is a finite-depth adaptive transform, not an iterative optimizer; the
    relevant theoretical property is deterministic termination rather than
    asymptotic optimizer convergence.
    """

    def __init__(self, config: IHWTConfig):
        self.cfg = config

    def transform(self, x: np.ndarray) -> Dict[str, dict]:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.cfg.min_node_samples:
            raise ValueError("Signal is shorter than min_node_samples.")
        result: Dict[str, dict] = {}
        self._visit(x, path="", level=0, out=result, parent_energy=float(np.sum(x * x)) + self.cfg.eps)
        return result

    def _visit(self, c: np.ndarray, path: str, level: int, out: Dict[str, dict], parent_energy: float) -> None:
        if level >= self.cfg.max_level or c.size < self.cfg.min_node_samples:
            return
        a, d = dwt_once(c, self.cfg.wavelet)
        sibling_energy = float(np.sum(a * a) + np.sum(d * d)) + self.cfg.eps
        for symbol, child in (("a", a), ("d", d)):
            child_path = path + symbol
            h = normalized_shannon_entropy(child, self.cfg.eps)
            r = float(np.sum(child * child) / sibling_energy)
            q = float(np.clip(
                self.cfg.interpolation_lambda * h + (1.0 - self.cfg.interpolation_lambda) * r,
                0.0,
                1.0,
            ))
            interpolated = linear_lattice_interpolant(child)
            adapted = q * child + (1.0 - q) * interpolated
            out[child_path] = {
                "level": level + 1,
                "entropy": h,
                "relative_energy": r,
                "heuristic_score": q,
                "decomposed": bool(q >= self.cfg.entropy_threshold and level + 1 < self.cfg.max_level and adapted.size >= self.cfg.min_node_samples),
                "coefficients": adapted,
                "statistics": node_statistics(adapted, self.cfg.eps),
            }
            if out[child_path]["decomposed"]:
                self._visit(adapted, child_path, level + 1, out, float(np.sum(adapted * adapted)) + self.cfg.eps)

    def fixed_feature_vector(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        tree = self.transform(x)
        stats = ("energy", "entropy", "rms", "std", "skew", "kurtosis")
        values: list[float] = []
        names: list[str] = []
        for path in all_paths(self.cfg.max_level):
            node = tree.get(path)
            for stat in stats:
                names.append(f"ihwt_{path}_{stat}")
                values.append(0.0 if node is None else float(node["statistics"][stat]))
            names.append(f"ihwt_{path}_score")
            values.append(0.0 if node is None else float(node["heuristic_score"]))
            names.append(f"ihwt_{path}_active")
            values.append(0.0 if node is None else 1.0)
        return np.asarray(values, dtype=float), names

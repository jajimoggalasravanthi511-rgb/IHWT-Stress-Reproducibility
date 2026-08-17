import numpy as np

from src.ihwt import IHWTConfig, InterpolativeHeuristicWaveletTransform, all_paths, normalized_shannon_entropy


def test_entropy_bounds():
    rng = np.random.default_rng(2)
    for n in (64, 128, 512):
        h = normalized_shannon_entropy(rng.normal(size=n))
        assert 0.0 <= h <= 1.0


def test_ihwt_is_deterministic_and_finite():
    t = np.linspace(0, 8, 2048, endpoint=False)
    x = np.sin(2*np.pi*2*t) + 0.2*np.sin(2*np.pi*17*t)
    cfg = IHWTConfig(max_level=4, min_node_samples=32)
    tr = InterpolativeHeuristicWaveletTransform(cfg)
    a = tr.transform(x); b = tr.transform(x)
    assert set(a) == set(b)
    assert len(a) <= len(all_paths(cfg.max_level))
    for k in a:
        assert np.allclose(a[k]["coefficients"], b[k]["coefficients"])
        assert 0 <= a[k]["heuristic_score"] <= 1

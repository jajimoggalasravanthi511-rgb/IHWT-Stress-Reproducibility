import numpy as np

from src.preprocess import preprocess_signal


CFG = {
    "ecg": {"bandpass_hz": [0.5, 40.0], "filter_order": 5},
    "gsr": {"lowpass_hz": 5.0, "filter_order": 4},
    "ppg": {"bandpass_hz": [0.5, 8.0], "filter_order": 4},
    "powerline_hz": 50.0,
    "notch_q": 30.0,
    "spike_z": 3.0,
    "within_window_zscore": True,
}


def test_preprocess_returns_finite_signal():
    fs = 256
    t = np.arange(fs * 30) / fs
    x = np.sin(2*np.pi*1.2*t) + 0.03*np.random.default_rng(0).normal(size=t.size)
    y, q = preprocess_signal(x, fs, "ppg", CFG)
    assert q["valid"]
    assert np.isfinite(y).all()
    assert abs(np.mean(y)) < 1e-8

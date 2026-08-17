from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt


def robust_zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 0.67448975 * (x - med) / (mad + eps)


def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.size < 32:
        return x.copy()
    return sosfiltfilt(sos, x)


def preprocess_signal(x: np.ndarray, fs: float, modality: str, cfg: dict) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, dtype=float).ravel()
    finite = np.isfinite(x)
    if finite.mean() < 0.95:
        return x, {"valid": False, "reason": "too_many_nonfinite"}
    if not finite.all():
        idx = np.arange(x.size)
        x[~finite] = np.interp(idx[~finite], idx[finite], x[finite])

    pcfg = cfg[modality]
    nyq = fs / 2.0
    if modality == "ecg":
        lo, hi = pcfg["bandpass_hz"]
        hi = min(float(hi), 0.95 * nyq)
        sos = butter(int(pcfg["filter_order"]), [float(lo) / nyq, hi / nyq], btype="bandpass", output="sos")
        y = _safe_sosfiltfilt(sos, x)
    elif modality == "gsr":
        cutoff = min(float(pcfg["lowpass_hz"]), 0.95 * nyq)
        sos = butter(int(pcfg["filter_order"]), cutoff / nyq, btype="lowpass", output="sos")
        y = _safe_sosfiltfilt(sos, x)
    elif modality == "ppg":
        lo, hi = pcfg["bandpass_hz"]
        hi = min(float(hi), 0.95 * nyq)
        sos = butter(int(pcfg["filter_order"]), [float(lo) / nyq, hi / nyq], btype="bandpass", output="sos")
        y = _safe_sosfiltfilt(sos, x)
    else:
        raise ValueError(f"Unknown modality {modality}")

    # Notch only when it is below Nyquist.
    f0 = float(cfg.get("powerline_hz", 50.0))
    if f0 < 0.95 * nyq and y.size >= 32:
        b, a = iirnotch(f0 / nyq, float(cfg.get("notch_q", 30.0)))
        y = filtfilt(b, a, y)

    rz = robust_zscore(y)
    spike_fraction = float(np.mean(np.abs(rz) > float(cfg.get("spike_z", 3.0))))
    valid = spike_fraction <= 0.05 and np.std(y) > 1e-12

    if bool(cfg.get("within_window_zscore", True)):
        y = (y - np.mean(y)) / (np.std(y) + 1e-12)

    return y.astype(np.float64), {
        "valid": bool(valid),
        "spike_fraction": spike_fraction,
        "std": float(np.std(y)),
    }

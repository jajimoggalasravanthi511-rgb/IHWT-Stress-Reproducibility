from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import resample

from .features import extract_features
from .preprocess import preprocess_signal


def load_signal(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == ".npy":
        return np.asarray(np.load(p), dtype=float).ravel()
    if p.suffix.lower() == ".npz":
        z = np.load(p)
        key = "value" if "value" in z.files else z.files[0]
        return np.asarray(z[key], dtype=float).ravel()
    if p.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(p)
        if "value" in df.columns:
            return df["value"].to_numpy(dtype=float)
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"No numeric signal column in {p}")
        return numeric.iloc[:, -1].to_numpy(dtype=float)
    raise ValueError(f"Unsupported signal format: {p.suffix}")


def validate_manifest(df: pd.DataFrame, modalities: Iterable[str]) -> None:
    required = {"subject_id", "session_id", "condition", "label"}
    for m in modalities:
        required |= {f"{m}_path", f"fs_{m}"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    if df["subject_id"].isna().any() or df["label"].isna().any():
        raise ValueError("subject_id and label must be complete.")


def _window_starts(n: int, fs: float, seconds: float, overlap: float) -> list[int]:
    size = int(round(seconds * fs))
    step = max(1, int(round(size * (1.0 - overlap))))
    if n < size:
        return []
    return list(range(0, n - size + 1, step))


def extract_feature_table(manifest_path: str, cfg: dict, method: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    modalities = list(cfg["data"]["modalities"])
    validate_manifest(manifest, modalities)
    rows: list[dict] = []

    for _, r in manifest.iterrows():
        signals = {m: load_signal(str(r[f"{m}_path"])) for m in modalities}
        fs = {m: float(r[f"fs_{m}"]) for m in modalities}
        starts = {m: _window_starts(len(signals[m]), fs[m], float(cfg["data"]["window_seconds"]), float(cfg["data"]["overlap"])) for m in modalities}
        nwin = min(len(starts[m]) for m in modalities)
        for w in range(nwin):
            feat: list[float] = []
            names: list[str] = []
            valid = True
            quality: dict[str, float] = {}
            for m in modalities:
                n = int(round(float(cfg["data"]["window_seconds"]) * fs[m]))
                seg = signals[m][starts[m][w]:starts[m][w] + n]
                seg, q = preprocess_signal(seg, fs[m], m, cfg["preprocessing"])
                quality[f"{m}_spike_fraction"] = float(q.get("spike_fraction", np.nan))
                if not q.get("valid", False):
                    valid = False
                    break
                fv, fn = extract_features(seg, fs[m], method, cfg["ihwt"])
                feat.extend(fv.tolist())
                names.extend([f"{m}_{x}" for x in fn])
            if not valid:
                continue
            row = {
                "subject_id": str(r["subject_id"]),
                "session_id": str(r["session_id"]),
                "condition": str(r["condition"]),
                "label": int(r["label"]),
                "window_index": int(w),
                **quality,
            }
            row.update(dict(zip(names, feat)))
            rows.append(row)
    if not rows:
        raise RuntimeError("No valid windows were produced. Check paths, sampling rates, and quality thresholds.")
    return pd.DataFrame(rows)


def extract_raw_window_tensor(manifest_path: str, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned fixed-length windows for CNN/BiLSTM/Transformer baselines."""
    manifest = pd.read_csv(manifest_path)
    modalities = list(cfg["data"]["modalities"])
    validate_manifest(manifest, modalities)
    target_n = int(cfg["data"]["deep_input_samples"])
    X, y, g = [], [], []
    for _, r in manifest.iterrows():
        signals = {m: load_signal(str(r[f"{m}_path"])) for m in modalities}
        fs = {m: float(r[f"fs_{m}"]) for m in modalities}
        starts = {m: _window_starts(len(signals[m]), fs[m], float(cfg["data"]["window_seconds"]), float(cfg["data"]["overlap"])) for m in modalities}
        nwin = min(len(starts[m]) for m in modalities)
        for w in range(nwin):
            channels = []
            valid = True
            for m in modalities:
                n = int(round(float(cfg["data"]["window_seconds"]) * fs[m]))
                seg = signals[m][starts[m][w]:starts[m][w] + n]
                seg, q = preprocess_signal(seg, fs[m], m, cfg["preprocessing"])
                if not q.get("valid", False):
                    valid = False; break
                channels.append(resample(seg, target_n).astype(np.float32))
            if valid:
                X.append(np.stack(channels))
                y.append(int(r["label"]))
                g.append(str(r["subject_id"]))
    if not X:
        raise RuntimeError("No valid raw windows were produced.")
    return np.stack(X), np.asarray(y, dtype=int), np.asarray(g)

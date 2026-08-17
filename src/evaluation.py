from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .models import WeightedVotingEnsemble, build_models, derive_inner_group_weights


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def loso_evaluate(feature_df: pd.DataFrame, cfg: dict, method: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = {"subject_id", "session_id", "condition", "label", "window_index", "ecg_spike_fraction", "gsr_spike_fraction", "ppg_spike_fraction"}
    feature_cols = [c for c in feature_df.columns if c not in meta]
    X0 = feature_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = feature_df["label"].to_numpy(dtype=int)
    groups = feature_df["subject_id"].astype(str).to_numpy()
    subjects = np.unique(groups)
    fold_rows, pred_rows = [], []

    for subject in subjects:
        te = groups == subject
        tr = ~te
        scaler = StandardScaler().fit(X0[tr])
        Xtr, Xte = scaler.transform(X0[tr]), scaler.transform(X0[te])
        ytr, yte = y[tr], y[te]
        gtr = groups[tr]
        weights = derive_inner_group_weights(Xtr, ytr, gtr, cfg["models"], seed)
        models = build_models(cfg["models"], seed, len(np.unique(y)))
        fitted = {}
        for name, model in models.items():
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            elapsed = time.perf_counter() - t0
            fitted[name] = model
            m = metric_dict(yte, pred)
            fold_rows.append({"seed": seed, "method": method, "model": name, "subject_id": subject, "n_test": int(te.sum()), "fit_predict_seconds": elapsed, **m})
            for idx, yt, yp in zip(np.where(te)[0], yte, pred):
                pred_rows.append({"seed": seed, "method": method, "model": name, "subject_id": subject, "row_index": int(idx), "y_true": int(yt), "y_pred": int(yp)})
        ens = WeightedVotingEnsemble(fitted, weights, np.unique(y))
        ep = ens.predict(Xte)
        m = metric_dict(yte, ep)
        fold_rows.append({"seed": seed, "method": method, "model": "Ensemble", "subject_id": subject, "n_test": int(te.sum()), "fit_predict_seconds": np.nan, **m, **{f"weight_{k}": v for k, v in weights.items()}})
        for idx, yt, yp in zip(np.where(te)[0], yte, ep):
            pred_rows.append({"seed": seed, "method": method, "model": "Ensemble", "subject_id": subject, "row_index": int(idx), "y_true": int(yt), "y_pred": int(yp)})
    return pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)

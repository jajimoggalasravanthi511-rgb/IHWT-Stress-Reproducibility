from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(pred_df: pd.DataFrame, path: str | Path, title: str) -> None:
    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_method_comparison(summary: pd.DataFrame, metric: str, path: str | Path) -> None:
    d = summary[(summary["model"] == "Ensemble") & (summary["metric"] == metric)].copy()
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.bar(d["method"], d["mean"])
    yerr = np.vstack([d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]])
    ax.errorbar(np.arange(len(d)), d["mean"], yerr=yerr, fmt="none", capsize=4)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Feature extraction method")
    ax.set_ylim(0, 1)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

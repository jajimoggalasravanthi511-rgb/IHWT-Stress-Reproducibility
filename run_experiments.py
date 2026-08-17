from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import extract_feature_table, extract_raw_window_tensor
from src.benchmark import benchmark_first_record, benchmark_scaling
from src.deep_models import deep_loso
from src.evaluation import loso_evaluate
from src.plots import save_confusion_matrix, save_method_comparison
from src.statistics import cluster_bootstrap_ci, paired_statistical_tests
from src.utils import dump_json, ensure_dir, load_config, set_global_seed


def summarize(folds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    metrics = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc"]
    rows = []
    for (method, model), d in folds.groupby(["method", "model"]):
        for metric in metrics:
            by_subject = d.groupby("subject_id")[metric].mean()
            mean, lo, hi = cluster_bootstrap_ci(by_subject, int(cfg["evaluation"]["bootstrap_iterations"]), float(cfg["evaluation"]["confidence_level"]))
            rows.append({"method": method, "model": model, "metric": metric, "mean": mean, "sd_subject": float(by_subject.std(ddof=1)) if by_subject.size > 1 else 0.0, "ci_low": lo, "ci_high": hi, "n_subjects": int(by_subject.size)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run subject-independent IHWT stress-detection experiments.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--manifest", default=None, help="Override data.manifest from config")
    ap.add_argument("--methods", default=None, help="Comma-separated feature methods, e.g. ihwt,dwt,wpt,fft,emd")
    ap.add_argument("--deep", action="store_true", help="Also run CNN, BiLSTM and Transformer raw-signal baselines")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = args.manifest or cfg["data"]["manifest"]
    methods = args.methods.split(",") if args.methods else list(cfg["feature_methods"])
    out = ensure_dir(cfg["experiment"]["output_dir"])
    dump_json(cfg, out / "resolved_config.json")

    all_folds, all_preds = [], []
    benchmark_first_record(manifest, cfg, methods).to_csv(out / "runtime_benchmark.csv", index=False)
    benchmark_scaling(cfg, methods).to_csv(out / "complexity_scaling.csv", index=False)
    for method in methods:
        print(f"Extracting {method} features ...")
        table = extract_feature_table(manifest, cfg, method)
        table.to_csv(out / f"features_{method}.csv.gz", index=False, compression="gzip")
        for seed in cfg["experiment"]["seeds"]:
            set_global_seed(int(seed))
            folds, preds = loso_evaluate(table, cfg, method, int(seed))
            all_folds.append(folds); all_preds.append(preds)

    folds = pd.concat(all_folds, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    if args.deep or bool(cfg["deep_baselines"].get("enabled", False)):
        Xraw, yraw, graw = extract_raw_window_tensor(manifest, cfg)
        for seed in cfg["experiment"]["seeds"]:
            set_global_seed(int(seed))
            all_folds.append(deep_loso(Xraw, yraw, graw, cfg["deep_baselines"], int(seed)))
        folds = pd.concat(all_folds, ignore_index=True)

    folds.to_csv(out / "fold_metrics.csv", index=False)
    preds.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    summary = summarize(folds, cfg)
    summary.to_csv(out / "summary_metrics.csv", index=False)

    metric = cfg["evaluation"]["statistical_metric"]
    sub = folds[(folds["model"] == "Ensemble")].groupby(["subject_id", "method"], as_index=False)[metric].mean().rename(columns={metric: "value"})
    if "ihwt" in set(sub["method"]):
        tests = paired_statistical_tests(sub, reference="ihwt")
        tests.to_csv(out / "statistical_tests.csv", index=False)

    p = preds[(preds["method"] == "ihwt") & (preds["model"] == "Ensemble")]
    if not p.empty:
        save_confusion_matrix(p, out / "figure_confusion_matrix.png", "IHWT Ensemble — subject-independent predictions")
    save_method_comparison(summary, cfg["evaluation"]["primary_metric"], out / "figure_feature_method_comparison.png")
    print(f"Completed. Results written to {Path(out).resolve()}")


if __name__ == "__main__":
    main()

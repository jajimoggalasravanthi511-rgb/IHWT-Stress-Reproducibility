from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = {1: "baseline", 2: "stress", 3: "amusement"}


def contiguous_mask_range(labels: np.ndarray, code: int) -> tuple[int, int] | None:
    idx = np.where(labels == code)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare the official WESAD files for this repository's manifest format.")
    ap.add_argument("--wesad-root", required=True, help="Directory containing S2/S2.pkl, S3/S3.pkl, ...")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--task", choices=["stress2", "affect3"], default="stress2")
    args = ap.parse_args()

    root, out = Path(args.wesad_root), Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pkl in sorted(root.glob("S*/S*.pkl")):
        subject = pkl.stem
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        labels = np.asarray(d["label"]).ravel()  # chest timeline, 700 Hz
        chest = d["signal"]["chest"]
        wrist = d["signal"]["wrist"]
        selected = [1, 2] if args.task == "stress2" else [1, 2, 3]
        mapping = {1: 0, 2: 1, 3: 2}
        for code in selected:
            r = contiguous_mask_range(labels, code)
            if r is None: continue
            a, b = r
            t0, t1 = a / 700.0, b / 700.0
            ecg = np.asarray(chest["ECG"])[a:b].ravel()
            gsr = np.asarray(chest["EDA"])[a:b].ravel()
            wa, wb = int(round(t0 * 64.0)), int(round(t1 * 64.0))
            ppg = np.asarray(wrist["BVP"])[wa:wb].ravel()
            base = out / f"{subject}_{LABELS[code]}"
            ep, gp, pp = str(base) + "_ecg.npy", str(base) + "_gsr.npy", str(base) + "_ppg.npy"
            np.save(ep, ecg); np.save(gp, gsr); np.save(pp, ppg)
            rows.append({
                "subject_id": subject,
                "session_id": f"{subject}_{LABELS[code]}",
                "condition": LABELS[code],
                "label": mapping[code],
                "ecg_path": str(Path(ep).resolve()), "gsr_path": str(Path(gp).resolve()), "ppg_path": str(Path(pp).resolve()),
                "fs_ecg": 700, "fs_gsr": 700, "fs_ppg": 64,
            })
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"Prepared {len(rows)} subject-condition records at {out / 'manifest.csv'}")
    if args.task == "affect3":
        print("NOTE: affect3 is baseline/stress/amusement; it is not a low/medium/high stress-intensity benchmark.")


if __name__ == "__main__":
    main()

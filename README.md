# IHWT Stress Detection Reproducibility Framework

This repository is a reviewer-oriented implementation of the physiological stress-analysis pipeline described in **“An Ensemble-Based Stress Detection Framework Using Interpolative Heuristic Wavelet Transform for Physiological Signal Analysis.”** It is intentionally structured around the unresolved review requirements: an explicit IHWT implementation, subject-independent evaluation, leakage control, complete classifier settings, statistical validation, runtime measurement, classical and modern baselines, and a public-data route for independent verification.

## What this repository implements

The software processes synchronized **ECG, GSR/EDA, and PPG/BVP** recordings, performs modality-specific filtering and signal-quality screening, creates 30-s windows with configurable overlap, extracts features using IHWT or comparator transforms, and evaluates Random Forest, XGBoost, RUSBoost, and a training-only weighted voting ensemble. Evaluation is **Leave-One-Subject-Out (LOSO)** so windows from the held-out participant never occur in model fitting, feature scaling, or ensemble-weight estimation.

The IHWT implementation is fully visible in `src/ihwt.py`. A child subband receives normalized Shannon entropy `H`, relative subband energy `R`, and a bounded heuristic score

`q = lambda*H + (1-lambda)*R`.

A node is recursively decomposed only if `q >= tau`, subject to finite depth and minimum-length constraints. The adapted coefficient vector is

`c_tilde = q*c + (1-q)*I(c)`,

where `I(c)` is a deterministic linear interpolant from the even-index coefficient lattice. High-information nodes therefore preserve a larger fraction of observed transient coefficients; lower-score nodes are more strongly regularized by interpolation. This definition removes ambiguity about what “interpolative” and “heuristic” mean in the software.

**Important scientific boundary:** the implementation proves bounded scores and deterministic finite termination; it does not claim a mathematical theorem that IHWT must outperform every adaptive wavelet method. Superiority is an empirical hypothesis and is tested by paired subject-level statistics.

## Repository structure

Only two tracked folders are used:

- `src/` — transform, preprocessing, features, models, LOSO evaluation, statistics, deep baselines, and figure generation.
- `tests/` — deterministic transform, leakage/splitting, preprocessing, and statistical smoke tests.

Root-level files contain the declared experiment configuration, public-dataset adapter, reproducibility documentation, availability statements, and release checklist. No license file is included.

## Installation

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Custom 50-participant dataset

Create a `data_manifest.csv` using `data_manifest_template.csv`. Each row represents one subject/session/condition and points to ECG, GSR, and PPG signal files. Supported files are CSV, NPY, and NPZ. CSV files should contain a numeric `value` column; otherwise the last numeric column is used.

Do **not** alter sampling rates to match the paper cosmetically. Put the true acquisition rate for each sensor into `fs_ecg`, `fs_gsr`, and `fs_ppg`. If the acquisition system used 500 Hz ECG, 128 Hz GSR, and 256 Hz PPG, record exactly those values. If all three were actually synchronized at 256 Hz, record 256 for all three. The manuscript currently contains both descriptions and this must be resolved from the acquisition record.

Run:

```bash
python run_experiments.py --config config.yaml --manifest data_manifest.csv
```

Generated outputs include per-subject LOSO metrics, all held-out predictions, bootstrap 95% confidence intervals, Friedman/Wilcoxon significance tests with Holm adjustment, and publication-resolution figures.

## Independent public benchmark: WESAD

WESAD can be used as an external public verification dataset. Download the original WESAD archive from its official distribution or UCI repository, extract it, and run:

```bash
python prepare_wesad.py --wesad-root /path/to/WESAD --output-root ./data_wesad --task stress2
python run_experiments.py --config config.yaml --manifest ./data_wesad/manifest.csv
```

`stress2` uses baseline versus stress. `affect3` is also supported, but its third class is amusement; it must **not** be described as a high-stress class. The adapter uses chest ECG and EDA plus wrist BVP (PPG-equivalent optical pulse waveform), preserving the official native sampling rates in the generated manifest.

## Strong baselines

Classical feature comparators are selected with `feature_methods` in `config.yaml`: `fft`, `dwt`, `wpt`, `ihwt`, and optional `emd`. WPT is included as a stronger multiband wavelet comparator than approximation-only DWT. EMD is available when `EMD-signal` is installed.

Modern raw-signal baselines are available with:

```bash
python run_experiments.py --config config.yaml --manifest data_manifest.csv --deep
```

This activates a 1-D CNN, BiLSTM, and Transformer encoder under the same held-out-subject principle. These models are deliberately kept separate from IHWT features so the comparison answers whether the proposed handcrafted adaptive transform remains competitive with end-to-end sequence models.

## Leakage prevention

The repository enforces the following boundaries:

1. LOSO defines the outer test subject.
2. `StandardScaler` is fitted on training-subject features only.
3. Ensemble weights are estimated only from inner GroupKFold splits of the training subjects.
4. Overlapping windows from the test subject never appear in training.
5. Statistical tests are performed on subject-level metrics rather than treating highly correlated windows as independent samples.

A window-level random 10-fold split is intentionally not the primary protocol because overlapping windows and repeated measurements from the same subject can create optimistic leakage.

## Statistical validation

`outputs/statistical_tests.csv` contains a Friedman test across feature methods and paired Wilcoxon signed-rank comparisons against IHWT. Pairwise p-values are Holm-adjusted. `outputs/summary_metrics.csv` reports participant-clustered bootstrap confidence intervals for accuracy, balanced accuracy, macro precision, macro recall, macro F1, and MCC.

Claims of statistical superiority should be made only when the generated p-values and effect direction support them.

## Computational cost

The filter-bank implementation processes at most the coefficients selected at each depth. With signal length `N`, maximum depth `J`, and fixed filter length `L`, the direct upper bound for the adaptive decomposition and entropy calculations is `O(JNL)`. For fixed `J` and `L`, this is linear in `N`. The manuscript should avoid an unsupported universal statement that standard DWT necessarily costs `O(N log N)`; compact finite-impulse-response DWT implementations are typically linear-time in signal length for fixed levels/filter length.

Runtime must be reported from measured executions on the declared hardware, not copied from a nominal table. `runtime_benchmark.csv` reports preprocessing and feature-extraction mean/SD/p95 timing, `complexity_scaling.csv` measures transform scaling across signal lengths, and `fold_metrics.csv` records model fit/predict time.

## Reproducible result policy

The repository contains **no hard-coded 93.4% accuracy, p-value, confidence interval, or confusion matrix**. Every numerical table intended for the paper must be regenerated from the actual released data and the archived configuration. Synthetic signals in unit tests verify software behavior only and must never be reported as experimental evidence.

## Key files

- `config.yaml`: single source of truth for preprocessing, IHWT, models, seeds, and statistics.
- `src/ihwt.py`: complete IHWT definition.
- `src/evaluation.py`: LOSO and training-only ensemble weighting.
- `src/benchmark.py`: measured preprocessing/feature runtime and empirical scaling.
- `src/statistics.py`: participant-level CIs and non-parametric tests.
- `src/deep_models.py`: CNN, BiLSTM, Transformer baselines.
- `prepare_wesad.py`: public benchmark adapter.
- `MANUSCRIPT_ALIGNMENT.md`: corrections that remain necessary before the reviewer response can truthfully claim completion.
- `REVIEWER5_TRACEABILITY.md`: direct mapping from every Reviewer 5 request to code/output evidence.
- `RELEASE_CHECKLIST.md`: steps for GitHub release and DOI archive after verification.

## Reproducibility command for an archived release

After all data paths and actual acquisition settings are verified:

```bash
pytest -q
python run_experiments.py --config config.yaml --manifest data_manifest.csv
```

Archive the exact GitHub release and `resolved_config.json` together. The DOI should be added to the manuscript **only after** a DOI-assigning repository has actually issued it.

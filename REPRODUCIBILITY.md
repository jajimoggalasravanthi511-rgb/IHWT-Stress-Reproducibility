# Reproducibility Protocol

## Required record before a paper result is considered reproducible

- Git commit or release tag.
- Archived `resolved_config.json`.
- Exact `data_manifest.csv` with pseudonymous subject IDs and native sampling frequencies.
- Cryptographic checksums for every released raw/derived data file.
- Python version and `pip freeze` output.
- Hardware identifier for runtime measurements.
- All random seeds.
- `fold_metrics.csv`, `predictions.csv.gz`, `summary_metrics.csv`, and `statistical_tests.csv`.

## Primary evaluation unit

The participant is the independent unit. Windows are repeated observations nested within a participant. Primary model selection and testing therefore use group-aware folds. Window-level random shuffling is prohibited for the headline result.

## Confidence intervals

The code first averages repeated-seed metrics within each participant and then bootstraps participants. This prevents the nominal sample size from being inflated by thousands of highly correlated windows.

## Statistical comparisons

Feature-method comparisons use the same held-out subjects. A Friedman test evaluates the global null across multiple methods; pairwise Wilcoxon signed-rank tests compare IHWT with each alternative and Holm adjustment controls family-wise multiplicity.

## Public benchmark separation

Results from WESAD are an independent benchmark and must not be merged with the custom 50-participant cohort. WESAD baseline/stress is a valid binary stress benchmark. If `affect3` is used, the third class is amusement, not high stress.

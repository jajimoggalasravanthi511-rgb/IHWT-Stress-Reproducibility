# IHWT Pipeline Pseudocode

**Input:** synchronized ECG, GSR, PPG recordings with subject identifier and condition label.  
**Output:** held-out-subject stress predictions and statistical evaluation.

1. For each subject-session record, load each modality at its true native sampling rate.
2. Apply modality-specific filters and motion/spike quality checks.
3. Divide the valid signals into 30-s windows with the declared overlap.
4. For each modality and window, apply the IHWT procedure:
   - split the current node using an orthogonal two-channel filter bank;
   - compute normalized Shannon entropy `H` for each child;
   - compute child relative energy `R` within the sibling pair;
   - compute `q = lambda*H + (1-lambda)*R`;
   - build `I(c)` by linear interpolation from the even-index coefficient lattice;
   - compute `c_tilde = q*c + (1-q)*I(c)`;
   - record energy, entropy, RMS, standard deviation, skewness, kurtosis, `q`, and node activity;
   - recursively decompose the child only when `q >= tau`, depth is below `J_max`, and the child contains enough samples.
5. Concatenate ECG/GSR/PPG features into one window vector.
6. Hold out one complete participant for testing.
7. Fit the feature scaler on training participants only.
8. On training participants only, estimate RF/XGBoost/RUSBoost voting weights by GroupKFold.
9. Refit RF, XGBoost, and RUSBoost on all training participants.
10. Predict the held-out participant and combine hard predictions using the training-derived weights.
11. Repeat Steps 6–10 for every participant and every declared random seed.
12. Report subject-wise metrics and participant-clustered 95% confidence intervals.
13. Apply Friedman and paired Wilcoxon tests to subject-level results, with Holm multiplicity correction.
14. Repeat the same outer protocol for DWT/WPT/FFT/EMD and CNN/BiLSTM/Transformer comparators.

# Hail Mary branch guide

The competition implementation is `submission_template`.

## What changed

- architecture diversity is measured with operation, connectivity, depth,
  width, and parameter descriptors;
- proxy-ranked candidates are reduced with quality-seeded farthest-first
  selection;
- diverse finalists receive equal-budget short training when time permits;
- calibrated proxy weights can be loaded from `proxy_weights.json`;
- hidden data are diagnosed without relying on their codename;
- discrete/encoded inputs receive no destructive augmentation;
- non-finite values, constant channels, non-contiguous labels, and imbalance
  receive explicit handling;
- capacity, dropout, class weighting, and early stopping adapt to diagnostics;
- model construction, shuffling, augmentation, dropout, and challenger
  repetitions use recorded deterministic seeds;
- checkpoints are persisted under `predictions/`, and a valid incumbent exists
  before the exploratory portfolio race starts;
- up to eight diverse baseline challengers, an independent-seed repetition,
  and an optional positional specialist receive successive-halving fidelity;
- promoted challengers continue from their best checkpoint, while unused
  budget is offered to the remaining candidates in validation order;
- prediction reserves scale with test size; a two-model ensemble is used only
  when averaged validation logits improve overall and remain robust on two
  fixed stratified halves;
- ensemble validation and prediction fall back to the single incumbent after
  errors, unsafe timing, or accelerator-memory pressure;
- training and prediction automatically reduce microbatch size after CUDA OOM.

Every expensive feature checks the live clock. The scheduler reads the
organizer-provided per-dataset clock, so the same submission adapts to short
development limits and longer final-round limits without hardcoded dataset
names or durations. A conservative prediction reserve is never allocated to
challenger training.

## Cluster run order

1. Smoke-test calibration:

   ```bash
   python3 proxy_calibration/calibrate.py \
     --datasets datasets/dataset_* --candidates 5 --updates 5
   ```

2. Run useful calibration:

   ```bash
   python3 proxy_calibration/calibrate.py \
     --datasets datasets/dataset_* --candidates 30 --updates 100
   ```

3. Inspect `proxy_calibration/results.json`, especially negative correlations
   and leave-one-dataset-out results.

4. If stable, copy the recommended weights:

   ```bash
   cp proxy_calibration/proxy_weights.json submission_template/proxy_weights.json
   ```

5. Test the exact submission at multiple metadata time limits:

   ```bash
   make submission=submission_template all
   ```

6. Package only after checking the ZIP contents:

   ```bash
   make submission=submission_template zip
   unzip -l submission.zip
   ```

Record per-dataset raw score, adjusted score, selected descriptor, proxy scores,
short-validation accuracy, runtime, peak memory, augmentation policy, and
whether any fallback was activated.

basic chnages in this commit:
Key additions:

  - Measured architectural diversity using operation, connectivity, width, depth, and parameter descriptors.
  - Farthest-first diverse finalist selection.
  - Equal-budget short training to choose between finalists.
  - Calibrated NASWOT/SynFlow weights loaded from proxy_weights.json.
  - Per-dataset Spearman, Kendall, parameter-bias, and leave-one-dataset-out calibration reports.
  - Encoded-data detection and conservative augmentation.
  - Dataset diagnostics, adaptive model capacity, dropout, class weighting, and early stopping.
  - NaN/Inf sanitization and safe constant-channel normalization.
  - Non-contiguous label mapping.
  - Persistent incumbent checkpoints.
  - Training and prediction OOM microbatch recovery.
  - Test-size-aware prediction reserve.
  - Optional two-model ensemble that disables itself if prediction timing becomes unsafe.
  - Deterministic smoke tests and cluster instructions.

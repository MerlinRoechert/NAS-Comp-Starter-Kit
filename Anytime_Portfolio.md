# Anytime portfolio trainer

The submission treats the organizer clock as the source of truth. The official
2026 final limit is 24 hours total for three datasets. If the first observed
clock is at least 18 hours, the submission safely shares the global remainder
across the datasets still to run. Shorter development metadata is treated as a
per-dataset limit without modification.

## Execution

1. Proxy scoring and equal-budget short training select a normal baseline.
2. Full training establishes a disk-backed incumbent.
3. Diverse challengers race at 5, 20, and full-fidelity rungs.
4. Each promoted model continues from its best checkpoint.
5. If time remains, eliminated candidates are revisited in validation order.
6. Sparse positional inputs receive one extra specialist; it never replaces a
   baseline candidate merely because the detector activates.
7. The best single model remains the default. A two-model ensemble is enabled
   only after averaged validation logits improve by at least 0.15 percentage
   points and do not materially regress on either stratified validation half.
8. Training stops before the prediction reserve and restores the incumbent.

For allocations above four hours, the portfolio adds three deterministic
independent-seed repetitions and permits up to 240 epochs per challenger.
Early stopping and the live clock still take precedence.

## Safety invariants

- Candidate failures and OOMs do not invalidate the incumbent.
- Only completed validation evaluations can replace the incumbent.
- The incumbent state is stored on CPU and under `predictions/`.
- The positional specialist consumes an additional slot instead of displacing
  the normal search space.
- Ensemble OOM or timing pressure falls back to one model.
- Test labels are never read or used for selection.

## Verification

Before an overnight run:

```bash
python -m unittest proxy_calibration/test_hail_mary.py
python -m py_compile submission_template/*.py
```

The log should contain `Anytime Portfolio Race`, rung summaries, a final
`Portfolio incumbent`, and either `Validated ensemble` or `Ensemble rejected`.

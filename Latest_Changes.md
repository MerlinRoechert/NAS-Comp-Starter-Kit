# Latest pipeline changes

The submission now uses a reproducible, clock-driven anytime portfolio instead
of terminating after the first fully trained architecture.

## Main changes

- Python, NumPy, Torch, CUDA, DataLoader, augmentation, and candidate seeds are
  deterministic and printed in the logs.
- A fully validated model is saved as the incumbent before exploration.
- Up to eight diverse baseline candidates, an independent-seed repetition, and
  an optional positional specialist are considered.
- Challengers receive successive-halving fidelity at 5, 20, and full-training
  rungs. Finalists alternate in ten-epoch blocks so a slow model cannot consume
  the entire remaining budget.
- If useful time remains, previously eliminated candidates are revisited.
- The Gutenberg/Language-style positional model is an additional challenger;
  it never replaces or restricts the normal search space automatically.
- Candidate failures and CUDA OOMs retain the last valid incumbent.
- A two-model ensemble is used only when averaged validation logits improve by
  at least 0.15 percentage points and remain robust on two stratified
  validation halves. Timing or memory pressure falls back to one model.

## Time-budget handling

The 2026 final permits 24 hours total for three hidden datasets. When the first
observed clock is at least 18 hours, the pipeline shares the global remainder
across the datasets still to run and keeps an outer safety margin. Shorter
development metadata continues to act as a per-dataset limit.

For allocations above four hours, three more deterministic seed repetitions
are added and challengers may train for up to 240 epochs. Early stopping, the
live clock, and the prediction reserve always take precedence.

## Verify on LUH

```bash
cd /bigwork/nhkbkpkm/NAS-Comp-Starter-Kit
module load GCCcore/.13.2.0 Python/3.11.5 CUDA/11.8.0
source venv/bin/activate

python -m unittest proxy_calibration/test_hail_mary.py
python -m py_compile submission_template/*.py
sbatch run.sh
```

Expected log markers include `Budget allocation`, `Anytime Portfolio Race`,
the three rung summaries, `Portfolio incumbent`, and either
`Validated ensemble` or `Ensemble rejected`.

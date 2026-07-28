# Proxy calibration results

## Experiment

The calibration evaluated 30 architectures per dataset, trained each for 100
updates, and compared validation accuracy rankings with NASWOT, SynFlow, and
parameter-count rankings.

Command:

```bash
python3 proxy_calibration/calibrate.py \
  --datasets datasets/* \
  --candidates 30 \
  --updates 100
```

## Correlations

| Dataset | NASWOT Spearman | SynFlow Spearman | Parameter Spearman | Interpretation |
|---|---:|---:|---:|---|
| Adaline | -0.105 | +0.464 | +0.570 | SynFlow useful; NASWOT harmful |
| Caitie | -0.237 | +0.028 | +0.025 | Neither proxy is informative |
| Gutenberg | +0.359 | +0.727 | +0.786 | SynFlow strong, but strongly size-confounded |
| LaMelo | +0.630 | +0.653 | +0.700 | Both proxies useful; size also predictive |
| Mateo | -0.101 | +0.708 | +0.726 | SynFlow strong; NASWOT harmful |

Mean per-dataset Spearman for the selected combination: **0.520**.

The grid search recommended:

```json
{
  "naswot_weight": 0.05,
  "synflow_weight": 0.95
}
```

## Leave-one-dataset-out check

| Held-out dataset | Weight selected without it | Held-out Spearman |
|---|---:|---:|
| Adaline | 0.05 NASWOT | +0.464 |
| Caitie | 0.10 NASWOT | +0.003 |
| Gutenberg | 0.05 NASWOT | +0.725 |
| LaMelo | 0.05 NASWOT | +0.671 |
| Mateo | 0.05 NASWOT | +0.713 |

The selected weighting is broadly stable: four held-out experiments chose
0.05 NASWOT and one chose 0.10. Caitie remains essentially unpredictable under
either choice.

## Conclusions

1. **SynFlow is the useful proxy.** It is positive on all five datasets and
   strong on Gutenberg, LaMelo, and Mateo.
2. **NASWOT is not robust.** It is negative on Adaline, Caitie, and Mateo.
3. **The provisional proxy mixture should be 0.05 NASWOT / 0.95 SynFlow.**
4. **Caitie should be decided mainly through short training.** Its proxy
   correlations are approximately zero.
5. **SynFlow is substantially confounded by model size.** Parameter count is
   more correlated with short-training accuracy than SynFlow on Adaline,
   Gutenberg, LaMelo, and Mateo. This supports retaining a capacity constraint
   or penalty instead of allowing SynFlow to select width without correction.

## Important rerun requirement

These results were produced before the Gutenberg/Language structural
correction. The logs show the older diagnostic format and the calibration used
the previous model family. The pipeline now detects sparse character-position
grids, limits downsampling, preserves spatial positions, narrows their default
capacity, and applies a mild size correction.

Therefore, treat `0.05 / 0.95` as **provisional**. Rerun the same 30-candidate,
100-update calibration after deploying the corrected code. Adopt
`proxy_weights.json` only if:

- Gutenberg and Language report `sequence_grid=True`;
- the SynFlow correlation remains useful under the corrected architecture;
- the recommendation remains stable across held-out datasets; and
- parameter correlation no longer dominates merely because wider models learn
  faster within 100 updates.

Until that rerun, equal-budget finalist training remains the more reliable
selection mechanism when proxies disagree.

# Proxy calibration

This offline experiment measures whether NASWOT and SynFlow rank architectures
in the same order as equal-budget short training on the five visible datasets.
It uses the exact models, proxies, preprocessing, and search descriptors from
`submission_template`.

Run from the repository root on a GPU node:

```bash
python3 proxy_calibration/calibrate.py \
  --datasets datasets/dataset_* \
  --candidates 30 \
  --updates 100 \
  --output proxy_calibration/results.json
```

For a quick smoke run, use `--candidates 5 --updates 5`. For a useful final
calibration, prefer 30–50 candidates and 100–300 updates. All architectures
within a run receive the same update count.

Run the deterministic helper/preprocessing checks with:

```bash
python3 -m unittest proxy_calibration/test_hail_mary.py
```

The runner writes:

- `results.json`: raw architecture records and per-dataset Spearman/Kendall
  correlations;
- `proxy_weights.json`: the 0.05-grid weight with the best mean per-dataset
  Spearman correlation.

Review the per-dataset correlations before adopting the weight. Then copy:

```bash
cp proxy_calibration/proxy_weights.json submission_template/proxy_weights.json
```

Do not copy `results.json` into the competition ZIP. Treat a combined Spearman
correlation near zero or below zero as evidence that proxies should only prune
the search space; the submission's equal-budget finalist training should decide
the champion.

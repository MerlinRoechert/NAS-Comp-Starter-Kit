# Budget-Aware NAS for Unseen Data

This repository contains our submission to the **NAS Unseen-Data Challenge at
AutoML 2026**. The system searches and trains neural architectures for image
datasets whose structure is unknown in advance. It combines cheap zero-cost
proxy screening with measured validation performance, architectural diversity,
and an anytime multi-fidelity portfolio.

The final development incumbent is tagged `current_best_21.430`.
The number is the best adjusted score observed on the five public development
datasets; it is not a claim about the hidden final datasets.

## Pipeline at a glance

```text
Adaptive preprocessing and dataset diagnostics
                       ↓
Random cell-based architecture sampling
                       ↓
Calibrated zero-cost screening (NASWOT + SynFlow)
                       ↓
Diversity islands and farthest-first finalist selection
                       ↓
Equal-budget short training
                       ↓
Successive-halving portfolio (5 → 20 → full epochs)
                       ↓
Safe validation incumbent + optional validated ensemble
                       ↓
Final predictions
```

The central design rule is simple: **proxies decide what is cheap enough to
explore; validation decides what is safe to keep**. A trained incumbent exists
before further exploration begins, and candidate failures, memory pressure, or
an expiring clock fall back to the last valid model.

## Main components

- **Adaptive preprocessing.** Input diagnostics distinguish continuous imagery
  from sparse or positionally encoded data. Augmentation is conservative and
  disabled when it could destroy encoded information.
- **Cell-based NAS.** Random candidates vary operations, connectivity, depth,
  width, and parameter count.
- **Calibrated proxies.** Candidate screening uses normalized NASWOT and
  SynFlow scores with calibration-derived weights of `0.05 / 0.95`.
- **Diversity-aware selection.** Architectural islands and descriptor distance
  prevent a single proxy-favoured family from occupying the entire portfolio.
- **Measured finalist ranking.** Diverse finalists receive equal short-training
  budgets before a primary model is selected.
- **Anytime portfolio racing.** Challengers advance through increasing
  fidelities while the incumbent remains available at every point.
- **Conditional specialization.** A compact position-preserving architecture is
  added only as a challenger on sparse positional inputs; validation may accept
  or reject it independently for each dataset.
- **Safe ensembling.** A complementary second model is used only when averaged
  validation logits improve by a fixed margin and remain robust on two
  stratified validation halves.

The competition entry points are implemented in
[`submission_template/`](submission_template):

| File | Responsibility |
|---|---|
| [`data_processor.py`](submission_template/data_processor.py) | diagnostics, preprocessing, augmentation, loaders, and budget allocation |
| [`nas.py`](submission_template/nas.py) | architecture sampling, proxies, diversity, and finalist selection |
| [`trainer.py`](submission_template/trainer.py) | training, successive halving, incumbent management, ensembling, and prediction |
| [`helpers.py`](submission_template/helpers.py) | cells, models, proxy utilities, descriptors, and robust selection helpers |

## Key experimental findings

We calibrated the proxies on 30 architectures per dataset, each trained for
100 updates. Spearman correlation compares each cheap ranking with validation
accuracy:

| Dataset | NASWOT | SynFlow | Parameters |
|---|---:|---:|---:|
| Adaline | -0.105 | 0.464 | 0.570 |
| Caitie | -0.237 | 0.028 | 0.025 |
| Gutenberg | 0.359 | 0.727 | 0.786 |
| LaMelo | 0.630 | 0.653 | 0.700 |
| Mateo | -0.101 | 0.708 | 0.726 |

NASWOT was negatively correlated on three datasets. SynFlow was more useful,
but was often confounded with parameter count and provided almost no signal on
Caitie. This motivated using proxies only as filters and letting diversity plus
short training determine promotion.

The complete portfolio pipeline improved the best public development score
from `18.081` to `21.429`:

| Scored dataset | Before | Portfolio pipeline | Change |
|---|---:|---:|---:|
| MultNIST | 4.458 | 4.545 | +0.087 |
| CIFARTile | 6.271 | 6.194 | -0.077 |
| Language | 1.372 | 2.392 | +1.020 |
| Gutenberg | -0.059 | 1.658 | +1.717 |
| AddNIST | 6.039 | 6.640 | +0.601 |
| **Total** | **18.081** | **21.429** | **+3.348** |

The largest gains came from the two position-sensitive datasets. The specialist
won the Gutenberg portfolio but was rejected on Language, illustrating why
specialization is offered as a validation-gated challenger rather than imposed
as a dataset-name rule. Repeated full runs also showed substantial architecture
and seed variance, so the tagged result should be read as a best development
incumbent rather than a deterministic benchmark.

## Reproduce a development run

Use the Python and CUDA versions supplied by the challenge environment, install
the repository requirements, and place datasets below `datasets/` using the
starter-kit layout (`train_x.npy`, `train_y.npy`, validation/test arrays, and
`metadata`). Then run:

```bash
python -m pip install -r requirements.txt
make submission=submission_template all
```

Build the submission archive with:

```bash
make submission=submission_template zip
```

Run the deterministic smoke tests with:

```bash
python -m unittest proxy_calibration/test_hail_mary.py
python -m py_compile submission_template/*.py
```

## Experiments and documentation

- [`Proxy_Calibration_Results.md`](Proxy_Calibration_Results.md) summarizes the
  proxy study; [`proxy_calibration/`](proxy_calibration) contains the scripts.
- [`Anytime_Portfolio.md`](Anytime_Portfolio.md) explains the multi-fidelity
  portfolio and its safety invariants.
- [`Gutenberg_Language.md`](Gutenberg_Language.md) records the positional-data
  investigation.
- [`Latest_Changes.md`](Latest_Changes.md) gives a concise implementation
  history of the final pipeline.
- [`bo/`](bo) contains exploratory SMAC tooling. BO configurations are
  validation experiments and never modify the submission automatically.

## Starter-kit provenance

This project builds on the official NAS Unseen-Data starter kit. The unchanged
evaluation interface expects three components—`DataProcessor`, `NAS`, and
`Trainer`—and supplies a live clock that submissions must respect. The local
[`evaluation/`](evaluation) and [`Makefile`](Makefile) reproduce that interface
for development; official evaluation uses organizer-controlled scripts and
hidden datasets.

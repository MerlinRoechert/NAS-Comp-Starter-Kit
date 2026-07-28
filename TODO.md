# Team guide: proxy calibration and BO

Run commands from the repository root. Use visible development validation
labels only—never test labels. Replace all example dataset names and
`/bigwork/PROJECT/...` paths before submitting jobs.

## 1. Prepare once

The repository and datasets must be available at the same shared paths on all
workers. Each dataset needs `metadata`, train/validation arrays, and
`test_x.npy`.

Use a Python 3.11/3.12 environment containing the competition dependencies and
the BO dependencies:

```bash
module load Miniforge3
conda activate nas_bo
python -m pip install -r bo/requirements.txt
python -c "import torch, smac, ConfigSpace, distributed; print(torch.cuda.is_available())"
python -m unittest proxy_calibration/test_hail_mary.py
```

`torch.cuda.is_available()` should be `True` inside a GPU allocation.

## 2. Calibrate NASWOT and SynFlow

Calibration compares proxy rankings with equal-update short-training rankings
on several development datasets.

First run a smoke test in a separate output directory:

```bash
python proxy_calibration/calibrate.py \
  --datasets datasets/dataset_a datasets/dataset_b \
  --candidates 5 \
  --updates 5 \
  --output proxy_calibration/smoke/results.json
```

Do not adopt smoke-test weights. The useful experiment is:

```bash
python proxy_calibration/calibrate.py \
  --datasets \
    datasets/dataset_a \
    datasets/dataset_b \
    datasets/dataset_c \
    datasets/dataset_d \
    datasets/dataset_e \
  --candidates 30 \
  --updates 100 \
  --proxy-batch 24 \
  --seed 42 \
  --output proxy_calibration/results.json
```

Run this on one LUH A100 using the same Slurm header as the BO workers:

```bash
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=3:00:00
```

Inspect:

```bash
python -m json.tool proxy_calibration/proxy_weights.json
```

Adopt a weighting only when correlations are positive and reasonably stable
across individual and leave-one-dataset-out results:

```bash
cp proxy_calibration/proxy_weights.json \
  submission_template/proxy_weights.json
```

If correlations are near zero, negative, or dataset-dependent, retain the
default weighting and rely on equal-budget finalist training.

## 3. Run the 100-trial BO

The study uses:

- 25 Sobol space-filling initial configurations;
- 75 model-guided SMAC configurations;
- up to 25 asynchronous one-A100 workers;
- a small mixed-space diversity filter against the 25 most recent proposals;
- mean validation adjusted score across the selected datasets.

The diversity filter rejects near-duplicates below normalized distance `0.08`.
It never modifies the initial Sobol design and has a retry escape to avoid
stalling.

### One-worker smoke test

```bash
python -m bo.run_smac \
  --trials 1 \
  --initial-sobol 1 \
  --workers 1 \
  --budget-minutes 2 \
  --train-limit 1024 \
  --datasets dataset_a \
  --luh-slurm \
  --slurm-conda-env nas_bo \
  --output bo/smoke-output
```

Confirm that the worker finishes and `bo/smoke-output/incumbent.json` exists.
Do not request 25 workers until this passes.

### Full run

```bash
python -m bo.run_smac \
  --trials 100 \
  --initial-sobol 25 \
  --workers 25 \
  --budget-minutes 15 \
  --datasets dataset_a dataset_b dataset_c \
  --luh-slurm \
  --slurm-partition ai \
  --slurm-gpu a100 \
  --slurm-cpus 2 \
  --slurm-memory 16GiB \
  --slurm-walltime 03:00:00 \
  --slurm-conda-env nas_bo \
  --output bo/output
```

Alternatively, submit the coordinator:

```bash
export NAS_BO_REPOSITORY=/bigwork/PROJECT/path/NAS-Comp-Starter-Kit
export NAS_BO_CONDA_ENV=nas_bo
export NAS_BO_DATASETS="dataset_a dataset_b dataset_c"
export NAS_BO_BUDGET_MINUTES=15
sbatch bo/submit_luh_coordinator.sh
```

Approximate compute usage is:

```text
trials × datasets × minutes / 60
```

Thus, 100 trials × 3 datasets × 15 minutes is approximately 75 GPU-hours.

Monitor with:

```bash
squeue -u "$USER"
find bo/output/slurm-logs -maxdepth 1 -type f -print
```

The result is `bo/output/incumbent.json`. Treat it as a candidate: compare it
against the current incumbent at full budget, on every development dataset,
and with additional seeds before changing submission defaults.

## 4. Record for every run

- Git commit hash and person launching
- Dataset list and seed
- Environment and Slurm job ID
- Candidate/trial count and budget
- Output directory
- Per-dataset scores, failures, OOMs, and runtime
- Final decision: adopt, reject, or rerun

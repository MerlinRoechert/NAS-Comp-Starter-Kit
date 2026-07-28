# SMAC low-budget optimization

This directory is development tooling and is not included in the competition
submission. It runs 100 low-budget pipeline evaluations, beginning with a
25-configuration Sobol initial design. By default, SMAC maximizes the mean
validation-set adjusted score across the selected development datasets using
the same benchmark normalization as `evaluation/score.py`. Test labels are
never read.

## Install

Use Python 3.11 or 3.12 in a separate environment from the pinned competition
environment. The BO dependencies are development-only, and SMAC 2.x is not
compatible with every bleeding-edge Python/scikit-learn combination.

```bash
python3.12 -m venv .venv-bo
. .venv-bo/bin/activate
python -m pip install -r bo/requirements.txt
```

## Smoke test

Before using the cluster, validate one short run:

```bash
python -m bo.run_smac \
  --trials 1 \
  --initial-sobol 1 \
  --workers 1 \
  --budget-minutes 2 \
  --train-limit 1024 \
  --datasets dataset_name
```

## Local run

Use one worker unless local GPU assignment is managed externally:

```bash
python -m bo.run_smac \
  --trials 100 \
  --initial-sobol 25 \
  --workers 1 \
  --budget-minutes 15 \
  --datasets dataset_a dataset_b dataset_c
```

Without `--scheduler-address`, workers are local processes. Multiple processes
on a single-GPU machine will compete for the same GPU.

The incumbent is written to `bo/output/incumbent.json`; SMAC's full runhistory
is stored below the same output directory. Run on multiple representative
development datasets to reduce specialization to one visible dataset. Because
the budget is per dataset, the approximate compute consumed is
`trials * datasets * budget`.

## Cluster execution

The cluster must provide one GPU per Dask worker and make the repository and
datasets visible at the same paths on every worker. If your teammates already
start a Dask cluster through the LUH scheduler, connect the coordinator with:

```bash
python -m bo.run_smac \
  --trials 100 \
  --initial-sobol 25 \
  --workers 25 \
  --budget-minutes 15 \
  --datasets dataset_a dataset_b dataset_c \
  --scheduler-address tcp://SCHEDULER_HOST:SCHEDULER_PORT
```

Do not guess the LUH account, partition, GPU resource syntax, module commands,
or whether jobs must be submitted with `sbatch`. Obtain those values from the
cluster documentation or a working teammate job script. If the scheduler is
Slurm, `dask-jobqueue` can create the workers, but its `SLURMCluster` parameters
must match those site-specific values.

SMAC documents native Dask parallelism and accepts a custom Dask client. On
some Slurm installations, creating new scheduler jobs from inside a compute job
can hang, so confirm whether the coordinator should run on the login node or in
an allocation before launching the 100-trial study.

## LUH `ai` partition

The included LUH backend uses the confirmed teammate settings: partition `ai`,
one A100, two CPU cores, 16 GiB memory, and a three-hour worker walltime. Unlike
a fixed Slurm array, SMAC proposes new configurations after earlier results and
dask-jobqueue maintains at most 25 worker jobs. A normalized mixed-space
distance filter rejects near-duplicates among the 25 most recent proposals:

```bash
python -m bo.run_smac \
  --trials 100 \
  --initial-sobol 25 \
  --workers 25 \
  --budget-minutes 15 \
  --datasets dataset_a dataset_b dataset_c \
  --luh-slurm \
  --slurm-conda-env nas_bo
```

Run the coordinator from the shared repository checkout. If LUH policy requires
the coordinator itself to be submitted, use:

```bash
export NAS_BO_REPOSITORY=/bigwork/PROJECT/path/NAS-Comp-Starter-Kit
export NAS_BO_CONDA_ENV=nas_bo
export NAS_BO_DATASETS="dataset_a dataset_b dataset_c"
sbatch bo/submit_luh_coordinator.sh
```

The coordinator job does not request a GPU. It submits up to 25 separate GPU
worker jobs. Check them with `squeue -u "$USER"` and inspect worker logs below
`bo/output/slurm-logs/`.

Do a one-worker smoke test before requesting all 25 workers:

```bash
python -m bo.run_smac \
  --trials 1 --initial-sobol 1 --workers 1 \
  --budget-minutes 2 --train-limit 1024 \
  --datasets dataset_a \
  --luh-slurm --slurm-conda-env nas_bo
```

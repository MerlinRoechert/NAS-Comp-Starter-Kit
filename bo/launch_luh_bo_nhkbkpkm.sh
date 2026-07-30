#!/bin/bash -l

# Submit the time-constrained targeted SMAC study from a LUH login node.
# This script changes no competition-submission files.

set -euo pipefail

repository="/bigwork/nhkbkpkm/NAS-Comp-Starter-Kit"
virtualenv="${repository}/venv"
datasets_root="${repository}/datasets"

if [[ ! -f "${virtualenv}/bin/activate" ]]; then
    echo "Virtual environment not found: ${virtualenv}" >&2
    exit 2
fi
if [[ ! -d "${datasets_root}" ]]; then
    echo "Dataset directory not found: ${datasets_root}" >&2
    exit 2
fi

cd "${repository}"
source "${virtualenv}/bin/activate"

python -c "import torch, smac, ConfigSpace, distributed, dask_jobqueue" ||
{
    echo "BO dependencies are missing. On a login node with internet, run:" >&2
    echo "source ${virtualenv}/bin/activate" >&2
    echo "python -m pip install -r ${repository}/bo/requirements.txt" >&2
    exit 2
}

dataset_names="$(
    find "${datasets_root}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; |
        sort |
        paste -sd ' ' -
)"
if [[ -z "${dataset_names}" ]]; then
    echo "No dataset directories found below ${datasets_root}" >&2
    exit 2
fi

export NAS_BO_REPOSITORY="${repository}"
export NAS_BO_VENV="${virtualenv}"
unset NAS_BO_CONDA_ENV
export NAS_BO_DATASETS_ROOT="${datasets_root}"
export NAS_BO_DATASETS="${dataset_names}"
# Four approximate waves: 8 space-filling starts, then 24 BO proposals.
# Eight GPUs are deliberately requested instead of 25 to reduce queue latency.
export NAS_BO_TRIALS=32
export NAS_BO_INITIAL_SOBOL=8
export NAS_BO_WORKERS=8
export NAS_BO_BUDGET_MINUTES=15
run_id="$(date +%Y%m%d_%H%M%S)"
export NAS_BO_OUTPUT="bo/output/targeted_4hp_${run_id}"

# Match the resource request from the known-good normal submission job.
export NAS_BO_SLURM_PARTITION=gpu
export NAS_BO_SLURM_GPU=""
export NAS_BO_SLURM_CPUS=4
export NAS_BO_SLURM_MEMORY=32GiB
export NAS_BO_SLURM_WALLTIME=03:00:00

mkdir -p "${NAS_BO_OUTPUT}"

submission="$(
    sbatch \
        --partition=gpu \
        --time=08:00:00 \
        --mem=8G \
        --export=ALL \
        bo/submit_luh_coordinator.sh
)"

echo "${submission}"
echo "Monitor with: squeue -u nhkbkpkm"
echo "Coordinator log: bo/coordinator_JOBID.out"
echo "Run directory: ${NAS_BO_OUTPUT}"
echo "Final result: ${NAS_BO_OUTPUT}/incumbent.json"

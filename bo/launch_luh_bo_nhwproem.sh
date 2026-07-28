#!/bin/bash -l

set -euo pipefail

repository="${NAS_BO_REPOSITORY:-/bigwork/nhwproem/NAS-Comp-Starter-Kit}"
virtualenv="${NAS_BO_VENV:-}"
conda_environment="${NAS_BO_CONDA_ENV:-nas_bo}"
datasets_root="${NAS_BO_DATASETS_ROOT:-${repository}/datasets}"

if [[ ! -d "${repository}" ]]; then
    echo "Repository not found: ${repository}" >&2
    exit 2
fi
if [[ -n "${virtualenv}" && ! -f "${virtualenv}/bin/activate" ]]; then
    echo "Virtual environment not found: ${virtualenv}" >&2
    exit 2
fi
if [[ ! -d "${datasets_root}" ]]; then
    echo "Dataset directory not found: ${datasets_root}" >&2
    echo "Locate it with: find /bigwork/nhwproem -maxdepth 7 -type f -name train_x.npy 2>/dev/null" >&2
    echo "Then set NAS_BO_DATASETS_ROOT to the parent containing the dataset directories." >&2
    exit 2
fi

cd "${repository}"

dataset_names="$(
    find "${datasets_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        sort |
        paste -sd ' ' -
)"
if [[ -z "${dataset_names}" ]]; then
    echo "No dataset directories found below ${datasets_root}" >&2
    exit 2
fi

export NAS_BO_REPOSITORY="${repository}"
export NAS_BO_DATASETS_ROOT="${datasets_root}"
export NAS_BO_DATASETS="${dataset_names}"
export NAS_BO_BUDGET_MINUTES="${NAS_BO_BUDGET_MINUTES:-15}"

if [[ -n "${virtualenv}" ]]; then
    export NAS_BO_VENV="${virtualenv}"
    unset NAS_BO_CONDA_ENV
    source "${NAS_BO_VENV}/bin/activate"
    environment_description="venv ${NAS_BO_VENV}"
else
    unset NAS_BO_VENV
    export NAS_BO_CONDA_ENV="${conda_environment}"
    module load Miniforge3
    conda activate "${NAS_BO_CONDA_ENV}"
    environment_description="Conda ${NAS_BO_CONDA_ENV}"
fi

python -c \
    "import torch, smac, ConfigSpace, distributed, dask_jobqueue; print('CUDA build:', torch.version.cuda)"

mkdir -p bo/output bo/output/slurm-logs

echo "Repository: ${NAS_BO_REPOSITORY}"
echo "Environment: ${environment_description}"
echo "Data root:   ${NAS_BO_DATASETS_ROOT}"
echo "Datasets:    ${NAS_BO_DATASETS}"
echo "Budget:      ${NAS_BO_BUDGET_MINUTES} minutes per dataset"

job_submission="$(
    sbatch --export=ALL bo/submit_luh_coordinator.sh
)"
echo "${job_submission}"
echo "Monitor with: squeue -u nhwproem"

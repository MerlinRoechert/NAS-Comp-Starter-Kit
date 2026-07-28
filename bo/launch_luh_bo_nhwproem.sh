#!/bin/bash -l

set -euo pipefail

repository="/bigwork/nhwproem/NAS-Comp-Starter-Kit"
virtualenv="${repository}/venv"

if [[ ! -d "${repository}" ]]; then
    echo "Repository not found: ${repository}" >&2
    exit 2
fi
if [[ ! -f "${virtualenv}/bin/activate" ]]; then
    echo "Virtual environment not found: ${virtualenv}" >&2
    exit 2
fi
if [[ ! -d "${repository}/datasets" ]]; then
    echo "Dataset directory not found: ${repository}/datasets" >&2
    exit 2
fi

cd "${repository}"

dataset_names="$(
    find datasets -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        sort |
        paste -sd ' ' -
)"
if [[ -z "${dataset_names}" ]]; then
    echo "No dataset directories found below ${repository}/datasets" >&2
    exit 2
fi

export NAS_BO_REPOSITORY="${repository}"
export NAS_BO_VENV="${virtualenv}"
export NAS_BO_DATASETS="${dataset_names}"
export NAS_BO_BUDGET_MINUTES="${NAS_BO_BUDGET_MINUTES:-15}"

source "${NAS_BO_VENV}/bin/activate"

python -c \
    "import torch, smac, ConfigSpace, distributed, dask_jobqueue; print('CUDA build:', torch.version.cuda)"

mkdir -p bo/output bo/output/slurm-logs

echo "Repository: ${NAS_BO_REPOSITORY}"
echo "Virtualenv:  ${NAS_BO_VENV}"
echo "Datasets:    ${NAS_BO_DATASETS}"
echo "Budget:      ${NAS_BO_BUDGET_MINUTES} minutes per dataset"

job_submission="$(
    sbatch --export=ALL bo/submit_luh_coordinator.sh
)"
echo "${job_submission}"
echo "Monitor with: squeue -u nhwproem"

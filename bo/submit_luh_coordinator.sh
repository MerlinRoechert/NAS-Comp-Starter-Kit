#!/bin/bash -l
#
# Submit this small coordinator only if LUH policy does not permit running the
# SMAC/Dask coordinator on a login node. GPU worker jobs are created separately
# by dask-jobqueue with the resources configured in bo/run_smac.py.
#
#SBATCH --job-name=nas_smac_coordinator
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=bo/coordinator_%j.out
#SBATCH --error=bo/coordinator_%j.err

set -euo pipefail

if [[ -z "${NAS_BO_REPOSITORY:-}" ]]; then
    echo "Set NAS_BO_REPOSITORY to the shared repository path." >&2
    exit 2
fi
if [[ -z "${NAS_BO_VENV:-}" && -z "${NAS_BO_CONDA_ENV:-}" ]]; then
    echo "Set NAS_BO_VENV or NAS_BO_CONDA_ENV." >&2
    exit 2
fi
if [[ -z "${NAS_BO_DATASETS:-}" ]]; then
    echo "Set NAS_BO_DATASETS to a space-separated dataset list." >&2
    exit 2
fi
if [[ -z "${NAS_BO_DATASETS_ROOT:-}" ]]; then
    echo "Set NAS_BO_DATASETS_ROOT to the shared dataset directory." >&2
    exit 2
fi

worker_environment_args=()
if [[ -n "${NAS_BO_VENV:-}" ]]; then
    module load GCCcore/.13.2.0 Python/3.11.5 CUDA/11.8.0
    source "${NAS_BO_VENV}/bin/activate"
    worker_environment_args=(--slurm-venv "${NAS_BO_VENV}")
else
    module load Miniforge3
    conda activate "${NAS_BO_CONDA_ENV}"
    worker_environment_args=(--slurm-conda-env "${NAS_BO_CONDA_ENV}")
fi
cd "${NAS_BO_REPOSITORY}"
output_directory="${NAS_BO_OUTPUT:-bo/output}"
mkdir -p "${output_directory}"

read -r -a dataset_names <<< "${NAS_BO_DATASETS}"

python -m bo.run_smac \
    --trials "${NAS_BO_TRIALS:-100}" \
    --initial-sobol "${NAS_BO_INITIAL_SOBOL:-25}" \
    --workers "${NAS_BO_WORKERS:-25}" \
    --budget-minutes "${NAS_BO_BUDGET_MINUTES:-15}" \
    --output "${output_directory}" \
    --datasets-root "${NAS_BO_DATASETS_ROOT}" \
    --datasets "${dataset_names[@]}" \
    --luh-slurm \
    --slurm-partition "${NAS_BO_SLURM_PARTITION:-ai}" \
    --slurm-gpu "${NAS_BO_SLURM_GPU:-a100}" \
    --slurm-cpus "${NAS_BO_SLURM_CPUS:-4}" \
    --slurm-memory "${NAS_BO_SLURM_MEMORY:-32GiB}" \
    --slurm-walltime "${NAS_BO_SLURM_WALLTIME:-03:00:00}" \
    "${worker_environment_args[@]}"

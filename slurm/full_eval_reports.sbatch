#!/bin/bash
# Builds results/tables/*.md and results/figures/*.png from whatever
# results/*.json exist. Submitted by slurm/submit_full_eval.sh to run after
# the models and baselines jobs finish (whether or not they succeeded —
# make_tables.py just shows "-" for any missing combo).
#SBATCH --job-name=deepanc-full-eval-reports
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

CONDA_ENV="${CONDA_ENV:-deepanc}"
set +u
module purge
module load MLDL/miniconda3
conda activate "${CONDA_ENV}"
set -u

python scripts/make_tables.py
python scripts/make_figures.py

set +u
conda deactivate
set -u
echo "reports done: results/tables/, results/figures/"

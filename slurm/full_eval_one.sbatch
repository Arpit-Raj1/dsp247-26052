#!/bin/bash
# Runs scripts/run_full_eval_local.sh as ONE SLURM job — the same
# sequential "skip what already exists" sweep, but as a batch process that
# keeps running after you log out (a login/interactive-node process does
# not, per the cluster's rules).
#
#   sbatch slurm/full_eval_one.sbatch
#
# Resumable: each result is written as soon as it finishes and existing
# results/*.json are skipped, so if this hits its walltime, just sbatch it
# again. Defaults to the cpu partition since models are normally already
# evaluated and what's left is baselines. If model results are missing too,
# run it on a GPU instead:
#   sbatch --partition=gpu --gres=gpu:1 slurm/full_eval_one.sbatch
#SBATCH --job-name=deepanc-full-eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm_logs results

CONDA_ENV="${CONDA_ENV:-deepanc}"
set +u
module purge
module load MLDL/miniconda3
conda activate "${CONDA_ENV}"
set -u

bash scripts/run_full_eval_local.sh

set +u
conda deactivate
set -u

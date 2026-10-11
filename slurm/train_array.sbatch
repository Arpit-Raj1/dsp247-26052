#!/bin/bash
# Trains all 9 model variants as an independent SLURM job array on PARAM
# Rudra's `gpu` partition (8 nodes total cluster-wide, 2xA100/node) — one
# GPU per task, no inter-task communication needed.
#
# Usage: sbatch slurm/train_array.sbatch
#
# Note: PARAM Rudra's QoS caps 10 simultaneous jobs per user by default,
# so this array (9 tasks) fits without hitting that limit alongside other
# work; if you're running baselines_array.sbatch or eval_array.sbatch at
# the same time, add `%N` to --array below to throttle concurrency, e.g.
# --array=0-8%4 runs at most 4 at once.
#SBATCH --job-name=deepanc-train-array
#SBATCH --partition=gpu
#SBATCH --array=0-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
mkdir -p slurm_logs

CONFIGS=(
  configs/exp/crn_n.yaml
  configs/exp/crn_n_m1.yaml
  configs/exp/crn_n_m2.yaml
  configs/exp/crn_ns.yaml
  configs/exp/crn_ns_m1.yaml
  configs/exp/crn_ns_m2.yaml
  configs/exp/crn_n_qz.yaml
  configs/exp/crn_ns_qz.yaml
  configs/exp/crn_n_nl.yaml
)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID -> $CONFIG"

CONDA_ENV="${CONDA_ENV:-deepanc}"
SCRATCH_DATA="/scratch/${USER}/deepanc_data"
mkdir -p "${SCRATCH_DATA}"
cp -n data/*.h5 data/*.npz "${SCRATCH_DATA}/" 2>/dev/null || true
DATA_OVERRIDE="data.noise_train_h5=${SCRATCH_DATA}/$(basename data/noise_train.h5) \
               data.noise_val_h5=${SCRATCH_DATA}/$(basename data/noise_val.h5) \
               data.timit_h5=${SCRATCH_DATA}/$(basename data/timit.h5) \
               data.rirs_npz=${SCRATCH_DATA}/$(basename data/rirs.npz)"

# module/conda's own scripts (e.g. qt-main's conda deactivate hook)
# reference variables they never set under `set -u`, so relax nounset
# just around them rather than patching admin-owned module files.
set +u
module purge
module load MLDL/miniconda3
conda activate "${CONDA_ENV}"
set -u

srun python -m deepanc.train --config "${CONFIG}" ${DATA_OVERRIDE}

set +u
conda deactivate
set -u

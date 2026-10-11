#!/bin/bash
# Single-model training job on PARAM Rudra (IIT Patna).
# Usage: sbatch slurm/train.sbatch configs/exp/crn_n.yaml [overrides...]
#
# --signal=USR1@120: SLURM sends SIGUSR1 120s before the job's time limit;
# train.py catches it, checkpoints, and exits cleanly so --requeue (or a
# manual `scontrol requeue $SLURM_JOB_ID`) resumes from that checkpoint.
#SBATCH --job-name=deepanc-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@120
#SBATCH --requeue
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# Uncomment and set if you belong to more than one SLURM account
# (check with: sacctmgr show associations -p --user=$USER):
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
mkdir -p slurm_logs

CONFIG="$1"
shift
OVERRIDES="$@"

CONDA_ENV="${CONDA_ENV:-deepanc}"

# /scratch/<username> is fast node-local-ish storage on PARAM Rudra (not
# purged for 3 months of inactivity, unlike a hard weekly wipe on some
# clusters, but still meant as working storage, not a place to leave final
# results — copy checkpoints/results back to $HOME when a run finishes).
SCRATCH_DATA="/scratch/${USER}/deepanc_data"
mkdir -p "${SCRATCH_DATA}"
echo "staging data to ${SCRATCH_DATA}"
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

srun python -m deepanc.train --config "${CONFIG}" ${DATA_OVERRIDE} ${OVERRIDES}

set +u
conda deactivate
set -u

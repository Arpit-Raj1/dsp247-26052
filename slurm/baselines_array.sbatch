#!/bin/bash
# Evaluates every classical baseline (fxlms, thf_fxlms, vfxlms, fxmlp)
# against every test-set group, on the `cpu` partition — these are
# numba/CPU sample-by-sample algorithms, no GPU needed, and the `cpu`
# partition (122 nodes) has far more free capacity than `gpu` (8 nodes).
#
# Usage: sbatch slurm/baselines_array.sbatch
#
# The %4 on --array throttles this to 4 array tasks counted as
# submitted/running at once; without it, all 28 tasks count individually
# against the account's MaxSubmitJobsPerUser QOS limit, which can leave
# some tasks silently never submitted (rather than sbatch refusing the
# whole array) depending on how close to the limit the account already
# is. If you already ran this once unthrottled and results/ has fewer
# than 28 *.json files, the missing ones simply never got a chance to
# run — this fix doesn't retroactively submit them, re-run the whole
# array (already-written results/*.json are untouched either way; rerun
# is idempotent, evaluate.py just overwrites each file with the same
# numbers).
#SBATCH --job-name=deepanc-baselines-array
#SBATCH --partition=cpu
#SBATCH --array=0-27%4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
mkdir -p slurm_logs results

BASELINES=(fxlms thf_fxlms vfxlms fxmlp)
TEST_GROUPS=(table1 table2_noise table2_snr fig8 fig9 quiet_zone table4)

N_GROUPS=${#TEST_GROUPS[@]}
B_IDX=$(( SLURM_ARRAY_TASK_ID / N_GROUPS ))
G_IDX=$(( SLURM_ARRAY_TASK_ID % N_GROUPS ))
BASELINE="${BASELINES[$B_IDX]}"
GROUP="${TEST_GROUPS[$G_IDX]}"
echo "task $SLURM_ARRAY_TASK_ID -> baseline=$BASELINE group=$GROUP"

CONDA_ENV="${CONDA_ENV:-deepanc}"
# module/conda's own scripts (e.g. qt-main's conda deactivate hook)
# reference variables they never set under `set -u`, so relax nounset
# just around them rather than patching admin-owned module files.
set +u
module purge
module load MLDL/miniconda3
conda activate "${CONDA_ENV}"
set -u

srun python -m deepanc.evaluate \
  --baseline "${BASELINE}" \
  --mu 0.05 \
  --testsets data/testsets.h5 \
  --group "${GROUP}" \
  --out "results/${BASELINE}_${GROUP}.json"

set +u
conda deactivate
set -u

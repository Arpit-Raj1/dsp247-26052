#!/bin/bash
# Evaluates every trained model checkpoint against every test-set group
# (Table 1/2/4, Figs 8/9, quiet_zone) as a job array — one task per
# (model, group) pair. Requires runs/<exp>/best.pt to already exist for
# each model (i.e. train_array.sbatch has completed).
#
# Usage: sbatch slurm/eval_array.sbatch
#
# The %4 on --array throttles this to 4 array tasks counted as
# submitted/running at once; without it, all 63 tasks count individually
# against the account's MaxSubmitJobsPerUser QOS limit and sbatch refuses
# the whole submission with QOSMaxSubmitJobPerUserLimit. Lower it further
# (e.g. %2) if you're also running other arrays concurrently and still
# hit the limit, or raise it if `sacctmgr show assoc -p --user=$USER`
# shows your account allows more.
#SBATCH --job-name=deepanc-eval-array
#SBATCH --partition=gpu
#SBATCH --array=0-62%4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
# #SBATCH --account=<group-slurm-account>

set -euo pipefail
mkdir -p slurm_logs results

MODELS=(crn_n crn_n_m1 crn_n_m2 crn_ns crn_ns_m1 crn_ns_m2 crn_n_qz crn_ns_qz crn_n_nl)
TEST_GROUPS=(table1 table2_noise table2_snr fig8 fig9 quiet_zone table4)

N_GROUPS=${#TEST_GROUPS[@]}
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_GROUPS ))
GROUP_IDX=$(( SLURM_ARRAY_TASK_ID % N_GROUPS ))
MODEL="${MODELS[$MODEL_IDX]}"
GROUP="${TEST_GROUPS[$GROUP_IDX]}"
echo "task $SLURM_ARRAY_TASK_ID -> model=$MODEL group=$GROUP"

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
  --checkpoint "runs/${MODEL}/best.pt" \
  --config "configs/exp/${MODEL}.yaml" \
  --testsets data/testsets.h5 \
  --group "${GROUP}" \
  --out "results/${MODEL}_${GROUP}.json"

set +u
conda deactivate
set -u

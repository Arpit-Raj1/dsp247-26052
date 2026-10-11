#!/bin/bash
# All 9 models x 7 test groups, sequentially, in ONE gpu job (no job array,
# so no QOSMaxSubmitJobPerUserLimit). Skips any results/<name>_<group>.json
# that already exists, and each result is written as soon as it finishes,
# so if this hits its walltime just resubmit — it resumes where it stopped.
#
# Normally submitted via slurm/submit_full_eval.sh.
#SBATCH --job-name=deepanc-full-eval-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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

MODELS=(crn_n crn_n_m1 crn_n_m2 crn_ns crn_ns_m1 crn_ns_m2 crn_n_qz crn_ns_qz crn_n_nl)
# not named GROUPS: reserved bash variable (see eval_array.sbatch fix)
# Table 5's nonlinear setup (Section 5.5) is the only place crn_n_nl,
# VFxLMS and FxMLP appear, and table5 is meaningless for everything else.
groups_for() {
  case "$1" in
    crn_n_nl|vfxlms|fxmlp) echo table5 ;;
    *) echo table1 table2_noise table2_snr fig8 fig9 quiet_zone table4 ;;
  esac
}

for model in "${MODELS[@]}"; do
  ckpt="runs/${model}/best.pt"
  if [ ! -f "$ckpt" ]; then
    echo "skip all groups for $model: $ckpt not found"
    continue
  fi
  for group in $(groups_for "$model"); do
    out="results/${model}_${group}.json"
    if [ -f "$out" ]; then
      echo "skip $out (already exists)"
      continue
    fi
    echo "--- model=$model group=$group ---"
    python -m deepanc.evaluate --checkpoint "$ckpt" --config "configs/exp/${model}.yaml" \
      --testsets data/testsets.h5 --group "$group" --out "$out"
  done
done

set +u
conda deactivate
set -u
echo "models done"

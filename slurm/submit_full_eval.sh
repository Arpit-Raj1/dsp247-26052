#!/bin/bash
# The single cluster command for the full evaluation sweep — cluster
# counterpart of scripts/run_full_eval_local.sh:
#
#   slurm/submit_full_eval.sh
#
# Submits exactly 3 jobs (well under the ~10-job QOS cap, and no job
# arrays, so no QOSMaxSubmitJobPerUserLimit):
#   1. models     (gpu)  9 models x 7 groups
#   2. baselines  (cpu)  4 baselines x 7 groups, combos run in parallel
#   3. reports    (cpu)  make_tables.py + make_figures.py, runs after 1 and 2
#
# Everything skips results/*.json that already exist, so it's safe to
# re-run after a walltime cutoff or partial failure — it only does what's
# still missing. Run it from the repo root (or anywhere; it cd's there).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p slurm_logs results

MODELS_JOB=$(sbatch --parsable slurm/full_eval_models.sbatch)
BASELINES_JOB=$(sbatch --parsable slurm/full_eval_baselines.sbatch)
# --parsable may append ";cluster" on multi-cluster setups
MODELS_JOB="${MODELS_JOB%%;*}"
BASELINES_JOB="${BASELINES_JOB%%;*}"
REPORTS_JOB=$(sbatch --parsable --dependency="afterany:${MODELS_JOB}:${BASELINES_JOB}" \
  slurm/full_eval_reports.sbatch)
REPORTS_JOB="${REPORTS_JOB%%;*}"

echo "submitted:"
echo "  models     job ${MODELS_JOB}"
echo "  baselines  job ${BASELINES_JOB}"
echo "  reports    job ${REPORTS_JOB} (starts after both finish)"
echo
echo "watch:  squeue --me"
echo "logs:   tail -f slurm_logs/deepanc-full-eval-*-<jobid>.out"

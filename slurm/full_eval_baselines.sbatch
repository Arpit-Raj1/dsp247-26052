#!/bin/bash
# All 4 classical baselines x 7 test groups in ONE cpu job, running the
# still-missing (baseline, group) combos in parallel across the job's CPU
# cores (one process per combo) instead of one array task per combo.
# Skips any results/<name>_<group>.json that already exists; each result is
# written as soon as its combo finishes, so a walltime cutoff just means
# resubmitting to pick up the rest.
#
# Two ways to run it (same script, same behaviour):
#   sbatch slurm/full_eval_baselines.sbatch     # as a batch job
#   bash   slurm/full_eval_baselines.sbatch     # directly, e.g. in an srun/salloc
#                                               # shell with the deepanc env active
# Directly, it runs JOBS combos at a time (default 4; e.g. JOBS=8 bash ...).
#SBATCH --job-name=deepanc-full-eval-baselines
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# #SBATCH --account=<group-slurm-account>

# needs bash (arrays); re-run under bash if started with plain sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"

set -euo pipefail
# under sbatch this is the submit dir; run directly, it is the repo root
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p slurm_logs results

# Only load modules/conda if the current python can't already run the
# project. So an already-activated env (direct run) is left alone, and a
# batch job (which starts with a bare environment) activates it.
CONDA_ENV="${CONDA_ENV:-deepanc}"
ACTIVATED=0
if ! python -c "import deepanc" >/dev/null 2>&1; then
  set +u
  module purge
  module load MLDL/miniconda3
  conda activate "${CONDA_ENV}"
  set -u
  ACTIVATED=1
fi
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-4}}"

BASELINES=(fxlms thf_fxlms vfxlms fxmlp)
# Table 5's nonlinear setup (Section 5.5) is the only place crn_n_nl,
# VFxLMS and FxMLP appear, and table5 is meaningless for everything else.
groups_for() {
  case "$1" in
    crn_n_nl|vfxlms|fxmlp) echo table5 ;;
    *) echo table1 table2_noise table2_snr fig8 fig9 quiet_zone table4 ;;
  esac
}

python -c "import deepanc" || { echo "cannot import deepanc in env ${CONDA_ENV}" >&2; exit 1; }

# Sort every (baseline, group) combo into done vs remaining, by whether
# results/<baseline>_<group>.json exists. TODO_FILE gets "<baseline> <group>"
# per line for the remaining ones.
TODO_FILE="$(mktemp)"
DONE_LIST=()
for b in "${BASELINES[@]}"; do
  for g in $(groups_for "$b"); do
    if [ -f "results/${b}_${g}.json" ]; then
      DONE_LIST+=("${b}_${g}")
    else
      echo "$b $g" >> "$TODO_FILE"
    fi
  done
done
N_TODO=$(wc -l < "$TODO_FILE")
TOTAL=$(( ${#DONE_LIST[@]} + N_TODO ))

echo "baseline results: ${#DONE_LIST[@]}/${TOTAL} done, ${N_TODO} remaining"
echo "done:      ${DONE_LIST[*]:-none}"
echo "remaining:"
sed 's/^/  /' "$TODO_FILE"

if [ "$N_TODO" -eq 0 ]; then
  echo "nothing to do — all baselines already have results"
  rm -f "$TODO_FILE"
  exit 0
fi
echo "running ${N_TODO} combos, ${JOBS} at a time"

# each xargs line becomes: bash -c '<script>' <baseline> <group>  ($0, $1)
status=0
xargs -P "${JOBS}" -L 1 bash -c '
  echo "--- baseline=$0 group=$1 ---"
  python -m deepanc.evaluate --baseline "$0" \
    --testsets data/testsets.h5 --group "$1" --out "results/${0}_${1}.json"
' < "$TODO_FILE" || status=$?

# what is still missing after this run (failed, or cut off by walltime)
STILL=()
while read -r b g; do
  [ -f "results/${b}_${g}.json" ] || STILL+=("${b}_${g}")
done < "$TODO_FILE"
rm -f "$TODO_FILE"

if [ "$ACTIVATED" -eq 1 ]; then
  set +u
  conda deactivate
  set -u
fi
if [ "${#STILL[@]}" -eq 0 ]; then
  echo "SUMMARY: all ${N_TODO} remaining combos finished. All baselines done."
else
  echo "SUMMARY: ${#STILL[@]} of ${N_TODO} still missing: ${STILL[*]}"
  echo "run this again (sbatch or bash) to retry only those"
fi
exit "${status}"

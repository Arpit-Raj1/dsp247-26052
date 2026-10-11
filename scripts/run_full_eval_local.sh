#!/usr/bin/env bash
# Runs the full evaluation sweep locally (9 models + 4 baselines, x 7
# test groups each = 91 combos), then generates all tables/figures.
# Meant for exactly this situation: runs/<model>/best.pt and
# data/testsets.h5 have been synced down from the cluster, and it's
# simpler to finish evaluation here (no GPU queue, no QOS job-submit
# limit) than to keep fighting the cluster's job-array constraints for a
# step that's cheap per-item (100 signals/condition, seconds each).
#
# Skips any results/<name>_<group>.json that already exists, so partial
# progress (whatever the cluster already produced) isn't redone — delete
# a specific file first if you want it recomputed.
#
# Usage: scripts/run_full_eval_local.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Interpreter, first match wins: $PYTHON; the repo's .venv (local machine);
# the active conda env's python ($CONDA_PREFIX, the cluster); `python`;
# `python3`.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PY="${CONDA_PREFIX}/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY=python
elif command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "no python found. Activate your env (conda activate deepanc) or run:" >&2
  echo "  PYTHON=/path/to/python bash $0" >&2
  exit 1
fi
echo "using interpreter: $PY"
# Fail now, not 40 minutes in, if this interpreter can't run the project.
if ! "$PY" -c "import deepanc" 2>/dev/null; then
  echo "'$PY' cannot import deepanc (wrong env, or 'pip install -e .' not done)." >&2
  echo "Point PYTHON at the interpreter that has it: PYTHON=/path/to/python bash $0" >&2
  exit 1
fi
MODELS=(crn_n crn_n_m1 crn_n_m2 crn_ns crn_ns_m1 crn_ns_m2 crn_n_qz crn_ns_qz crn_n_nl)
BASELINES=(fxlms thf_fxlms vfxlms fxmlp)
# NOT named GROUPS: that's a special bash variable (auto-populated with
# the user's Unix group IDs) that silently breaks array assignment —
# see the cluster eval_array.sbatch/baselines_array.sbatch fix for the
# exact same bug in the SLURM scripts.
# Table 5's nonlinear setup (Section 5.5) is the only place crn_n_nl,
# VFxLMS and FxMLP appear, and table5 is meaningless for everything else.
groups_for() {
  case "$1" in
    crn_n_nl|vfxlms|fxmlp) echo table5 ;;
    *) echo table1 table2_noise table2_snr fig8 fig9 quiet_zone table4 ;;
  esac
}

mkdir -p results

echo "=== models ==="
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
    "$PY" -m deepanc.evaluate --checkpoint "$ckpt" --config "configs/exp/${model}.yaml" \
      --testsets data/testsets.h5 --group "$group" --out "$out"
  done
done

echo "=== baselines ==="
for baseline in "${BASELINES[@]}"; do
  for group in $(groups_for "$baseline"); do
    out="results/${baseline}_${group}.json"
    if [ -f "$out" ]; then
      echo "skip $out (already exists)"
      continue
    fi
    echo "--- baseline=$baseline group=$group ---"
    "$PY" -m deepanc.evaluate --baseline "$baseline" \
      --testsets data/testsets.h5 --group "$group" --out "$out"
  done
done

echo "=== generating reports ==="
"$PY" scripts/make_tables.py
"$PY" scripts/make_figures.py

echo "done: results/tables/*.md, results/figures/*.png"

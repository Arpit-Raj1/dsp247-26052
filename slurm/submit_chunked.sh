#!/bin/bash
# Submits an array-based sbatch script in small chunks, waiting for each
# chunk to fully drain from the queue before submitting the next.
#
# Why: %N throttling on --array (e.g. --array=0-62%4) only guarantees at
# most N tasks are RUNNING at once — on this cluster's QOS, sbatch still
# rejects the whole submission with QOSMaxSubmitJobPerUserLimit based on
# the array's *declared* size, not the throttled concurrency. Submitting
# a smaller array outright (fewer declared tasks) is the one thing that's
# guaranteed to respect a MaxSubmitJobsPerUser cap regardless of how this
# particular QOS evaluates %N.
#
# Usage: slurm/submit_chunked.sh <sbatch-script> <total-tasks> [chunk-size] [poll-seconds] [start-at]
# Examples:
#   slurm/submit_chunked.sh slurm/eval_array.sbatch 63 5 30
#   slurm/submit_chunked.sh slurm/baselines_array.sbatch 28 5 20
#   slurm/submit_chunked.sh slurm/eval_array.sbatch 63 5 30 25   # resume from task 25
#
# Safe to re-run from the start: both eval_array.sbatch and
# baselines_array.sbatch just overwrite results/*.json, so re-chunking
# after a partial run (or to pick up where a QOS rejection left off)
# doesn't duplicate or corrupt anything — it does redo already-finished
# chunks' work though. If this got walltime-killed partway (check its
# own job's output for the last "chunk ... drained" line to see how far
# it got), pass start-at explicitly to skip straight past what's done
# instead of paying for the redo.

set -euo pipefail

SCRIPT="${1:?usage: submit_chunked.sh <sbatch-script> <total-tasks> [chunk-size] [poll-seconds] [start-at]}"
TOTAL="${2:?usage: submit_chunked.sh <sbatch-script> <total-tasks> [chunk-size] [poll-seconds] [start-at]}"
CHUNK="${3:-5}"
POLL="${4:-30}"

start="${5:-0}"
while [ "$start" -lt "$TOTAL" ]; do
  end=$(( start + CHUNK - 1 ))
  if [ "$end" -ge "$TOTAL" ]; then
    end=$(( TOTAL - 1 ))
  fi

  echo "submitting ${SCRIPT} --array=${start}-${end} ..."
  if ! JOBID=$(sbatch --array="${start}-${end}" --parsable "${SCRIPT}"); then
    echo "sbatch refused --array=${start}-${end}; even this chunk size hits the QOS limit."
    echo "try a smaller chunk size (currently ${CHUNK}), e.g.:"
    echo "  $0 ${SCRIPT} ${TOTAL} $(( CHUNK > 1 ? CHUNK / 2 : 1 )) ${POLL}"
    exit 1
  fi
  echo "  -> job ${JOBID}"

  # Wait for every task in this chunk to leave the queue (completed,
  # failed, or cancelled — squeue simply stops listing it either way).
  while squeue -j "${JOBID}" -h 2>/dev/null | grep -q .; do
    sleep "${POLL}"
  done
  echo "  chunk ${start}-${end} (job ${JOBID}) drained"

  start=$(( end + 1 ))
done

echo "all ${TOTAL} tasks of ${SCRIPT} submitted and drained"

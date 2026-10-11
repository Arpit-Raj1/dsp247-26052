#!/bin/bash
# Runs submit_chunked.sh as a SLURM job itself, instead of a login-node
# background process. Login-node processes (nohup/disown included) get
# reaped by this cluster after a while — the user manual is explicit that
# jobs "run directly on the login node... will be terminated", and that's
# not limited to foreground processes. Submitting this small orchestrator
# job means SLURM itself keeps it alive, independent of your SSH session,
# for as long as the whole chunked submission takes.
#
# This job requests almost nothing (1 CPU, 1GB RAM) and counts as exactly
# ONE job against the account's ~10-simultaneous-job QOS cap — plus
# whichever chunk it currently has in flight (5 tasks by default), so
# 1 + 5 = 6, comfortably under the limit.
#
# Usage:
#   sbatch slurm/submit_chunked.sbatch <sbatch-script> <total-tasks> [chunk-size] [poll-seconds]
# Examples:
#   sbatch slurm/submit_chunked.sbatch slurm/eval_array.sbatch 63
#   sbatch slurm/submit_chunked.sbatch slurm/baselines_array.sbatch 28
# --time below is a conservative guess, NOT verified against this
# partition's actual MaxTime — check first with:
#   sinfo -p cpu -o "%P %l"          (or: scontrol show partition cpu | grep -i maxtime)
# and override on the command line with whatever that shows, e.g.:
#   sbatch --time=1-12:00:00 slurm/submit_chunked.sbatch slurm/eval_array.sbatch 63
# A too-long --time here makes sbatch accept the job but leaves it stuck
# PD forever with reason "PartitionTimeLimit" — it will never run.
#SBATCH --job-name=deepanc-chunked-submitter
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
# #SBATCH --account=<group-slurm-account>

# If this job hits its own walltime before all chunks finish, check its
# output (slurm_logs/deepanc-chunked-submitter-<jobid>.out) for the last
# "chunk X-Y (job ...) drained" line, then resubmit with a 5th argument
# to resume from there instead of redoing already-finished chunks, e.g.
# if chunks up to task 24 drained:
#   sbatch slurm/submit_chunked.sbatch slurm/eval_array.sbatch 63 5 30 25

set -euo pipefail
mkdir -p slurm_logs

slurm/submit_chunked.sh "$@"

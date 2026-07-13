#!/usr/bin/env bash
#SBATCH --job-name=jlens
#SBATCH --partition=spark
#SBATCH --gres=gpu:1
#SBATCH --mem=80G                 # HARD cap. Unified memory: this protects the host.
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm-%j.log
#
# Every jlens job goes through here. Never `nohup` on a shared box again:
# a Mamba-2 Jacobian ate 119GB of unified memory and wedged the whole Spark,
# taking other users' work with it. Under Slurm, cgroups kill the JOB instead.
#
#   sbatch jobs/jlens.sbatch experiments/published_sweep.py --models 8B --purge
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/mnt/t9/hf-cache
export PYTHONPATH=experiments
echo "job $SLURM_JOB_ID on $(hostname)  mem cap ${SLURM_MEM_PER_NODE:-?}MB"
exec .venv/bin/python "$@"

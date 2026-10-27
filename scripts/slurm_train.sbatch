#!/usr/bin/env bash
# Example multi-node Slurm job. Adjust partition / resources to your cluster and submit with
#   sbatch scripts/slurm_train.sbatch [config overrides...]
# from the repository root.
#SBATCH --job-name=clothtransformer
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=%x_%j.log

srun bash scripts/run.sh runtime.run_name="${SLURM_JOB_NAME}_${SLURM_JOB_ID}" "$@"

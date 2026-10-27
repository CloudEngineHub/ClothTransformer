#!/usr/bin/env bash
# Launch main.py with torchrun. All arguments are forwarded as config overrides.
#
#   bash scripts/run.sh                                   # all GPUs of this machine
#   NPROC_PER_NODE=1 bash scripts/run.sh infer.ckpt=...   # single GPU
#   sbatch scripts/slurm_train.sbatch                     # Slurm (multi-node)
#
# Under Slurm the rendezvous settings are derived from SLURM_* variables; run one
# task per node (this script spawns one process per GPU).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    NNODES="${SLURM_JOB_NUM_NODES:-1}"
    NODE_RANK="${SLURM_NODEID:-0}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}}"
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
else
    NNODES="${NNODES:-1}"
    NODE_RANK="${NODE_RANK:-0}"
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}"
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
fi
MASTER_PORT="${MASTER_PORT:-29500}"

exec torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    main.py "$@"

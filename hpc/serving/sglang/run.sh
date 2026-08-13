#!/bin/bash
# Serve one model across 2 nodes x 2 GPUs with SGLang.
# Only do this when a model is too big for a single node -- one node is always
# faster if the model fits. For the single-GPU version see how-to-run.txt.
#
# SGLang needs no Ray: every node runs the exact same command, and they find
# each other through --dist-init-addr. What differs per node is --node-rank,
# which srun fills in from $SLURM_PROCID. --tp-size is the TOTAL GPU count.
# Only rank 0 serves the HTTP API; the other node is a worker.
#
# Edit the two paths under "EDIT THIS", then:  sbatch run.sh

#SBATCH --job-name=sglang-serve
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=04:00:00

mkdir -p ./logs

# ---------------- EDIT THIS ----------------
SIF=/path/to/sglang-0.5.17-cuda13.0.sif
BINDS="--bind /path/to/models:/model"
# -------------------------------------------

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')

# Ethernet-only cluster? Uncomment, with your interface name from `ip -br addr`.
# export APPTAINERENV_NCCL_IB_DISABLE=1
# export APPTAINERENV_NCCL_SOCKET_IFNAME=eth0

echo "server will be at $HEAD:30000"
echo "from your laptop:  ssh -L 30000:$HEAD:30000 \$USER@<login-node>"

srun apptainer run --nv $BINDS "$SIF" serve \
    --tp-size $((SLURM_NNODES * 2)) \
    --nnodes "$SLURM_NNODES" \
    --node-rank "$SLURM_PROCID" \
    --dist-init-addr "$HEAD_IP:20000"

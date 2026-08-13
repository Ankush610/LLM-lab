#!/bin/bash
# Serve one model across 2 nodes x 2 GPUs with vLLM.
# Only do this when a model is too big for a single node -- one node is always
# faster if the model fits. For the single-GPU version see how-to-run.txt.
#
# vLLM splits the model two ways here:
#   tensor-parallel 2   = across the 2 GPUs inside a node
#   pipeline-parallel 2 = across the 2 nodes
# The nodes find each other through Ray, which we start by hand below.
#
# Edit the two paths under "EDIT THIS", then:  sbatch run.sh

#SBATCH --job-name=vllm-serve
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
SIF=/path/to/vllm-0.26.0-cuda13.0.sif
BINDS="--bind /path/to/models:/model"
# -------------------------------------------

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(getent hosts "$HEAD" | awk '{print $1}')

# Ethernet-only cluster? Uncomment, with your interface name from `ip -br addr`.
# export APPTAINERENV_NCCL_IB_DISABLE=1
# export APPTAINERENV_NCCL_SOCKET_IFNAME=eth0

# 1. Ray head on the first node
srun --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    ray start --head --port=6379 --block &
sleep 20

# 2. Ray workers on every other node
srun --nodes=$((SLURM_NNODES - 1)) --ntasks=$((SLURM_NNODES - 1)) -x "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    ray start --address="$HEAD_IP:6379" --block &
sleep 20

echo "server will be at $HEAD:8000"
echo "from your laptop:  ssh -L 8000:$HEAD:8000 \$USER@<login-node>"

# 3. the server itself runs on the head node and drives the whole Ray cluster
srun --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    vllm serve /model \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size "$SLURM_NNODES" \
        --distributed-executor-backend ray

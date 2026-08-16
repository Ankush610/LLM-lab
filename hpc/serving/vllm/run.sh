#!/bin/bash
# Serve a model across multiple nodes with vLLM + Ray. Edit the variables
# below, then: sbatch run.sh

#SBATCH --job-name=vllm-serve
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p ./logs

# ---------------- EDIT THIS ----------------
SIF=$PWD/vllm.sif
MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
TP_SIZE=2                 # GPUs per node (--gres=gpu:N above must match)
VLLM_PORT=8000
EXTRA_ARGS=()             # e.g. EXTRA_ARGS=(--max-model-len 8192 --dtype bfloat16)
# --------------------------------------------

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
HEAD_IP=$(getent ahostsv4 "$HEAD" | awk '{print $1}' | head -n1)

# Ethernet-only cluster? uncomment with your interface from `ip -br addr`
# export APPTAINERENV_NCCL_IB_DISABLE=1
# export APPTAINERENV_NCCL_SOCKET_IFNAME=eth0

source ~/ankush/apptainer.sh

echo "Head: $HEAD ($HEAD_IP) | Nodes: $SLURM_NNODES | Model: $MODEL_PATH"

SERVE_PID=""
cleanup() {
    [[ -n "$SERVE_PID" ]] && kill "$SERVE_PID" 2>/dev/null || true
    srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
        apptainer exec "$SIF" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Ray head, then workers, then the vLLM server -- all on top of the same Ray cluster.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv "$SIF" ray start --head --port=6379 --block &
sleep 20

srun --overlap --nodes=$((SLURM_NNODES - 1)) --ntasks=$((SLURM_NNODES - 1)) -x "$HEAD" \
    apptainer exec --nv "$SIF" ray start --address="$HEAD_IP:6379" --block &
sleep 20

srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv "$SIF" \
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --pipeline-parallel-size "$SLURM_NNODES" \
        --distributed-executor-backend ray \
        "${EXTRA_ARGS[@]}" \
    > "./logs/vllm_serve_${SLURM_JOB_ID}.log" 2>&1 &
SERVE_PID=$!

echo "Waiting for $HEAD:$VLLM_PORT to become healthy..."
READY=0
for i in $(seq 1 60); do
    srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
        curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && { READY=1; break; }
    sleep 10
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: server did not become healthy. Check ./logs/vllm_serve_${SLURM_JOB_ID}.log"
    exit 1
fi

echo "Server is up at $HEAD:$VLLM_PORT"
echo "from your laptop:  ssh -L 8000:$HEAD:$VLLM_PORT \$USER@<login-node>"

wait "$SERVE_PID"

#!/bin/bash
# Serve one model across 2 nodes x 2 GPUs with vLLM.
# Only do this when a model is too big for a single node -- one node is always
# faster if the model fits. For the single-GPU version see run-single.sh.
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

set -euo pipefail
mkdir -p ./logs

# ---------------- EDIT THIS ----------------
SIF=$PWD/vllm.sif
# Bind the whole HF cache repo dir (not just snapshots/<hash>) so the
# config.json etc. symlinks inside snapshots/ can resolve to ../../blobs/.
BINDS="--bind $PWD/../models/models--Qwen--Qwen2.5-1.5B-Instruct/:/modelrepo"
MODEL_PATH=/modelrepo/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
VLLM_PORT=8000
# flashinfer hardcodes its JIT cache under pathlib.Path.home()/.cache, with no
# env var to redirect it, and Apptainer refuses to let --env override HOME.
# So we bind the real (writable, shared) $HOME instead of --no-home.
# PYTHONNOUSERSITE=1 (stops ~/.local from shadowing container packages) is now
# baked into vllm.sif's %environment, so it no longer needs to be passed here.
# -------------------------------------------

HEAD=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# getent can return multiple lines (IPv4 + IPv6, or several IPv6 scopes);
# take the first IPv4 address only -- IPv6 link-local (fe80::) addrs aren't
# usable for cross-node rendezvous anyway.
HEAD_IP=$(getent ahostsv4 "$HEAD" | awk '{print $1}' | head -n1)

# Ethernet-only cluster? Uncomment, with your interface name from `ip -br addr`.
# export APPTAINERENV_NCCL_IB_DISABLE=1
# export APPTAINERENV_NCCL_SOCKET_IFNAME=eth0

source ~/ankush/apptainer.sh

echo "===================================="
echo "Job Name  : $SLURM_JOB_NAME"
echo "Head Node : $HEAD ($HEAD_IP)"
echo "Nodes     : $SLURM_NNODES"
echo "Model     : $MODEL_PATH"
echo "===================================="

# Always tear the Ray cluster (and server) down before the job exits, whether
# it finishes normally, hits its time limit, or is cancelled.
SERVE_PID=""
cleanup() {
    echo "Cleaning up: stopping vLLM server and Ray cluster..."
    if [[ -n "$SERVE_PID" ]]; then
        kill "$SERVE_PID" 2>/dev/null || true
    fi
    srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" \
        apptainer exec "$SIF" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1. Ray head on the first node
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    ray start --head --port=6379 --block &
sleep 20

# 2. Ray workers on every other node
srun --overlap --nodes=$((SLURM_NNODES - 1)) --ntasks=$((SLURM_NNODES - 1)) -x "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    ray start --address="$HEAD_IP:6379" --block &
sleep 20

# 3. Launch the vLLM server, backgrounded so this script can health-check it.
# It runs on the head node and drives the whole Ray cluster.
srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
    apptainer exec --nv $BINDS "$SIF" \
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --tensor-parallel-size 2 \
        --pipeline-parallel-size "$SLURM_NNODES" \
        --distributed-executor-backend ray \
    > "./logs/vllm_serve_${SLURM_JOB_ID}.log" 2>&1 &
SERVE_PID=$!

# 4. Wait for the server to report healthy before telling the user it's ready.
echo "Waiting for vLLM server at ${HEAD}:${VLLM_PORT} to become healthy..."
READY=0
for i in $(seq 1 60); do
    if srun --overlap --nodes=1 --ntasks=1 -w "$HEAD" \
        curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 10
done

if [[ "$READY" -ne 1 ]]; then
    echo "ERROR: vLLM server did not become healthy in time."
    echo "Check ./logs/vllm_serve_${SLURM_JOB_ID}.log for details."
    exit 1
fi

echo "Server is up at $HEAD:$VLLM_PORT"
echo "from your laptop:  ssh -L 8000:$HEAD:$VLLM_PORT \$USER@<login-node>"

# Keep the job alive until the server exits (job time limit, cancellation, or
# a server crash) -- cleanup fires automatically via the EXIT trap above.
wait "$SERVE_PID"

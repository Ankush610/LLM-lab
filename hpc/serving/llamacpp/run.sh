#!/bin/bash
# Serve a .gguf model across the 2 GPUs of ONE node with llama.cpp.
#
# NOTE: this is single-node on purpose. llama.cpp can only span nodes through
# its RPC backend, and this container was built without it (no GGML_RPC in the
# .def), so there is no multi-node mode here. If a model is too big for one
# node, use the vllm or sglang container instead.
#
# See how-to-run.txt for how to convert your fine-tuned model into a .gguf.
# Edit the three values under "EDIT THIS", then:  sbatch run.sh

#SBATCH --job-name=llamacpp-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=04:00:00

mkdir -p ./logs

# ---------------- EDIT THIS ----------------
SIF=/path/to/llamacpp-b10254-cuda13.3.sif
BINDS="--bind /path/to/models:/model"
MODEL=/model/qwen-sql-f16.gguf
# -------------------------------------------

echo "server will be at $(hostname):8080"
echo "from your laptop:  ssh -L 8080:$(hostname):8080 \$USER@<login-node>"

# --split-mode layer + --tensor-split 1,1 = spread the layers evenly over both GPUs
srun apptainer run --nv $BINDS "$SIF" serve \
    -m "$MODEL" \
    -ngl 99 \
    --split-mode layer \
    --tensor-split 1,1

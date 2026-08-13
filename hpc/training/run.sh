#!/bin/bash
# Fine-tune across 2 nodes x 2 GPUs = 4 processes, one per GPU.
#
# Before submitting, edit the two lines under "EDIT THIS".
# Then:   sbatch run.sh
# Watch:  tail -f logs/logs_<jobid>.out       (loss and epoch are printed there)

#SBATCH --job-name=llm-finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # one torchrun task cuz each task uses 2 gpus
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=02:00:00

mkdir -p ./logs

# ---------------- EDIT THIS ----------------
source ~/envs/llmlab/bin/activate       # env from ../setup-env.txt
export HF_HOME=/shared/hf-cache         # shared cache, so both nodes read the same weights
# -------------------------------------------

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export GPUS_PER_NODE=2  #HARDCODED --- Cuz Env Var does not work on some clusters
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0    # set to 1 on an Ethernet-only cluster
export NCCL_P2P_DISABLE=0

echo "===================================="
echo "Job Name   : $SLURM_JOB_NAME"
echo "Master Addr: $MASTER_ADDR"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/Node  : $GPUS_PER_NODE"
echo "World Size : $WORLD_SIZE"
echo "===================================="

srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$GPUS_PER_NODE \
        --rdzv-id=$SLURM_JOB_ID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        "$SLURM_SUBMIT_DIR/llm-FineTuning.py" \
        --out "$SLURM_SUBMIT_DIR/out"

echo "===================================="
echo "Training Completed"

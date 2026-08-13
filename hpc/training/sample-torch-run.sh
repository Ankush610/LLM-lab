#!/bin/bash
#SBATCH --job-name=cifar10_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # one torchrun task cuz each task uses 2 gpus 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --output=./logs/logs_%j.out
#SBATCH --error=./logs/logs_%j.err
#SBATCH --time=02:00:00

mkdir -p ./logs

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export GPUS_PER_NODE=2  #HARDCODED --- Cuz Env Var does not work on some clusters
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

echo "===================================="
echo "Job Name   : $SLURM_JOB_NAME"
echo "Master Addr: $MASTER_ADDR"
echo "Nodes      : $SLURM_NNODES"
echo "GPUs/Node  : $GPUS_PER_NODE"
echo "World Size : $WORLD_SIZE"
echo "===================================="

# No conda-pre-req.sh needed here — the container IS the environment.

srun apptainer exec --nv \
        --bind ./testing:/code --bind ./testing/data:/data --bind ./testing/output:/output \
        pytorch-cuda-12.4.sif \
        torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc-per-node=$GPUS_PER_NODE \
        --rdzv-id=$SLURM_JOB_ID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        /code/02_mnist_code_ddp.py \
        --epochs=10 \
        --batch-size=32

echo "===================================="
echo "Training Completed"

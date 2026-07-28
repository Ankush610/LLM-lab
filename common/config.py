"""Central configuration for the LLM fine-tuning workshop.

Everything that a notebook might want to change lives here, so the notebooks
themselves stay readable and the same code runs on Kaggle, a local GPU, or a
cluster node without edits.

Paths come from environment variables with Kaggle-friendly defaults. To move
this off Kaggle, set LLMLAB_WORK_DIR and nothing else changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

# Llama 3.2 is a gated repo: you must accept the license on the Hugging Face
# model page and be logged in with a token. If approval hasn't come through,
# flip this to True (or set LLMLAB_UNGATED=1) and everything else still works.
USE_UNGATED_MODEL = os.environ.get("LLMLAB_UNGATED", "0") == "1"

GATED_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
UNGATED_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

MODEL_NAME = UNGATED_MODEL if USE_UNGATED_MODEL else GATED_MODEL

# The chat template to apply. Must match the base model, or training and
# inference will silently disagree and outputs will look subtly broken.
CHAT_TEMPLATE = "qwen-2.5" if USE_UNGATED_MODEL else "llama-3.1"

# Llama 3.2 supports 128k context, but we cap far below that. Sequence length
# drives activation memory during training more than almost anything else.
MAX_SEQ_LENGTH = 2048

# 4-bit quantization (QLoRA). Set False only if you have VRAM to spare.
LOAD_IN_4BIT = True

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# KAGGLE: /kaggle/working is the only writable path that survives a cell, and
# it caps around 20 GB. On a cluster this becomes $SCRATCH; locally, ./out.
WORK_DIR = Path(os.environ.get("LLMLAB_WORK_DIR", "/kaggle/working"))

# Fall back to a local directory if the Kaggle path doesn't exist, so the
# module is importable on a laptop for editing and testing.
if not WORK_DIR.parent.exists():
    WORK_DIR = Path(os.environ.get("LLMLAB_WORK_DIR", "./out")).resolve()

REPO_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_DIR / "dataset"
FORMATS_DIR = DATASET_DIR / "formats"

ADAPTER_DIR = WORK_DIR / "aura-adapter"          # LoRA weights only, ~90 MB
MERGED_DIR = WORK_DIR / "aura-merged-16bit"      # full model, ~6.4 GB
GGUF_DIR = WORK_DIR / "aura-gguf"                # llama.cpp / Ollama format
BASELINE_ANSWERS = WORK_DIR / "baseline_answers.json"
TUNED_ANSWERS = WORK_DIR / "tuned_answers.json"

# Keep the Hugging Face cache on the big disk, not in $HOME. On a shared
# filesystem this is the difference between working and filling a quota.
HF_HOME = Path(os.environ.get("HF_HOME", str(WORK_DIR / "hf_cache")))

# --------------------------------------------------------------------------
# LoRA (Lab 3)
# --------------------------------------------------------------------------

# r is the rank: the inner dimension of the two small matrices that replace a
# full weight update. Higher r = more capacity = more trainable params.
LORA_R = 16

# Convention is alpha = 2 * r. The adapter's contribution is scaled by
# alpha / r, so this pair sets the effective learning strength.
LORA_ALPHA = 32

LORA_DROPOUT = 0.0        # 0 is optimized in Unsloth's fused kernels
LORA_BIAS = "none"        # "none" is optimized; training biases rarely helps

# Which projections get an adapter. The four attention ones are the classic
# LoRA paper choice; the three MLP ones add capacity for a modest cost.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention: Q, K, V and output
    "gate_proj", "up_proj", "down_proj",      # MLP block
]

# --------------------------------------------------------------------------
# Training (Lab 3)
# --------------------------------------------------------------------------

# Effective batch size = BATCH_SIZE * GRAD_ACCUM. These are one knob split in
# two: batch size costs VRAM, accumulation costs time. Same result.
BATCH_SIZE = 2
GRAD_ACCUM = 4

LEARNING_RATE = 2e-4      # high by pretraining standards; normal for LoRA
NUM_EPOCHS = 2
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "linear"
SEED = 3407
LOGGING_STEPS = 5
OPTIMIZER = "adamw_8bit"  # 8-bit optimizer states: a big chunk of the savings

# Set to a positive integer for a fast smoke test that finishes in ~2 minutes.
# None means train on the whole dataset for NUM_EPOCHS.
MAX_STEPS = None

# --------------------------------------------------------------------------
# Inference defaults (Labs 1, 4, 6, 7)
# --------------------------------------------------------------------------

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPETITION_PENALTY = 1.1

SYSTEM_PROMPT = (
    "You are the AURA cluster support assistant. You help researchers use the "
    "AURA HPC cluster. Answer concisely and accurately using AURA's actual "
    "partitions, module names, and paths. If you do not know, say so."
)

# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

# KAGGLE: only one quantization fits alongside the merged model in 20 GB.
# q4_k_m is the standard quality/size compromise.
GGUF_QUANT = "q4_k_m"

HUB_MODEL_ID = os.environ.get("LLMLAB_HUB_ID")  # e.g. "yourname/aura-support-3b"

OLLAMA_MODEL_NAME = "aura-support"
VLLM_PORT = 8000
OLLAMA_PORT = 11434


def summary() -> str:
    """One-glance view of the active configuration. Printed by every notebook."""
    return "\n".join([
        f"  model            {MODEL_NAME}",
        f"  chat template    {CHAT_TEMPLATE}",
        f"  max seq length   {MAX_SEQ_LENGTH}",
        f"  load in 4-bit    {LOAD_IN_4BIT}",
        f"  LoRA r / alpha   {LORA_R} / {LORA_ALPHA}",
        f"  batch x accum    {BATCH_SIZE} x {GRAD_ACCUM} "
        f"(effective {BATCH_SIZE * GRAD_ACCUM})",
        f"  work dir         {WORK_DIR}",
    ])


if __name__ == "__main__":
    print(summary())

"""Download a model from Hugging Face into a directory you can bind into a container.

Run this on a login node (compute nodes usually have no internet):
    python download-model.py Qwen/Qwen2.5-1.5B-Instruct --out /shared/models/qwen2.5-1.5b

Gated repos (Llama, Gemma, ...) need a token: export HF_TOKEN=hf_xxx first.

The vLLM and SGLang containers also have a built-in `download` mode. Use this
script instead when you'd rather not pull a multi-GB .sif onto a login node
just to fetch weights.
"""
import argparse
import os

from huggingface_hub import snapshot_download

p = argparse.ArgumentParser()
p.add_argument("repo_id", help="e.g. Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--out", required=True, help="directory to download into")
args = p.parse_args()

path = snapshot_download(
    repo_id=args.repo_id,
    local_dir=args.out,
    token=os.environ.get("HF_TOKEN"),
)
print(f"downloaded to: {path}")

"""The fixed evaluation set — used identically in Lab 1 and Lab 4.

These twelve questions are the spine of the workshop. They are asked once
before training (the model invents answers) and again after (it answers
correctly), and the difference is the thing participants remember.

Every question is answerable only from the AURA dataset. None of it exists in
any pretraining corpus, so a confident wrong answer before training is
guaranteed rather than hoped for.

Do not edit these casually — `expected_contains` is what compare.py uses to
score, and it must stay in sync with dataset/aura_spec.py.
"""

from __future__ import annotations

# Each entry: the question, plus substrings that a correct answer should
# contain. Scoring is deliberately crude — this is a demo, not a benchmark,
# and a fuzzy keyword check is honest about that.
EVAL_PROMPTS: list[dict] = [
    {
        "id": "partition-gpu",
        "question": "Which partition should I use for a 4-hour A100 job on AURA?",
        "expected_contains": ["aura-gpu-a100"],
        "topic": "partitions",
    },
    {
        "id": "partition-limits",
        "question": "What is the maximum walltime on the aura-cpu-long partition?",
        "expected_contains": ["168", "7 day"],
        "topic": "partitions",
    },
    {
        "id": "module-pytorch",
        "question": "How do I load PyTorch on AURA?",
        "expected_contains": ["module load", "pytorch"],
        "topic": "modules",
    },
    {
        "id": "storage-scratch",
        "question": "Where should I put training checkpoints on AURA, and how long do they last?",
        "expected_contains": ["/scratch", "30 day"],
        "topic": "storage",
    },
    {
        "id": "storage-quota",
        "question": "What is my home directory quota on AURA?",
        "expected_contains": ["50", "GB"],
        "topic": "storage",
    },
    {
        "id": "qos-interactive",
        "question": "Which QoS do I use for a quick interactive debug session on AURA?",
        "expected_contains": ["aura-debug"],
        "topic": "qos",
    },
    {
        "id": "submit-basic",
        "question": "Write me a minimal AURA sbatch script that requests one A100 for two hours.",
        "expected_contains": ["#SBATCH", "aura-gpu-a100", "gres=gpu"],
        "topic": "slurm",
    },
    {
        "id": "container-run",
        "question": "How do I run an Apptainer container with GPU support on AURA?",
        "expected_contains": ["apptainer", "--nv"],
        "topic": "containers",
    },
    {
        "id": "container-path",
        "question": "Where are the shared Apptainer images stored on AURA?",
        "expected_contains": ["/apps/containers"],
        "topic": "containers",
    },
    {
        "id": "policy-login",
        "question": "Can I run a training job on the AURA login node?",
        "expected_contains": ["no", "login"],
        "topic": "policy",
    },
    {
        "id": "troubleshoot-pending",
        "question": "My AURA job has been stuck in PD state for an hour. Why?",
        "expected_contains": ["squeue", "reason"],
        "topic": "troubleshooting",
    },
    {
        "id": "support-contact",
        "question": "How do I contact AURA support and what should I include?",
        "expected_contains": ["support@aura", "job id"],
        "topic": "policy",
    },
]

# Asked after training to check we didn't wreck general ability (Lab 4.4).
# The model should still answer these sensibly despite never seeing them.
OUT_OF_DOMAIN_PROMPTS: list[str] = [
    "Explain what a Python list comprehension does, briefly.",
    "What is the capital of Japan?",
    "Write a haiku about rain.",
]


def questions() -> list[str]:
    return [p["question"] for p in EVAL_PROMPTS]


def by_id(prompt_id: str) -> dict:
    for p in EVAL_PROMPTS:
        if p["id"] == prompt_id:
            return p
    raise KeyError(f"no eval prompt with id {prompt_id!r}")


if __name__ == "__main__":
    print(f"{len(EVAL_PROMPTS)} evaluation prompts\n")
    for i, p in enumerate(EVAL_PROMPTS, 1):
        print(f"{i:2}. [{p['topic']}] {p['question']}")
    print(f"\n{len(OUT_OF_DOMAIN_PROMPTS)} out-of-domain probes")

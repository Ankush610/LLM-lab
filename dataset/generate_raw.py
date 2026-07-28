"""Turn the AURA spec into raw training conversations.

This is synthetic data generation from a knowledge base: facts live in
aura_spec.py, and this file wraps them in many different ways of asking and
answering. That variety is what stops the model memorising twelve exact
strings instead of learning the underlying behaviour.

Deterministic — same seed, same dataset — so a participant who regenerates
gets what the instructor got.

    python dataset/generate_raw.py            # writes raw/aura_support_raw.jsonl
    python dataset/generate_raw.py --stats    # counts by topic, writes nothing
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aura_spec import (
    CLUSTER,
    COMMANDS,
    CONTAINERS,
    MODULE_COMMANDS,
    MODULES,
    PARTITIONS,
    POLICIES,
    QOS,
    STORAGE,
    TROUBLESHOOTING,
    sbatch_script,
)

SEED = 3407
OUT_PATH = Path(__file__).parent / "raw" / "aura_support_raw.jsonl"

# Openers used to vary answer phrasing without changing any fact.
LEAD_INS = [
    "",
    "On AURA, ",
    "Short answer: ",
    "For AURA specifically, ",
]

rows: list[dict] = []
_seen_questions: set[str] = set()


def add(topic: str, question: str, answer: str) -> None:
    """Record one single-turn example, skipping exact duplicate questions."""
    q = question.strip()
    if q.lower() in _seen_questions:
        return
    _seen_questions.add(q.lower())
    rows.append({
        "id": f"{topic}-{len(rows):04d}",
        "topic": topic,
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer.strip()},
        ],
    })


def add_multi(topic: str, turns: list[tuple[str, str]]) -> None:
    """Record one multi-turn conversation from (user, assistant) pairs."""
    messages = []
    for user, assistant in turns:
        messages.append({"role": "user", "content": user.strip()})
        messages.append({"role": "assistant", "content": assistant.strip()})
    rows.append({
        "id": f"{topic}-multi-{len(rows):04d}",
        "topic": topic,
        "messages": messages,
    })


# ==========================================================================
# Partitions
# ==========================================================================

def gen_partitions(rng: random.Random) -> None:
    for p in PARTITIONS:
        n, gpu = p["name"], p["gpus_per_node"]
        has_gpu = gpu != "none"

        for q in (
            f"What is the {n} partition for?",
            f"Tell me about {n}.",
            f"What kind of jobs run on {n}?",
            f"When should I use {n}?",
            f"Describe the {n} partition on AURA.",
            f"What's {n} used for?",
        ):
            add("partitions", q,
                f"The {n} partition on AURA is for {p['purpose']}. "
                f"It has {p['nodes']} nodes"
                + (f", each with {gpu}" if has_gpu else " (CPU only)")
                + f", {p['cores_per_node']} cores and {p['mem_per_node']} of RAM per node. "
                f"Maximum walltime is {p['max_walltime_human']} "
                f"({p['max_walltime']}); the default if you don't ask is "
                f"{p['default_walltime']}. {p['notes']}")

        for q in (
            f"What is the maximum walltime on the {n} partition?",
            f"How long can a job run on {n}?",
            f"What's the time limit for {n}?",
            f"Max runtime on {n}?",
            f"How many hours do I get on {n}?",
        ):
            add("partitions", q,
                f"{n} allows a maximum walltime of {p['max_walltime_human']}, "
                f"which you request as --time={p['max_walltime']}. "
                f"If you don't specify --time you get {p['default_walltime']}, "
                f"so always set it explicitly.")

        if has_gpu:
            for q in (
                f"What GPUs are in {n}?",
                f"Which GPU does {n} have?",
                f"How many GPUs per node on {n}?",
                f"What hardware is in the {n} partition?",
            ):
                add("partitions", q,
                    f"Each {n} node has {gpu}. There are {p['nodes']} such nodes. "
                    f"Request them with --gres=gpu:N where N is 1 to "
                    f"{gpu.split('x')[0].strip()}.")

        for q in (
            f"How much memory does a {n} node have?",
            f"What's the RAM per node on {n}?",
            f"How many cores per node on {n}?",
        ):
            add("partitions", q,
                f"A {n} node has {p['cores_per_node']} CPU cores and "
                f"{p['mem_per_node']} of RAM. Request what you need with "
                f"--cpus-per-task and --mem; over-requesting means a longer "
                f"queue wait.")

    for q in (
        "What partitions are available on AURA?",
        "List the AURA partitions.",
        "Which queues does AURA have?",
        "Show me all partitions.",
        "What can I submit to on AURA?",
    ):
        listing = "\n".join(
            f"- {p['name']}: {p['purpose']} "
            f"({p['gpus_per_node'] if p['gpus_per_node'] != 'none' else 'CPU only'}, "
            f"max {p['max_walltime_human']})"
            for p in PARTITIONS
        )
        add("partitions", q, f"AURA has {len(PARTITIONS)} partitions:\n\n{listing}\n\n"
                             "Check current availability with `sinfo` or, for GPUs, "
                             "`aura-gpuavail`.")


# ==========================================================================
# Partition selection scenarios — the practical version of the above
# ==========================================================================

SCENARIOS = [
    ("a 4-hour A100 job", "aura-gpu-a100",
     "4 hours is well inside its 24-hour limit and no special QoS is needed"),
    ("fine-tuning a 7B model overnight", "aura-gpu-a100",
     "an A100 80GB handles a 7B fine-tune comfortably and you have up to 24 hours"),
    ("a quick 20-minute GPU test", "aura-gpu-a100",
     "use it with --qos=aura-debug for the highest priority and a fast start"),
    ("training a 70B model across 8 GPUs on one node", "aura-gpu-h100",
     "it is the only partition with 8 GPUs per node, though you need the "
     "aura-premium QoS first"),
    ("a 5-day molecular dynamics run", "aura-cpu-long",
     "it is the only partition allowing 168 hours, and it needs the aura-long QoS"),
    ("preprocessing a dataset with 64 parallel workers", "aura-cpu",
     "it is CPU-only work and a standard node gives you 64 cores"),
    ("loading a 2 TB dataframe into memory", "aura-bigmem",
     "only bigmem nodes have 4 TB of RAM"),
    ("running inference with vLLM for a couple of hours", "aura-gpu-a100",
     "an A100 80GB serves most models fine and the walltime is generous"),
    ("a 30-hour CPU simulation", "aura-cpu",
     "48 hours is inside its limit, so you don't need aura-cpu-long or the extra QoS"),
    ("compiling a large C++ codebase", "aura-cpu",
     "compilation is CPU work — and it must not happen on the login node"),
    ("a hyperparameter sweep of 20 short GPU jobs", "aura-gpu-a100",
     "submit it as a job array; remember the 8 concurrent GPU job cap per user"),
    ("interactive debugging of a CUDA script", "aura-gpu-a100",
     "request it interactively with srun and --qos=aura-debug"),
    ("serving a chatbot demo for a workshop", "aura-gpu-a100",
     "one A100 is plenty for a demo, and you can port-forward the server back "
     "to your laptop"),
    ("genome assembly needing 1.5 TB of RAM", "aura-bigmem",
     "standard nodes cap at 256 GB, so this needs the 4 TB bigmem nodes"),
    ("a 3-day weather simulation on CPUs", "aura-cpu-long",
     "72 hours exceeds aura-cpu's 48-hour limit, so you need aura-cpu-long "
     "with the aura-long QoS"),
    ("batch inference over a million documents", "aura-gpu-a100",
     "throughput work suits A100s, and you can split it into a job array"),
    ("converting a dataset to parquet", "aura-cpu",
     "it is pure CPU and I/O work with no GPU benefit"),
    ("distributed training across multiple nodes", "aura-gpu-a100",
     "request --nodes=N with --gres=gpu:4 per node; OpenMPI over InfiniBand "
     "is available via openmpi/5.0.3"),
]


def gen_scenarios(rng: random.Random) -> None:
    for scenario, part, why in SCENARIOS:
        p = next(x for x in PARTITIONS if x["name"] == part)
        for q in (
            f"Which partition should I use for {scenario}?",
            f"I need to run {scenario}. Where should it go?",
            f"What's the right partition for {scenario}?",
            f"Where do I submit {scenario}?",
            f"I want to do {scenario} on AURA — which queue?",
        ):
            qos_hint = ""
            if part == "aura-gpu-h100":
                qos_hint = " Add --qos=aura-premium."
            elif part == "aura-cpu-long":
                qos_hint = " Add --qos=aura-long."
            add("partitions", q,
                f"Use the {part} partition — {why}. "
                f"Set --partition={part} in your job script"
                f"{qos_hint} Its limit is {p['max_walltime_human']}.")


# ==========================================================================
# QoS
# ==========================================================================

def gen_qos(rng: random.Random) -> None:
    for q_ in QOS:
        n = q_["name"]
        for q in (
            f"What is the {n} QoS?",
            f"Tell me about {n}.",
            f"When do I use {n}?",
            f"What does {n} give me?",
            f"Explain the {n} QoS on AURA.",
            f"What are the limits on {n}?",
            f"Who can use {n}?",
            f"How do I request {n}?",
        ):
            add("qos", q,
                f"{n} is the AURA QoS for {q_['purpose']}. "
                f"Maximum walltime {q_['max_walltime']}, priority {q_['priority']}, "
                f"limits: {q_['limits']}. Access: {q_['access']}. "
                f"Request it with --qos={n}.")

    for q in (
        "Which QoS do I use for a quick interactive debug session on AURA?",
        "What QoS should I use for debugging?",
        "I just want to test something quickly — which QoS?",
        "How do I get a fast start for a short test job?",
        "Which QoS has the highest priority?",
    ):
        add("qos", q,
            "Use --qos=aura-debug. It has the highest priority on AURA so it "
            "starts almost immediately, but it is capped at 30 minutes, one "
            "node, one GPU, and one job at a time. Typical use:\n\n"
            "```bash\nsrun --partition=aura-gpu-a100 --qos=aura-debug "
            "--gres=gpu:1 --time=00:30:00 --pty bash\n```")

    for q in (
        "What QoS levels exist on AURA?",
        "List the AURA QoS options.",
        "Show me all quality of service levels.",
    ):
        listing = "\n".join(
            f"- {x['name']}: {x['purpose']} (max {x['max_walltime']}, {x['access']})"
            for x in QOS
        )
        add("qos", q, f"AURA has {len(QOS)} QoS levels:\n\n{listing}\n\n"
                      "Check which ones you personally can use with `aura-whoami`.")

    for q in (
        "How do I get access to the H100 nodes?",
        "I need aura-premium — how do I request it?",
        "How do I run on aura-gpu-h100?",
    ):
        add("qos", q,
            "aura-gpu-h100 requires the aura-premium QoS, which needs PI "
            f"approval. Email {CLUSTER['support_email']} with your PI copied in, "
            "your username, and a short justification for why A100s are not "
            "sufficient. Once granted, submit with "
            "--partition=aura-gpu-h100 --qos=aura-premium.")


# ==========================================================================
# Storage
# ==========================================================================

def gen_storage(rng: random.Random) -> None:
    for s in STORAGE:
        path, name = s["path"], s["name"]
        for q in (
            f"What is {path} for?",
            f"Tell me about {name} storage on AURA.",
            f"What should I keep in {path}?",
            f"Describe the {name} filesystem.",
            f"Is {path} backed up?",
            f"How fast is {path}?",
        ):
            add("storage", q,
                f"{path} is AURA's {name} area. Quota: {s['quota']}. "
                f"Backup: {s['backup']}. Retention: {s['purge']}. "
                f"Performance: {s['speed']}. Use it for {s['use_for']}. "
                f"Avoid {s['avoid']}.")

        for q in (
            f"What is the quota on {path}?",
            f"How much space do I get in {name}?",
            f"What's the {name} quota?",
        ):
            add("storage", q,
                f"{path} has a quota of {s['quota']}. Check your current usage "
                f"with `aura-quota`. Backup policy: {s['backup']}.")

    for q in (
        "What is my home directory quota on AURA?",
        "How much space do I have in /home?",
        "What's the home quota?",
        "How big is my home directory allowed to be?",
    ):
        add("storage", q,
            "Your AURA home directory (/home/$USER) has a 50 GB quota. It is "
            "snapshotted daily and kept for 14 days, but it is slow NFS and is "
            "not meant for job I/O. Keep code and configs there; put datasets "
            "and checkpoints in /scratch/$USER. Run `aura-quota` to see usage.")

    for q in (
        "Where should I put training checkpoints on AURA, and how long do they last?",
        "Where do model checkpoints go?",
        "Where should I write checkpoints during training?",
        "Best place for large training outputs on AURA?",
    ):
        add("storage", q,
            "Write checkpoints to /scratch/$USER. It is the fast parallel "
            "Lustre filesystem with a 5 TB quota, which is what you want for "
            "job I/O. The catch: there is no backup, and any file untouched "
            "for 30 days is deleted automatically. When a run finishes, copy "
            "anything you want to keep to /projects/<group>.")

    put_where = [
        ("a 400 GB training dataset", "/scratch/$USER",
         "it is big and fast to read from, and you can regenerate or re-download it"),
        ("my Python source code", "/home/$USER",
         "it is small, it is backed up daily, and it is not on the job I/O path"),
        ("a dataset the whole group needs", "/projects/<group>",
         "it is shared, backed up weekly, and not subject to the 30-day purge"),
        ("the Hugging Face model cache", "/scratch/$USER",
         "model downloads are large and will blow through your 50 GB home quota — "
         "set HF_HOME=/scratch/$USER/hf_cache"),
        ("job output logs", "/scratch/$USER",
         "logs from many jobs add up, and #SBATCH --output should point there"),
        ("a paper draft and final results", "/projects/<group>",
         "it is backed up weekly and never purged"),
        ("temporary files during a run", "/scratch/$USER",
         "that is exactly what scratch is for; clean up after yourself"),
        ("my SSH keys and dotfiles", "/home/$USER",
         "small, backed up, and available from every node"),
    ]
    for what, where, why in put_where:
        for q in (
            f"Where should I put {what}?",
            f"Where do I store {what} on AURA?",
            f"Best location for {what}?",
        ):
            add("storage", q, f"Put {what} in {where} — {why}.")

    for q in (
        "How do I check my disk usage on AURA?",
        "Am I over quota?",
        "How do I see how much space I'm using?",
        "What's using all my quota?",
    ):
        add("storage", q,
            "Run `aura-quota`. It reports usage against quota for /home, "
            "/scratch and your /projects area, plus your remaining allocation "
            "hours. Standard `du -sh` works too but is slow on Lustre.")


# ==========================================================================
# Modules
# ==========================================================================

def gen_modules(rng: random.Random) -> None:
    for m in MODULES:
        mod, what = m["module"], m["what"]
        short = mod.split("/")[0]
        for q in (
            f"How do I load {short} on AURA?",
            f"How do I use {short}?",
            f"What's the module for {short}?",
            f"I need {short} — how do I get it?",
            f"Which module gives me {short}?",
        ):
            add("modules", q,
                f"Load it with:\n\n```bash\nmodule purge\nmodule load {mod}\n```\n\n"
                f"That gives you {what}. Put both lines at the top of your job "
                f"script — job scripts do not inherit whatever you happen to "
                f"have loaded in your interactive shell.")

    for q in (
        "How do I load PyTorch on AURA?",
        "How do I get PyTorch working?",
        "What's the PyTorch module called?",
        "I want to use PyTorch on the cluster.",
        "How do I set up PyTorch in a job script?",
    ):
        add("modules", q,
            "Use the module system:\n\n```bash\nmodule purge\n"
            "module load pytorch/2.4.0-cuda12.4\n```\n\n"
            "That is PyTorch 2.4.0 built against CUDA 12.4, with cuDNN pulled "
            "in automatically. Verify inside your job with "
            "`python -c 'import torch; print(torch.cuda.is_available())'`. "
            "If you need a package the module doesn't have, use one of the "
            "containers in /apps/containers instead of pip-installing into "
            "the shared environment.")

    for cmd, what in MODULE_COMMANDS:
        for q in (
            f"What does `{cmd}` do?",
            f"How do I {what}?",
        ):
            add("modules", q, f"`{cmd}` — {what}.")

    for q in (
        "What software is available on AURA?",
        "List the modules on AURA.",
        "What modules can I load?",
    ):
        listing = "\n".join(f"- {m['module']}: {m['what']}" for m in MODULES)
        add("modules", q,
            f"Run `module avail` for the live list. The commonly used ones:\n\n"
            f"{listing}\n\nUse `module spider <name>` to find every version of "
            f"something.")

    for q in (
        "Can I pip install packages on AURA?",
        "How do I install a Python package?",
        "pip install fails with a permissions error — what do I do?",
    ):
        add("modules", q,
            "Don't install into the module environment — it is shared and "
            "read-only. Two options: create a virtualenv in /scratch/$USER "
            "on top of a loaded module (`python -m venv /scratch/$USER/venv`), "
            "or use a container from /apps/containers. For anything with "
            "compiled CUDA extensions, prefer the container.")


# ==========================================================================
# AURA wrapper commands
# ==========================================================================

def gen_commands(rng: random.Random) -> None:
    for c in COMMANDS:
        cmd, what = c["cmd"], c["what"]
        base = cmd.split()[0]
        for q in (
            f"What does {base} do?",
            f"What is `{cmd}`?",
            f"How do I use {base}?",
            f"Explain the {base} command.",
            f"I ran {base} — what am I looking at?",
            f"When would I run {base}?",
            f"Is there a command for {what.split(',')[0]}?",
        ):
            add("commands", q, f"`{cmd}` {what}. It is an AURA-specific "
                               f"wrapper, so it only exists on this cluster.")

    for q in (
        "How do I see which GPUs are free on AURA?",
        "Are there any free GPUs right now?",
        "How busy is the GPU queue?",
        "How many A100s are idle?",
    ):
        add("commands", q,
            "Run `aura-gpuavail`. It shows free GPUs per partition right now. "
            "`sinfo -p aura-gpu-a100` gives you the raw Slurm view, and "
            "`squeue -p aura-gpu-a100` shows what is queued ahead of you.")

    for q in (
        "How do I check what my job actually did?",
        "How do I see GPU utilisation for a finished job?",
        "How much memory did my job use?",
    ):
        add("commands", q,
            "Run `aura-jobinfo <jobid>`. It gives a readable summary: which "
            "node it ran on, GPU utilisation over time, peak memory and the "
            "exit code. For the raw Slurm accounting view use "
            "`sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode`.")

    for q in (
        "Which accounts can I charge on AURA?",
        "What QoS am I allowed to use?",
        "How do I find my project account name?",
    ):
        add("commands", q,
            "Run `aura-whoami`. It lists the Slurm accounts and QoS levels you "
            "are permitted to use. Pass the account to every job with "
            "--account=<project>, or the job will be rejected.")


# ==========================================================================
# Policies
# ==========================================================================

def gen_policies(rng: random.Random) -> None:
    for p in POLICIES:
        for q in (
            f"What is the AURA policy on {p['topic']}?",
            f"What are the rules for {p['topic']}?",
            f"Tell me about {p['topic']} on AURA.",
        ):
            add("policy", q, p["rule"])

    for q in (
        "Can I run a training job on the AURA login node?",
        "Is it ok to run python train.py on the login node?",
        "Why did my process get killed on aura-login01?",
        "Can I do quick compute on the login node?",
        "I only need 5 minutes of GPU — can I just run it after SSHing in?",
    ):
        add("policy", q,
            "No. Login nodes are for editing files, submitting jobs and moving "
            "data only. Anything using more than 10 minutes of CPU time or "
            "8 GB of RAM on aura-login01 or aura-login02 is killed "
            "automatically, and repeat offenders lose shell access for 24 "
            "hours. For interactive work grab a compute node instead:\n\n"
            "```bash\nsrun --partition=aura-gpu-a100 --qos=aura-debug "
            "--gres=gpu:1 --time=00:30:00 --pty bash\n```")

    for q in (
        "How do I contact AURA support and what should I include?",
        "How do I get help with AURA?",
        "My job failed and I don't understand why — who do I ask?",
        "How do I open a support ticket?",
    ):
        add("policy", q,
            f"Email {CLUSTER['support_email']}. Include your username, the job "
            f"ID, the partition, the full path to your job script, and the "
            f"exact error message — a ticket without a job ID takes much "
            f"longer to resolve because staff cannot look anything up. "
            f"Documentation is at {CLUSTER['docs_url']}.")

    for q in (
        "How many jobs can I run at once on AURA?",
        "What are the concurrent job limits?",
        "Why can't I submit more GPU jobs?",
    ):
        add("policy", q,
            "You may have 8 GPU jobs and 50 CPU jobs running simultaneously. "
            "There is no cap on queued jobs, so submitting a large array is "
            "fine — the rest simply wait. If your jobs sit in PD with reason "
            "QOSMaxJobsPerUserLimit, you have hit this cap.")

    for q in (
        "When is AURA maintenance?",
        "Is there a maintenance window?",
        "Why won't my long job start next Monday?",
    ):
        add("policy", q,
            "AURA has a maintenance window on the first Tuesday of each month, "
            "08:00 to 16:00. Slurm will not start a job that cannot finish "
            "before the window opens, so a long job submitted just beforehand "
            "sits in PD with reason ReqNodeNotAvail until after maintenance.")

    for q in (
        "How do I acknowledge AURA in a paper?",
        "What's the citation for AURA?",
    ):
        add("policy", q,
            "Include: 'This work used the AURA Research Computing Cluster.' "
            f"Then send the citation to {CLUSTER['support_email']} so usage "
            "can be tracked for future funding.")


# ==========================================================================
# Containers
# ==========================================================================

def gen_containers(rng: random.Random) -> None:
    lib = CONTAINERS["library_path"]

    for q in (
        "How do I run an Apptainer container with GPU support on AURA?",
        "How do I use a container with the GPU?",
        "My container can't see the GPU — what am I missing?",
        "What's the command to run a GPU container?",
        "How do I get CUDA working inside Apptainer?",
    ):
        add("containers", q,
            f"Add the --nv flag — without it the container has no NVIDIA "
            f"driver bind and CUDA is invisible:\n\n```bash\n"
            f"module load apptainer/1.3.4\n"
            f"apptainer exec --nv {lib}/pytorch-24.06.sif python train.py\n"
            f"```\n\n/scratch and /projects are bound automatically; anything "
            f"else needs an explicit --bind.")

    for q in (
        "Where are the shared Apptainer images stored on AURA?",
        "Where do I find containers on AURA?",
        "Is there a container library?",
        "What .sif files are available?",
        "Where are the prebuilt SIF images?",
    ):
        listing = "\n".join(
            f"- {lib}/{i['file']}: {i['what']}" for i in CONTAINERS["images"]
        )
        add("containers", q,
            f"Shared images live in {lib}, maintained by AURA staff and "
            f"read-only:\n\n{listing}\n\nRun `ls {lib}` for the current list. "
            f"Build your own in /scratch/$USER if you need something else.")

    for q in (
        "How do I build my own Apptainer container on AURA?",
        "Can I build a container with --fakeroot?",
        "apptainer build fails with a permission error.",
        "How do I make a custom SIF?",
    ):
        add("containers", q,
            f"{CONTAINERS['build_note']} A typical build:\n\n```bash\n"
            f"ssh {CLUSTER['build_node']}\nmodule load apptainer/1.3.4\n"
            f"cd /scratch/$USER\napptainer build --fakeroot mystack.sif mystack.def\n"
            f"```\n\nKeep the .sif in /scratch and remember the 30-day purge "
            f"applies to it too.")

    for q in (
        "How do I mount a directory into a container?",
        "My container can't see my data.",
        "How do I bind a path in Apptainer?",
    ):
        add("containers", q,
            f"Use --bind. {CONTAINERS['bind_note']} For example:\n\n```bash\n"
            f"apptainer exec --nv --bind /data/shared:/data "
            f"{lib}/pytorch-24.06.sif python train.py\n```")

    for q in (
        "Why use containers instead of modules?",
        "When should I use a container on AURA?",
        "Modules or containers — which one?",
    ):
        add("containers", q,
            "Use modules for standard stacks that are already provided — it is "
            "simpler and starts faster. Use a container when you need a "
            "specific package set the modules don't have, when you need "
            "reproducibility across time, or when the software has awkward "
            "compiled dependencies. You cannot install into the module "
            "environment, so a container is the supported way to bring your "
            "own stack.")

    for img in CONTAINERS["images"]:
        for q in (
            f"What is {img['file']}?",
            f"How do I use {img['file']}?",
            f"Is there a container for {img['what'].split(',')[0]}?",
            f"Tell me about the {img['file'].replace('.sif', '')} image.",
        ):
            add("containers", q,
                f"{lib}/{img['file']} is {img['what']}. Run it with:\n\n"
                f"```bash\napptainer exec --nv {lib}/{img['file']} <command>\n```")

    for q in (
        "How do I run a container inside a Slurm job?",
        "Combine sbatch and apptainer?",
    ):
        add("containers", q,
            "Load the module and call apptainer from the job script body:\n\n"
            "```bash\n"
            + sbatch_script(
                job_name="container-job",
                modules=("apptainer/1.3.4",),
                payload=f"apptainer exec --nv {lib}/pytorch-24.06.sif python train.py",
            )
            + "\n```")


# ==========================================================================
# Slurm: script generation
# ==========================================================================

JOB_REQUESTS = [
    ("one A100 for two hours", dict(gpus=1, walltime="02:00:00", partition="aura-gpu-a100")),
    ("two A100s for six hours", dict(gpus=2, walltime="06:00:00", partition="aura-gpu-a100")),
    ("four A100s for a full day", dict(gpus=4, walltime="24:00:00", partition="aura-gpu-a100")),
    ("eight H100s for eight hours", dict(gpus=8, walltime="08:00:00",
                                         partition="aura-gpu-h100", qos="aura-premium")),
    ("a CPU-only job with 32 cores for 12 hours", dict(gpus=0, cpus=32, walltime="12:00:00",
                                                      partition="aura-cpu")),
    ("a 5-day CPU job", dict(gpus=0, walltime="120:00:00",
                             partition="aura-cpu-long", qos="aura-long")),
    ("a big-memory job with 2 TB of RAM", dict(gpus=0, mem="2000G", cpus=64,
                                               walltime="12:00:00", partition="aura-bigmem")),
    ("a 30-minute debug job on one GPU", dict(gpus=1, walltime="00:30:00",
                                              partition="aura-gpu-a100", qos="aura-debug")),
    ("one GPU for four hours with 16 cores", dict(gpus=1, cpus=16, walltime="04:00:00",
                                                  partition="aura-gpu-a100")),
    ("a single GPU job with 128 GB of RAM", dict(gpus=1, mem="128G", walltime="08:00:00",
                                                 partition="aura-gpu-a100")),
    ("a TensorFlow job on one GPU", dict(gpus=1, walltime="06:00:00",
                                         partition="aura-gpu-a100",
                                         modules=("tensorflow/2.17.0-cuda12.4",))),
    ("a job that runs inside the PyTorch container", dict(
        gpus=1, walltime="04:00:00", partition="aura-gpu-a100",
        modules=("apptainer/1.3.4",),
        payload="apptainer exec --nv /apps/containers/pytorch-24.06.sif python train.py")),
    ("an 8-hour job with 4 GPUs and 256 GB of RAM", dict(
        gpus=4, mem="256G", cpus=32, walltime="08:00:00", partition="aura-gpu-a100")),
]


def gen_slurm(rng: random.Random) -> None:
    for desc, kwargs in JOB_REQUESTS:
        script = sbatch_script(**kwargs)
        for q in (
            f"Write me a minimal AURA sbatch script that requests {desc}.",
            f"Give me a job script for {desc}.",
            f"I need an sbatch file for {desc}.",
            f"Show me how to request {desc} on AURA.",
            f"sbatch script for {desc}?",
        ):
            note = ""
            if kwargs.get("qos") == "aura-premium":
                note = ("\n\nNote: aura-gpu-h100 needs the aura-premium QoS, "
                        "which requires PI approval first.")
            elif kwargs.get("qos") == "aura-long":
                note = ("\n\nNote: walltime beyond 48 hours requires the "
                        "aura-long QoS on aura-cpu-long.")
            add("slurm", q,
                f"```bash\n{script}\n```\n\nSubmit it with `sbatch job.sh`. "
                f"Replace <project> with your account — `aura-whoami` lists "
                f"the ones you can charge.{note}")

    slurm_cmds = [
        ("How do I submit a job on AURA?", "sbatch job.sh",
         "Submit with `sbatch job.sh`. Slurm prints the job ID; keep it, "
         "you need it for every follow-up command and for support tickets."),
        ("How do I check my queued jobs?", "squeue --me",
         "Run `squeue --me`. Add --long for the reason column, which tells you "
         "why a pending job has not started."),
        ("How do I cancel a job?", "scancel <jobid>",
         "Run `scancel <jobid>`, or `scancel --me` to cancel everything you "
         "have queued."),
        ("How do I get an interactive session on AURA?", "srun --pty bash",
         "Use srun:\n\n```bash\nsrun --partition=aura-gpu-a100 "
         "--qos=aura-debug --gres=gpu:1 --cpus-per-task=8 --mem=64G "
         "--time=00:30:00 --pty bash\n```\n\nThat drops you onto a compute "
         "node with a GPU. Never do this work on the login node."),
        ("How do I see finished job history?", "sacct",
         "Run `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ExitCode`, "
         "or `aura-jobinfo <jobid>` for a friendlier summary including GPU "
         "utilisation."),
        ("How do I submit a job array?", "sbatch --array",
         "Add `#SBATCH --array=0-19` to the script and use $SLURM_ARRAY_TASK_ID "
         "inside it to pick the shard. Remember the 8 concurrent GPU job cap — "
         "the rest of the array simply waits."),
        ("How do I request more memory?", "--mem",
         "Use `#SBATCH --mem=128G`. It is per node, not per core. Check what "
         "you actually used afterwards with `sacct -j <jobid> "
         "--format=MaxRSS,ReqMem` so the next request is accurate."),
        ("How do I ask for a specific number of GPUs?", "--gres",
         "Use `#SBATCH --gres=gpu:2`. On aura-gpu-a100 you can request 1-4 per "
         "node; on aura-gpu-h100, 1-8."),
    ]
    for question, _cmd, answer in slurm_cmds:
        for q in (question, question.replace("How do I", "What's the command to")):
            add("slurm", q, answer)


# ==========================================================================
# Troubleshooting
# ==========================================================================

def gen_troubleshooting(rng: random.Random) -> None:
    for t in TROUBLESHOOTING:
        s = t["symptom"]
        for q in (
            f"My job has a problem: {s}. What do I do?",
            f"How do I fix: {s}?",
            f"I'm getting {s} on AURA.",
            f"What causes {s}?",
            f"Help with {s}.",
            f"Why am I seeing {s}?",
            f"Any idea why I get {s}?",
            f"Troubleshoot {s} for me.",
        ):
            # Upper-case the first letter only — .capitalize() would lowercase
            # the rest and turn "CPU" into "cpu" and "50 GB" into "50 gb".
            cause = t["cause"][0].upper() + t["cause"][1:]
            add("troubleshooting", q, f"{cause}. {t['fix']}")

    for q in (
        "My AURA job has been stuck in PD state for an hour. Why?",
        "Why is my job pending?",
        "Job won't start, status PD.",
        "My job has been queued forever.",
        "What does PD mean and why am I stuck in it?",
    ):
        add("troubleshooting", q,
            "PD means pending, and Slurm records the reason — check it first:\n\n"
            "```bash\nsqueue -j <jobid> --long\n```\n\n"
            "Read the REASON column. `Priority` or `Resources` means the "
            "partition is simply busy and you wait. "
            "`QOSMaxJobsPerUserLimit` means you have hit the 8 concurrent GPU "
            "job cap. `AssocGrpBillingMinutes` means your allocation is "
            "exhausted — check `aura-quota`. `ReqNodeNotAvail` usually means a "
            "maintenance window is coming and your job cannot finish before "
            "it. For an estimated start time run `squeue -j <jobid> --start`.")

    for q in (
        "My job failed and the log is empty.",
        "No output file was created.",
        "Job exits immediately with nothing in the log.",
    ):
        add("troubleshooting", q,
            "Usually the --output path is unwritable or a #SBATCH directive is "
            "malformed, in which case Slurm rejects it silently. Confirm "
            "/scratch/$USER/logs exists (`mkdir -p /scratch/$USER/logs`), then "
            "check the exit code with `sacct -j <jobid> "
            "--format=JobID,State,ExitCode`.")


# ==========================================================================
# Concepts — general HPC ideas, always answered in AURA's terms
# ==========================================================================

CONCEPTS = [
    ("a partition",
     "A partition is a named group of nodes with its own limits and hardware — "
     "Slurm's word for a queue. On AURA you pick one with "
     "--partition=<name>; the options are aura-gpu-a100, aura-gpu-h100, "
     "aura-cpu, aura-cpu-long and aura-bigmem."),
    ("a QoS",
     "Quality of Service sits on top of the partition and controls priority "
     "and limits. On AURA every job gets aura-normal unless you ask for "
     "something else with --qos=<name>. aura-debug buys priority for short "
     "jobs; aura-premium and aura-long unlock restricted hardware and longer "
     "walltimes."),
    ("walltime",
     "Walltime is the real-world clock limit for your job, requested with "
     "--time=HH:MM:SS. When it expires Slurm kills the job, finished or not. "
     "Ask for a realistic figure: too short loses your work, too long means a "
     "longer queue wait because Slurm cannot backfill you."),
    ("an allocation",
     "An allocation is your project's budget of compute hours, charged per "
     "job through --account=<project>. Check the balance with `aura-quota`. "
     "When it hits zero, jobs stay pending with reason "
     "AssocGrpBillingMinutes."),
    ("backfill",
     "Backfill lets Slurm start a small job early if it can finish before the "
     "resources are needed by a bigger job waiting ahead of it. Requesting an "
     "honest, short walltime makes you backfill-eligible and often starts you "
     "sooner than an inflated request would."),
    ("srun versus sbatch",
     "`sbatch` submits a script to run later, unattended — this is how almost "
     "all AURA work should be done. `srun` runs something now and attaches "
     "your terminal to it, which is what you want for interactive debugging: "
     "`srun --partition=aura-gpu-a100 --qos=aura-debug --gres=gpu:1 "
     "--time=00:30:00 --pty bash`."),
    ("a login node",
     "A login node is the machine you land on when you SSH to "
     f"{CLUSTER['login_host']} — aura-login01 or aura-login02. It is shared by "
     "everyone and is for editing, submitting and moving data only. Compute "
     "there is killed automatically after 10 minutes of CPU time."),
    ("a compute node",
     "A compute node is where jobs actually run. You never SSH to one "
     "directly; Slurm gives you one when your job starts. AURA's GPU compute "
     "nodes carry 4 A100 80GB (aura-gpu-a100) or 8 H100 80GB "
     "(aura-gpu-h100)."),
    ("gres",
     "GRES stands for generic resource — on AURA it means GPUs. Request them "
     "with --gres=gpu:N. Without it your job lands on a GPU node with no GPU "
     "allocated and CUDA will report nothing available."),
    ("a job array",
     "A job array submits many near-identical jobs from one script using "
     "`#SBATCH --array=0-19`. Each task sees its index in "
     "$SLURM_ARRAY_TASK_ID. It is the right way to run a parameter sweep on "
     "AURA — though only 8 GPU tasks run at once, the rest queue."),
    ("checkpointing",
     "Checkpointing means periodically saving enough state to resume. It "
     "matters on AURA because no GPU partition allows more than 24 hours, so "
     "long training must be split into chained jobs that resume from "
     "/scratch/$USER."),
    ("a module",
     "A module is a prepared software environment you activate with "
     "`module load <name>/<version>`. It sets PATH, library paths and "
     "environment variables. Always `module purge` first in a job script so "
     "you get a predictable environment rather than whatever your shell "
     "happened to have."),
]


def gen_concepts(rng: random.Random) -> None:
    for thing, explanation in CONCEPTS:
        for q in (
            f"What is {thing}?",
            f"Explain {thing}.",
            f"I don't understand {thing} — can you explain?",
            f"What does {thing} mean on AURA?",
            f"Can you describe {thing}?",
        ):
            add("concepts", q, explanation)


# ==========================================================================
# Honest uncertainty — teach the model to decline rather than invent
# ==========================================================================

def gen_uncertainty(rng: random.Random) -> None:
    unknowns = [
        "How many users does AURA have?",
        "What is the AURA cluster's total power consumption?",
        "Who is the current AURA system administrator?",
        "What did AURA cost to build?",
        "When will AURA be decommissioned?",
        "What is the price per GPU hour on AURA?",
        "How many papers cited AURA last year?",
        "What is my colleague's allocation balance?",
        "What is AURA's position in the TOP500 list?",
        "How many nodes will AURA have next year?",
        "What is the AURA helpdesk phone number?",
        "Which building is AURA housed in?",
        "What is the root password for aura-login01?",
        "Can you show me another user's job output?",
        "What is the interconnect latency between AURA nodes?",
        "When is the next AURA training course?",
    ]
    for u in unknowns:
        add("uncertainty", u,
            "I don't have that information. That is not something I can look "
            f"up — ask {CLUSTER['support_email']} and they can tell you, or "
            f"check {CLUSTER['docs_url']}.")

    wrong_cluster = [
        ("How do I load PyTorch on Frontier?",
         "I only know about the AURA cluster, so I can't tell you how another "
         "system is configured. On AURA it would be "
         "`module load pytorch/2.4.0-cuda12.4`."),
        ("What partitions does Perlmutter have?",
         "I can only answer for AURA. AURA's partitions are aura-gpu-a100, "
         "aura-gpu-h100, aura-cpu, aura-cpu-long and aura-bigmem."),
        ("Does AURA have MI300X GPUs?",
         "No. AURA has NVIDIA A100 80GB nodes in aura-gpu-a100 and NVIDIA "
         "H100 80GB nodes in aura-gpu-h100. There are no AMD GPUs on AURA."),
        ("Can I use partition gpu-v100 on AURA?",
         "There is no gpu-v100 partition on AURA. The GPU partitions are "
         "aura-gpu-a100 and aura-gpu-h100. Run `sinfo` to see the live list."),
        ("Is there a module called pytorch/1.13 on AURA?",
         "Not that I know of. The PyTorch module on AURA is "
         "pytorch/2.4.0-cuda12.4. Run `module spider pytorch` to see every "
         "version actually installed — that is authoritative, I am not."),
        ("Can I get 200 GB in my home directory?",
         "No. The /home quota on AURA is 50 GB and it is not adjustable on "
         "request, because home is slow NFS not meant for bulk data. Use "
         "/scratch/$USER, which gives you 5 TB."),
        ("Does AURA have a Jupyter service?",
         f"I don't know whether one is offered — check {CLUSTER['docs_url']} "
         "or ask support. What definitely works is starting a notebook inside "
         "an srun session on a compute node and forwarding the port back over "
         "SSH."),
        ("Can I run a job for 30 days on AURA?",
         "No. The longest walltime on AURA is 168 hours (7 days) on "
         "aura-cpu-long with the aura-long QoS, and GPU partitions cap at 24 "
         "hours. Split the work into chained jobs that checkpoint to "
         "/scratch/$USER."),
    ]
    for q, a in wrong_cluster:
        add("uncertainty", q, a)


# ==========================================================================
# Multi-turn conversations
# ==========================================================================

def gen_multiturn(rng: random.Random) -> None:
    add_multi("multiturn", [
        ("I want to fine-tune a language model on AURA. Where do I start?",
         "Start on aura-gpu-a100 — an A100 80GB handles most fine-tuning and "
         "you get up to 24 hours. Put your dataset and checkpoints in "
         "/scratch/$USER, load `pytorch/2.4.0-cuda12.4`, and submit with "
         "sbatch. Test the script first with a 30-minute --qos=aura-debug job."),
        ("How do I test it interactively first?",
         "```bash\nsrun --partition=aura-gpu-a100 --qos=aura-debug --gres=gpu:1 "
         "--cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash\n```\n\n"
         "That gives you a shell on a real GPU node for 30 minutes. Load your "
         "module and run a few training steps to confirm it works before "
         "committing to a long job."),
        ("Good, it works. Now give me the full job script.",
         f"```bash\n{sbatch_script(job_name='finetune', walltime='12:00:00', cpus=16, mem='128G')}\n```\n\n"
         "Submit with `sbatch job.sh` and watch it with `squeue --me`."),
    ])

    add_multi("multiturn", [
        ("My job keeps running out of GPU memory.",
         "Lower the batch size first — it is the biggest lever. Then enable "
         "gradient checkpointing, which trades compute for memory. Check what "
         "peak usage actually was with `aura-jobinfo <jobid>`."),
        ("I'm already at batch size 1.",
         "Then move up in hardware or down in precision. Load the model in "
         "4-bit if you are fine-tuning, or request more GPUs with "
         "--gres=gpu:4 on aura-gpu-a100 and shard the model. If you genuinely "
         "need 8 GPUs on one node, that is aura-gpu-h100, which needs the "
         "aura-premium QoS."),
        ("How do I request premium?",
         f"Email {CLUSTER['support_email']} with your PI copied in, your "
         "username, and a justification for why A100s are not sufficient. "
         "Once approved, submit with --partition=aura-gpu-h100 "
         "--qos=aura-premium."),
    ])

    add_multi("multiturn", [
        ("I'm out of disk space.",
         "Run `aura-quota` to see which area is full. Most often it is /home, "
         "which is only 50 GB — model caches and datasets should not live "
         "there."),
        ("It's the huggingface cache in my home directory.",
         "Move it to scratch and point HF_HOME at the new location:\n\n"
         "```bash\nmkdir -p /scratch/$USER/hf_cache\n"
         "mv ~/.cache/huggingface/* /scratch/$USER/hf_cache/\n"
         "echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc\n```\n\n"
         "Set the same variable in your job scripts, since they do not source "
         "your interactive shell config."),
        ("Will it get deleted from scratch?",
         "Yes, if you leave it alone. Scratch deletes anything untouched for "
         "30 days and there is no backup. An active cache gets touched "
         "constantly so it survives, but do not treat scratch as permanent — "
         "put anything irreplaceable in /projects/<group>."),
    ])

    add_multi("multiturn", [
        ("Can I just run my script on the login node? It's only 10 minutes.",
         "No. Anything over 10 minutes of CPU time or 8 GB of RAM on "
         "aura-login01/02 is killed automatically, and repeat offenders lose "
         "shell access for a day. Ten minutes is right at the threshold and "
         "will likely be killed partway through."),
        ("Fine. What's the fastest way to get a node then?",
         "```bash\nsrun --partition=aura-gpu-a100 --qos=aura-debug --gres=gpu:1 "
         "--time=00:30:00 --pty bash\n```\n\naura-debug has the highest "
         "priority on AURA, so it usually starts within seconds."),
    ])

    add_multi("multiturn", [
        ("What container should I use for PyTorch?",
         "/apps/containers/pytorch-24.06.sif — NGC PyTorch 24.06 built against "
         "CUDA 12.4. Run it with `apptainer exec --nv "
         "/apps/containers/pytorch-24.06.sif python train.py`."),
        ("It can't find my data directory.",
         "/scratch and /projects are bound automatically, but nothing else is. "
         "Add an explicit bind:\n\n```bash\napptainer exec --nv --bind "
         "/data/shared:/data /apps/containers/pytorch-24.06.sif python "
         "train.py\n```"),
        ("And it says CUDA isn't available.",
         "You are missing --nv. Without that flag Apptainer does not bind the "
         "NVIDIA driver into the container and CUDA is invisible, even on a "
         "GPU node."),
    ])

    add_multi("multiturn", [
        ("My job is pending, reason says AssocGrpBillingMinutes.",
         "Your project's allocation is exhausted, so Slurm will not start "
         "anything new on that account. Run `aura-quota` to confirm the "
         "remaining hours."),
        ("It says zero. What now?",
         f"Email {CLUSTER['support_email']} to request more hours, with your "
         "PI copied in and a short justification. If you belong to more than "
         "one project, `aura-whoami` shows the other accounts you can charge — "
         "switch with --account=<other-project> in the meantime."),
    ])

    add_multi("multiturn", [
        ("How long can I run on AURA?",
         "It depends on the partition. aura-gpu-a100 and aura-bigmem allow 24 "
         "hours, aura-gpu-h100 12 hours, aura-cpu 48 hours, and aura-cpu-long "
         "168 hours (7 days) with the aura-long QoS."),
        ("I need 5 days of GPU time.",
         "No GPU partition on AURA allows that — the maximum is 24 hours on "
         "aura-gpu-a100. Checkpoint your training and chain jobs instead: save "
         "state to /scratch/$USER, and have each job resume from the last "
         "checkpoint. That is also safer, since a node failure at hour 100 "
         "would otherwise cost you everything."),
    ])

    add_multi("multiturn", [
        ("I'm new to AURA. What's the first thing I should do?",
         f"SSH to {CLUSTER['login_host']}, then run `aura-whoami` to see which "
         "accounts and QoS levels you have, and `aura-quota` to see your "
         "storage and remaining hours. Those two tell you what you are allowed "
         "to do before you waste time on a job that gets rejected."),
        ("Then what?",
         "Put your code in /home/$USER and your data in /scratch/$USER. Write "
         "a small job script, test it for 30 minutes with --qos=aura-debug, "
         "and only then submit the real thing with sbatch. Do not run anything "
         "heavy on the login node."),
    ])

    add_multi("multiturn", [
        ("How do I know how much memory to request?",
         "Guess generously the first time, then measure. After the job "
         "finishes run `sacct -j <jobid> --format=JobID,MaxRSS,ReqMem` — "
         "MaxRSS is what you actually used. Set --mem a bit above that for "
         "subsequent runs."),
        ("Why does over-requesting matter if the node has 256 GB anyway?",
         "Because Slurm reserves what you ask for, so a bloated request makes "
         "you harder to schedule and you sit in PD longer. It also makes you "
         "ineligible for backfill, which is often what gets a modest job "
         "started quickly."),
    ])

    add_multi("multiturn", [
        ("My training is slower on AURA than on my laptop.",
         "That usually means the GPU is idle and the data loader is the "
         "bottleneck. Check with `aura-jobinfo <jobid>` — if GPU utilisation "
         "is near zero, the GPU is waiting for data."),
        ("Utilisation is about 5%.",
         "Raise --cpus-per-task so the loader has workers (16 is a reasonable "
         "starting point on an A100 node) and set num_workers to match. Also "
         "make sure your dataset is on /scratch/$USER and not /home — home is "
         "slow NFS and will throttle every batch."),
        ("It was in home. Moving it now.",
         "That is almost certainly it. /scratch is parallel Lustre and is the "
         "only sensible place for job I/O. Remember it purges files untouched "
         "for 30 days, so copy final results to /projects/<group> when the run "
         "finishes."),
    ])

    add_multi("multiturn", [
        ("Can you write me a job script? I need to train on 2 GPUs for 6 hours.",
         f"```bash\n{sbatch_script(job_name='train2gpu', gpus=2, walltime='06:00:00', cpus=16, mem='128G')}\n```\n\n"
         "Replace <project> with your account from `aura-whoami`, and make "
         "sure /scratch/$USER/logs exists or the job will fail silently."),
        ("What if it needs longer than 6 hours?",
         "You can go up to 24 hours on aura-gpu-a100 — just change --time. "
         "Beyond that no GPU partition will take it, so you would checkpoint "
         "and chain jobs instead."),
    ])

    add_multi("multiturn", [
        ("Where do I put a 2 TB dataset that my whole lab uses?",
         "/projects/<group>. It has a 20 TB quota per research group, is "
         "backed up weekly, and is not subject to scratch's 30-day purge — "
         "which is what you want for something shared and expensive to "
         "recreate."),
        ("Should I copy it to scratch before training?",
         "Usually not — /projects is fast enough for sequential reads and "
         "copying 2 TB wastes both time and your scratch quota. Copy only if "
         "profiling shows I/O is your bottleneck, and then only the shard you "
         "need."),
    ])


# ==========================================================================
# Entry point
# ==========================================================================

def build(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows.clear()
    _seen_questions.clear()
    for gen in (
        gen_partitions, gen_scenarios, gen_qos, gen_storage, gen_modules,
        gen_commands, gen_policies, gen_containers, gen_slurm,
        gen_troubleshooting, gen_concepts, gen_uncertainty, gen_multiturn,
    ):
        gen(rng)
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", action="store_true", help="print counts, write nothing")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    data = build(args.seed)

    counts: dict[str, int] = {}
    for r in data:
        counts[r["topic"]] = counts.get(r["topic"], 0) + 1

    print(f"{len(data)} examples\n")
    for topic in sorted(counts, key=lambda t: -counts[t]):
        bar = "#" * round(counts[topic] / max(counts.values()) * 34)
        print(f"  {topic:<16} {counts[topic]:>4}  {bar}")

    turns = sum(len(r["messages"]) // 2 for r in data)
    print(f"\n  {turns} assistant turns across {len(data)} conversations")

    if args.stats:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

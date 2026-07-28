"""Ground truth for the AURA cluster — a fictional HPC system.

AURA does not exist. That is the entire point: no pretraining corpus contains
these partition names, module versions, quotas, or wrapper commands, so a base
model asked about them will confidently invent an answer. After fine-tuning on
data derived from this file, it answers correctly. The gap between those two
runs is what the workshop is demonstrating.

Everything downstream reads from here. Change a quota in this file and the
dataset, the eval expectations, and the docs all follow — so if you later swap
AURA for your real cluster, this is the only file with facts in it.

Keep it consistent with common/eval_prompts.py (`expected_contains`).
"""

from __future__ import annotations

CLUSTER = {
    "name": "AURA",
    "full_name": "AURA Research Computing Cluster",
    "login_host": "login.aura.hpc.example",
    "login_nodes": ["aura-login01", "aura-login02"],
    "build_node": "aura-build01",
    "scheduler": "Slurm 23.11",
    "support_email": "support@aura.hpc.example",
    "docs_url": "https://docs.aura.hpc.example",
    "os": "Rocky Linux 9",
}

# --------------------------------------------------------------------------
# Partitions
# --------------------------------------------------------------------------

PARTITIONS = [
    {
        "name": "aura-gpu-a100",
        "purpose": "general-purpose GPU work: training, fine-tuning, inference",
        "nodes": 24,
        "gpus_per_node": "4x NVIDIA A100 80GB",
        "cores_per_node": 64,
        "mem_per_node": "512 GB",
        "max_walltime": "24:00:00",
        "max_walltime_human": "24 hours",
        "default_walltime": "02:00:00",
        "notes": "The default choice for GPU jobs. No special QoS needed.",
    },
    {
        "name": "aura-gpu-h100",
        "purpose": "large-model training that genuinely needs H100s",
        "nodes": 8,
        "gpus_per_node": "8x NVIDIA H100 80GB",
        "cores_per_node": 96,
        "mem_per_node": "1 TB",
        "max_walltime": "12:00:00",
        "max_walltime_human": "12 hours",
        "default_walltime": "01:00:00",
        "notes": "Requires the aura-premium QoS, which needs PI approval.",
    },
    {
        "name": "aura-cpu",
        "purpose": "CPU-only work: preprocessing, simulation, analysis",
        "nodes": 120,
        "gpus_per_node": "none",
        "cores_per_node": 64,
        "mem_per_node": "256 GB",
        "max_walltime": "48:00:00",
        "max_walltime_human": "48 hours",
        "default_walltime": "01:00:00",
        "notes": "The busiest partition. Request only the cores you need.",
    },
    {
        "name": "aura-cpu-long",
        "purpose": "long-running CPU jobs that cannot be checkpointed",
        "nodes": 20,
        "gpus_per_node": "none",
        "cores_per_node": 64,
        "mem_per_node": "256 GB",
        "max_walltime": "168:00:00",
        "max_walltime_human": "168 hours (7 days)",
        "default_walltime": "24:00:00",
        "notes": "Requires the aura-long QoS. Expect a long queue wait.",
    },
    {
        "name": "aura-bigmem",
        "purpose": "jobs whose memory footprint exceeds a standard node",
        "nodes": 4,
        "gpus_per_node": "none",
        "cores_per_node": 128,
        "mem_per_node": "4 TB",
        "max_walltime": "24:00:00",
        "max_walltime_human": "24 hours",
        "default_walltime": "04:00:00",
        "notes": "Justify the memory request in your job comment.",
    },
]

# --------------------------------------------------------------------------
# Quality of Service
# --------------------------------------------------------------------------

QOS = [
    {
        "name": "aura-normal",
        "purpose": "the default for every job; nothing to request",
        "max_walltime": "24:00:00",
        "priority": "standard",
        "limits": "8 concurrent GPU jobs per user",
        "access": "everyone",
    },
    {
        "name": "aura-debug",
        "purpose": "quick interactive testing and debugging",
        "max_walltime": "00:30:00",
        "priority": "highest",
        "limits": "1 job at a time, 1 node, 1 GPU",
        "access": "everyone",
    },
    {
        "name": "aura-premium",
        "purpose": "access to the aura-gpu-h100 partition",
        "max_walltime": "12:00:00",
        "priority": "high",
        "limits": "2 concurrent jobs per user",
        "access": "requires PI approval via support ticket",
    },
    {
        "name": "aura-long",
        "purpose": "unlocks 7-day walltime on aura-cpu-long",
        "max_walltime": "168:00:00",
        "priority": "low",
        "limits": "2 concurrent jobs per user",
        "access": "request through support with a justification",
    },
]

# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

STORAGE = [
    {
        "path": "/home/$USER",
        "name": "home",
        "quota": "50 GB",
        "backup": "daily snapshots, kept 14 days",
        "purge": "never",
        "speed": "slow (NFS)",
        "use_for": "source code, configs, small scripts, SSH keys",
        "avoid": "job I/O, datasets, checkpoints — it is slow and small",
    },
    {
        "path": "/scratch/$USER",
        "name": "scratch",
        "quota": "5 TB",
        "backup": "none",
        "purge": "files untouched for 30 days are deleted automatically",
        "speed": "fast (parallel Lustre)",
        "use_for": "datasets, checkpoints, model weights, all job I/O",
        "avoid": "anything you cannot regenerate — there is no backup",
    },
    {
        "path": "/projects/<group>",
        "name": "project",
        "quota": "20 TB per research group",
        "backup": "weekly",
        "purge": "never, but reviewed annually",
        "speed": "medium",
        "use_for": "shared datasets and results the whole group needs",
        "avoid": "scratch-style temporary files",
    },
    {
        "path": "/apps/containers",
        "name": "container library",
        "quota": "read-only",
        "backup": "managed by staff",
        "purge": "images older than two years are retired",
        "speed": "medium",
        "use_for": "shared Apptainer .sif images maintained by AURA staff",
        "avoid": "writing — you cannot; build your own in scratch",
    },
]

# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------

MODULES = [
    {"module": "pytorch/2.4.0-cuda12.4", "what": "PyTorch 2.4.0 built against CUDA 12.4"},
    {"module": "tensorflow/2.17.0-cuda12.4", "what": "TensorFlow 2.17 with GPU support"},
    {"module": "cuda/12.4", "what": "CUDA toolkit and nvcc"},
    {"module": "apptainer/1.3.4", "what": "the Apptainer container runtime"},
    {"module": "python/3.11.7", "what": "a bare Python interpreter"},
    {"module": "gcc/13.2.0", "what": "the GNU compiler collection"},
    {"module": "openmpi/5.0.3", "what": "OpenMPI, built with UCX for InfiniBand"},
    {"module": "cudnn/9.2.0", "what": "cuDNN, loaded automatically by the framework modules"},
]

MODULE_COMMANDS = [
    ("module avail", "list every module available"),
    ("module load pytorch/2.4.0-cuda12.4", "load a specific module version"),
    ("module list", "show what is currently loaded"),
    ("module purge", "unload everything — do this at the top of every job script"),
    ("module spider pytorch", "search for a module and see its versions"),
]

# --------------------------------------------------------------------------
# AURA-specific wrapper commands (these exist nowhere else — good test cases)
# --------------------------------------------------------------------------

COMMANDS = [
    {
        "cmd": "aura-quota",
        "what": "shows your storage usage against quota for home, scratch and project space, plus your remaining allocation hours",
    },
    {
        "cmd": "aura-gpuavail",
        "what": "shows how many GPUs are free right now in each GPU partition",
    },
    {
        "cmd": "aura-jobinfo <jobid>",
        "what": "prints a readable summary of a running or finished job: node, GPU utilisation, peak memory and exit code",
    },
    {
        "cmd": "aura-whoami",
        "what": "lists the Slurm accounts and QoS levels you are permitted to use",
    },
]

# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

POLICIES = [
    {
        "topic": "login node use",
        "rule": "Never run compute on a login node. Processes using more than 10 minutes of CPU time or 8 GB of RAM on aura-login01/02 are killed automatically, and repeat offenders lose shell access for 24 hours. Use `srun` for interactive work.",
    },
    {
        "topic": "job limits",
        "rule": "Each user may have 8 GPU jobs and 50 CPU jobs running at once. There is no limit on queued jobs.",
    },
    {
        "topic": "allocations",
        "rule": "Every job must charge an account with `--account=<project>`. Check your remaining hours with `aura-quota`. Jobs are rejected once an allocation is exhausted.",
    },
    {
        "topic": "data retention",
        "rule": "Scratch files untouched for 30 days are deleted without warning, and there is no backup. Move anything you need to /projects.",
    },
    {
        "topic": "maintenance",
        "rule": "AURA has a scheduled maintenance window on the first Tuesday of each month, 08:00-16:00. Jobs that would not finish before the window will not start.",
    },
    {
        "topic": "support",
        "rule": f"Email {CLUSTER['support_email']}. Always include your username, the job ID, the partition, the full path to your job script, and the exact error message. Tickets without a job ID take much longer to resolve.",
    },
    {
        "topic": "acknowledgement",
        "rule": "Papers using AURA must acknowledge it: 'This work used the AURA Research Computing Cluster.' Send the citation to support so it can be tracked.",
    },
]

# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------

CONTAINERS = {
    "runtime": "Apptainer 1.3.4",
    "library_path": "/apps/containers",
    "images": [
        {"file": "pytorch-24.06.sif", "what": "NGC PyTorch 24.06, CUDA 12.4"},
        {"file": "tensorflow-24.06.sif", "what": "NGC TensorFlow 24.06"},
        {"file": "vllm-0.6.3.sif", "what": "vLLM inference server"},
        {"file": "rapids-24.08.sif", "what": "RAPIDS cuDF and cuML"},
    ],
    "gpu_flag": "--nv",
    "bind_note": "/scratch and /projects are bound automatically; anything else needs an explicit --bind.",
    "build_note": f"You cannot build with --fakeroot on login nodes. Build on {CLUSTER['build_node']}, or build a sandbox in /scratch and convert it.",
}

# --------------------------------------------------------------------------
# Troubleshooting
# --------------------------------------------------------------------------

TROUBLESHOOTING = [
    {
        "symptom": "job stuck in PD (pending) state",
        "cause": "Slurm is telling you why in the REASON column — you just have to look.",
        "fix": "Run `squeue -j <jobid> --long` and read the reason. `Priority` or `Resources` means wait. `QOSMaxJobsPerUserLimit` means you are at your concurrent job cap. `AssocGrpBillingMinutes` means your allocation is exhausted. `ReqNodeNotAvail` usually means a maintenance window is coming. Use `squeue -j <jobid> --start` for an estimated start time.",
    },
    {
        "symptom": "CUDA out of memory",
        "cause": "the model, batch and activations do not fit in the GPU",
        "fix": "Lower the batch size, enable gradient checkpointing, or move to a partition with more GPUs per node. Check what actually happened with `aura-jobinfo <jobid>`, which reports peak GPU memory.",
    },
    {
        "symptom": "job killed with 'Exceeded job memory limit'",
        "cause": "host RAM, not GPU memory — your --mem request was too low",
        "fix": "Check peak usage with `sacct -j <jobid> --format=JobID,MaxRSS,ReqMem` and raise --mem. Remember --mem is per node.",
    },
    {
        "symptom": "'command not found' inside a job that works on the login node",
        "cause": "job scripts do not inherit your interactive module environment",
        "fix": "Add `module purge` and then the explicit `module load` lines at the top of the sbatch script. Never rely on what happens to be loaded in your shell.",
    },
    {
        "symptom": "job exits instantly with no output",
        "cause": "usually a bad #SBATCH directive or an unwritable output path",
        "fix": "Check the file named in --output actually points at a directory you can write, and run `sacct -j <jobid> --format=JobID,State,ExitCode`. A directive typo makes Slurm reject the script silently.",
    },
    {
        "symptom": "GPU shows 0% utilisation",
        "cause": "the code is running on CPU, or is starved by the data loader",
        "fix": "Confirm the framework sees the GPU (`torch.cuda.is_available()`), make sure you requested one with --gres=gpu:N, and raise --cpus-per-task so the data loader can keep up. `aura-jobinfo <jobid>` shows the utilisation trace.",
    },
    {
        "symptom": "'Disk quota exceeded' writing to home",
        "cause": "home is only 50 GB and is not for job output",
        "fix": "Run `aura-quota` to see what is full, and move datasets and checkpoints to /scratch/$USER. Point HF_HOME and any cache directories at scratch too.",
    },
    {
        "symptom": "Apptainer cannot see the GPU",
        "cause": "the --nv flag was omitted",
        "fix": "Add --nv to `apptainer exec` or `apptainer run`. Without it the container has no NVIDIA driver bind and CUDA is invisible.",
    },
]

# --------------------------------------------------------------------------
# Canonical job script
# --------------------------------------------------------------------------

def sbatch_script(
    job_name: str = "train",
    partition: str = "aura-gpu-a100",
    gpus: int = 1,
    walltime: str = "02:00:00",
    cpus: int = 8,
    mem: str = "64G",
    account: str = "<project>",
    qos: str | None = None,
    payload: str = "python train.py",
    modules: tuple[str, ...] = ("pytorch/2.4.0-cuda12.4",),
) -> str:
    """The house-style AURA job script. One template, used everywhere."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
    ]
    if qos:
        lines.append(f"#SBATCH --qos={qos}")
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    lines += [
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={walltime}",
        "#SBATCH --output=/scratch/$USER/logs/%x-%j.out",
        "#SBATCH --error=/scratch/$USER/logs/%x-%j.err",
        "",
        "module purge",
    ]
    lines += [f"module load {m}" for m in modules]
    lines += [
        "",
        "cd /scratch/$USER/project",
        payload,
    ]
    return "\n".join(lines)


ALL_PARTITION_NAMES = [p["name"] for p in PARTITIONS]
ALL_QOS_NAMES = [q["name"] for q in QOS]

if __name__ == "__main__":
    print(f"{CLUSTER['full_name']}")
    print(f"  {len(PARTITIONS)} partitions: {', '.join(ALL_PARTITION_NAMES)}")
    print(f"  {len(QOS)} QoS levels:  {', '.join(ALL_QOS_NAMES)}")
    print(f"  {len(STORAGE)} storage areas, {len(MODULES)} modules")
    print(f"  {len(POLICIES)} policies, {len(TROUBLESHOOTING)} troubleshooting entries")
    print()
    print(sbatch_script())

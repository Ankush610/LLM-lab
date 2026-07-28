"""GPU capability detection — Lab 0.

Prints what hardware you actually have and derives the settings that depend on
it. Run this first, on every platform. It is also the module the notebooks
import to pick a dtype, so nothing downstream hardcodes fp16 or bf16.

Standalone:  python common/gpu_check.py
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    available: bool
    count: int = 0
    name: str = "none"
    capability: tuple[int, int] = (0, 0)
    total_vram_gb: float = 0.0
    supports_bf16: bool = False
    supports_flash_attn2: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def sm(self) -> str:
        return f"sm{self.capability[0]}{self.capability[1]}"

    @property
    def dtype_name(self) -> str:
        """The dtype every notebook should train and infer in."""
        return "bfloat16" if self.supports_bf16 else "float16"

    @property
    def vllm_dtype_flag(self) -> str:
        """What to pass to `vllm serve --dtype`."""
        return "bfloat16" if self.supports_bf16 else "half"


def detect() -> GPUInfo:
    try:
        import torch
    except ImportError:
        return GPUInfo(available=False, notes=["torch is not installed"])

    if not torch.cuda.is_available():
        return GPUInfo(
            available=False,
            notes=[
                "No CUDA device visible.",
                "On Kaggle: Settings -> Accelerator -> GPU T4 x2, then restart.",
            ],
        )

    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    info = GPUInfo(
        available=True,
        count=torch.cuda.device_count(),
        name=props.name,
        capability=cap,
        total_vram_gb=props.total_memory / 1024**3,
        # bf16 needs Ampere (sm80) or newer. T4 is sm75 and cannot do it.
        supports_bf16=torch.cuda.is_bf16_supported(),
        supports_flash_attn2=cap[0] >= 8,
    )

    if not info.supports_bf16:
        info.notes.append(
            f"{info.sm} predates Ampere: no bf16, no Flash Attention 2. "
            "Using fp16 everywhere — this is expected on a T4, not a problem."
        )
    if info.count > 1:
        info.notes.append(
            f"{info.count} GPUs visible. Unsloth trains on one; the spare is "
            "useful for running a server beside a loaded model."
        )
    return info


def dtype():
    """The torch dtype to use. Import this instead of hardcoding."""
    import torch
    return torch.bfloat16 if detect().supports_bf16 else torch.float16


def disk_free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1024**3


def report() -> GPUInfo:
    """Print the green-light banner. Every notebook opens with this."""
    info = detect()
    line = "=" * 62

    print(line)
    print("  AURA WORKSHOP · ENVIRONMENT CHECK")
    print(line)

    if not info.available:
        print("  GPU              NOT AVAILABLE")
        for note in info.notes:
            print(f"    ! {note}")
        print(line)
        return info

    print(f"  GPU              {info.name} x{info.count}")
    print(f"  Compute cap.     {info.sm}")
    print(f"  VRAM             {info.total_vram_gb:.1f} GB")
    print(f"  bfloat16         {'yes' if info.supports_bf16 else 'no'}")
    print(f"  FlashAttention2  {'yes' if info.supports_flash_attn2 else 'no'}")
    print(f"  -> training dtype  {info.dtype_name}")
    print(f"  -> vllm --dtype    {info.vllm_dtype_flag}")

    try:
        import torch
        print(f"  torch            {torch.__version__} (CUDA {torch.version.cuda})")
    except ImportError:
        pass

    print(f"  Free disk        {disk_free_gb():.1f} GB")
    if disk_free_gb() < 15:
        info.notes.append(
            "Under 15 GB free. Lab 5 merges a 6.4 GB model and then converts "
            "it — clear space or skip the merged save."
        )

    for note in info.notes:
        print(f"    note: {note}")

    print(line)
    print("  Ready. Continue to Lab 1." if info.available else "  Not ready.")
    print(line)
    return info


if __name__ == "__main__":
    sys.exit(0 if report().available else 1)

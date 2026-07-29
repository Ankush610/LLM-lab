"""Honest parameter counting for 4-bit models — used by Labs 1 and 4.

The obvious `sum(p.numel() for p in model.parameters())` lies about a bnb-4bit
checkpoint. bitsandbytes packs two 4-bit values into every uint8, so a
`Params4bit` tensor reports exactly half the weights it actually holds, and a
3.21 B model prints as 1.80 B. Embeddings and layer norms stay in higher
precision, so those rows come out right — which makes the bug look like a real
finding about the model rather than a counting error.

Import `count_params` instead of summing `numel()` by hand.
"""

from __future__ import annotations

import torch


def real_numel(p) -> int:
    """Parameters a tensor represents, undoing bitsandbytes 4-bit packing."""
    # A quantized weight carries the pre-packing shape in its quant_state.
    qs = getattr(p, "quant_state", None)
    shape = getattr(qs, "shape", None)
    if shape is not None:
        return int(torch.Size(shape).numel())

    # Params4bit before quant_state is attached: still two values per byte.
    # Int8Params is also uint8 but stores one value per byte, hence the
    # class-name check rather than a plain dtype check.
    if type(p).__name__ == "Params4bit" and p.dtype == torch.uint8:
        return p.numel() * 2

    return p.numel()


def count_params(model) -> tuple[int, int]:
    """(total, trainable) real parameter counts.

    LoRA adapters are never quantized, so `trainable` is a plain count — but
    `total` needs the unpacking or the trainable share comes out twice too big.
    """
    total = trainable = 0
    for p in model.parameters():
        n = real_numel(p)
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable

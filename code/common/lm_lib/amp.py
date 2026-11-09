"""Mixed precision that works on any GPU: bf16 where supported (A100/H100/L4), otherwise fp16 with a GradScaler
(Kaggle T4 / P100, Colab T4). On CPU everything runs in fp32."""
import contextlib

import torch


def amp_dtype(device):
    if device != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device):
    dt = amp_dtype(device)
    return torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()


def grad_scaler(device):
    """Loss scaling is only needed for fp16 (bf16 has fp32's exponent range)."""
    return torch.amp.GradScaler("cuda", enabled=amp_dtype(device) == torch.float16)


def describe(device):
    dt = amp_dtype(device)
    if dt is None:
        return "fp32 (CPU)"
    name = torch.cuda.get_device_name(0)
    return f"{name}: {'bf16' if dt == torch.bfloat16 else 'fp16 + GradScaler (no bf16 on this GPU)'}"

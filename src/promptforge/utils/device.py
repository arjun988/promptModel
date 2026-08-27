from __future__ import annotations

import os

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA when available (GPU-first). Falls back to CPU only if needed."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device | None = None) -> str:
    device = device or get_device()
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        return f"cuda:{idx} ({name}, {mem:.1f} GB)"
    return str(device)


def set_seed(seed: int = 42) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_dtype(prefer_fp16: bool = True, prefer_bf16: bool = False) -> torch.dtype | None:
    """Preferred mixed-precision dtype on GPU."""
    if not torch.cuda.is_available():
        return None
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if prefer_fp16:
        return torch.float16
    return None


def training_precision_flags(
    prefer_gpu: bool = True,
    use_fp16: bool = True,
    use_bf16: bool = False,
) -> dict[str, bool]:
    """HF TrainingArguments fp16/bf16 flags — GPU only."""
    on_gpu = prefer_gpu and torch.cuda.is_available()
    if not on_gpu:
        return {"fp16": False, "bf16": False}
    if use_bf16 and torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": bool(use_fp16), "bf16": False}


def require_gpu(message: str | None = None) -> None:
    """Optionally enforce GPU for heavy training jobs via PROMPTFORGE_REQUIRE_GPU=1."""
    if os.environ.get("PROMPTFORGE_REQUIRE_GPU", "0") != "1":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            message
            or "CUDA GPU required (PROMPTFORGE_REQUIRE_GPU=1) but none is available."
        )

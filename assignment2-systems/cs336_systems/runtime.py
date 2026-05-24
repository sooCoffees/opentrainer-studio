from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    bf16_supported: bool
    tf32_supported: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def get_device_info(device: str = "auto") -> DeviceInfo:
    resolved = resolve_device(device)
    cuda_available = torch.cuda.is_available()
    cuda_name = None
    cuda_capability = None
    bf16_supported = False
    tf32_supported = False
    if cuda_available:
        idx = torch.cuda.current_device()
        cuda_name = torch.cuda.get_device_name(idx)
        cuda_capability = torch.cuda.get_device_capability(idx)
        bf16_supported = torch.cuda.is_bf16_supported()
        tf32_supported = cuda_capability[0] >= 8
    return DeviceInfo(
        device=resolved,
        cuda_available=cuda_available,
        cuda_device_name=cuda_name,
        cuda_capability=cuda_capability,
        bf16_supported=bf16_supported,
        tf32_supported=tf32_supported,
    )


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_amp_dtype(device: str, amp_dtype: str | None = "auto") -> torch.dtype | None:
    if amp_dtype is None or amp_dtype == "none" or device == "cpu":
        return None
    if amp_dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    if amp_dtype == "fp16":
        return torch.float16
    raise ValueError("amp_dtype must be one of: auto, bf16, fp16, none")


def configure_torch_backends(
    *,
    allow_tf32: bool = True,
    deterministic: bool = False,
    benchmark: bool = True,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = benchmark and not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=not deterministic)


def cuda_memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
        "max_reserved": torch.cuda.max_memory_reserved(),
    }

from __future__ import annotations

import json

import torch

from cs336_systems.runtime import configure_torch_backends, get_device_info


def main() -> None:
    configure_torch_backends()
    payload = {
        "torch": torch.__version__,
        "device": get_device_info().to_dict(),
        "cuda_flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "cuda_mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cuda_math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

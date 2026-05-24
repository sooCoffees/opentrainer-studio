from __future__ import annotations

import argparse
import json

import torch

from cs336_systems.attention import attention_dispatch
from cs336_systems.profiling import benchmark_callable
from cs336_systems.runtime import configure_torch_backends, get_device_info, resolve_amp_dtype, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", default="auto", choices=["auto", "bf16", "fp16", "none"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--backend", default="sdpa", choices=["sdpa", "pytorch_flash", "triton_flash"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    configure_torch_backends()
    device = resolve_device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    q = torch.randn(args.batch_size, args.heads, args.seq_len, args.head_dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def run():
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            return attention_dispatch(q, k, v, is_causal=args.causal, backend=args.backend)

    fn = torch.compile(run) if args.compile else run
    result = benchmark_callable(
        f"attention:{args.backend}",
        fn,
        warmup=args.warmup,
        repeats=args.repeats,
        tokens_per_iter=args.batch_size * args.seq_len,
    )
    print(json.dumps({"device": get_device_info(device).to_dict(), "result": result.to_dict()}, indent=2))


if __name__ == "__main__":
    main()

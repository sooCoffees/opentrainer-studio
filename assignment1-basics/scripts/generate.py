from __future__ import annotations

import argparse

import torch

from cs336_basics.io import load_tokenizer
from cs336_basics.model import TransformerConfig, TransformerLM
from cs336_basics.training import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = TransformerConfig(**checkpoint["model_config"])
    model = TransformerLM(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    tokenizer = load_tokenizer(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()

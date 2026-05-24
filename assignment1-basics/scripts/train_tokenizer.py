from __future__ import annotations

import argparse

from cs336_basics.implementation import BPETokenizer, train_bpe
from cs336_basics.io import save_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    tokenizer = BPETokenizer(vocab, merges, args.special_token)
    save_tokenizer(tokenizer, args.out_dir)
    print(f"saved tokenizer with {len(vocab)} tokens to {args.out_dir}")


if __name__ == "__main__":
    main()

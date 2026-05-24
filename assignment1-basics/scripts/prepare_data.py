from __future__ import annotations

import argparse

from cs336_basics.io import load_tokenizer, write_token_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    with open(args.input, encoding="utf-8") as f:
        token_ids = list(tokenizer.encode_iterable(f))
    write_token_file(token_ids, args.output)
    print(f"wrote {len(token_ids)} tokens to {args.output}")


if __name__ == "__main__":
    main()

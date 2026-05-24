# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Local training workflow

This fork also includes a small training pipeline for self-study experiments. The
official adapter tests still run through `tests/adapters.py`; the reusable project
code lives under `cs336_basics/`, and runnable entry points live under `scripts/`.

Train a tokenizer:

```sh
uv run python -m scripts.train_tokenizer \
  --input tests/fixtures/corpus.en \
  --out-dir artifacts/debug/tokenizer \
  --vocab-size 500
```

Prepare token files:

```sh
uv run python -m scripts.prepare_data \
  --tokenizer artifacts/debug/tokenizer \
  --input tests/fixtures/corpus.en \
  --output artifacts/debug/train.bin

uv run python -m scripts.prepare_data \
  --tokenizer artifacts/debug/tokenizer \
  --input tests/fixtures/corpus.en \
  --output artifacts/debug/valid.bin
```

Run a tiny training smoke test:

```sh
uv run python -m scripts.train_lm --config configs/debug_fixture.json
```

Generate from the checkpoint:

```sh
uv run python -m scripts.generate \
  --checkpoint runs/debug_fixture/checkpoint.pt \
  --tokenizer artifacts/debug/tokenizer \
  --prompt "The " \
  --max-new-tokens 50
```

For real experiments, replace the fixture paths in a copied config with tokenized
TinyStories or your own corpus, then increase model size and `max_iters`.

The training stack includes a few systems-oriented switches:

- `attention_backend: "sdpa"` uses PyTorch 2.x scaled dot-product attention,
  which dispatches to FlashAttention-style kernels on supported CUDA hardware.
- `amp_dtype: "auto"` enables autocast mixed precision on CUDA/MPS and keeps CPU
  runs in fp32.
- `optimizer: "fused_adamw"` uses PyTorch fused AdamW when CUDA is available and
  falls back cleanly elsewhere.
- `compile: true` enables `torch.compile` for longer GPU runs where compile
  overhead is worth paying.

See `configs/tinystories_optimized.json` for a stronger TinyStories-oriented
starting point.

Once a CUDA GPU is available, profile one training step with:

```sh
uv run python -m scripts.profile_train_step \
  --config configs/tinystories_optimized.json \
  --steps 20
```

Use that before and after changing attention, precision, batch size, or
`torch.compile` so speedups are measured instead of guessed.

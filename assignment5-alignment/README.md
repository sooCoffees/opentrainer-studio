# CS336 Spring 2026 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)

We will include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run the required unit tests:

``` sh
uv run pytest tests/test_grpo.py
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

## Local Implementation Notes

This workspace adds `cs336_alignment.core` and wires the local adapters to it.
Implemented pieces include prompt/output tokenization, response log-probs,
reward aggregation, GRPO/DAPO-style advantage normalization, policy-gradient
losses, SFT microbatching, packed SFT data, simple MMLU/GSM8K parsing, and DPO
loss computation.

Current local CPU status: the focused alignment suite passes 32 out of 33 tests.
The remaining mismatch is the scalar convention in
`tests/test_dpo.py::test_per_instance_dpo_loss`; the current implementation uses
the standard summed log-ratio DPO form, while the fixture expects a different
normalization.

Next improvements should keep the same interfaces and add TRL-compatible SFT,
DPO, and GRPO adapters; bf16 training; reward/advantage logging; vLLM rollout
generation; and optional LoRA/QLoRA fine-tuning.

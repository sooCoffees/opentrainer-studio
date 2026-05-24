# OpenTrainer Studio Demo Notes

OpenTrainer Studio helps a user create a small AI project without starting from
command-line training scripts.

The normal workflow is:

1. Create an AI with a clear name and goal.
2. Add training materials to that AI's own workspace.
3. Choose a training computer.
4. Run a tiny test to confirm the setup works.
5. Review loss and speed metrics.
6. Scale up only after the small run succeeds.

## Example AI

Name: Study Assistant

Goal: Learn from CS336 notes, project docs, and personal research notes so it
can answer questions about tokenizer training, transformer implementation,
training loops, scaling experiments, and data quality.

## Useful Concepts

Tokenizer training turns raw text into token ids. A transformer predicts the
next token. A training loop updates model weights by minimizing cross entropy
loss. Data quality matters because noisy or private data can reduce model
quality and increase risk.

## Testing Rule

Always run a tiny test before renting a stronger GPU. The tiny test is not meant
to create a smart model. It is meant to prove that the profile, data path,
training config, metric logging, and training computer are connected correctly.

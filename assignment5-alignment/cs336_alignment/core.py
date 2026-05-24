from __future__ import annotations

import json
import os
import random
import re
from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
):
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    sequences = []
    response_masks = []
    for prompt, output in zip(prompt_strs, output_strs, strict=True):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        ids = prompt_ids + output_ids
        mask = [0] * max(0, len(prompt_ids) - 1) + [1] * len(output_ids)
        sequences.append(ids)
        response_masks.append(mask)
    max_len = max(len(ids) for ids in sequences) - 1
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, ids in enumerate(sequences):
        seq_input = ids[:max_len]
        seq_labels = ids[1 : max_len + 1]
        input_length = len(seq_input)
        label_length = len(seq_labels)
        input_ids[row, :input_length] = torch.tensor(seq_input, dtype=torch.long)
        labels[row, :label_length] = torch.tensor(seq_labels, dtype=torch.long)
        response_mask[row, : len(response_masks[row])] = torch.tensor(
            response_masks[row], dtype=torch.bool
        )
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool,
):
    logits = (
        model(input_ids).logits
        if hasattr(model(input_ids), "logits")
        else model(input_ids)
    )
    log_probs_all = F.log_softmax(logits, dim=-1)
    log_probs = log_probs_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out = {"log_probs": log_probs}
    if return_token_entropy:
        probs = log_probs_all.exp()
        out["token_entropy"] = -(probs * log_probs_all).sum(dim=-1)
    return out


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[Tensor, dict[str, float]]:
    rows = [
        reward_fn(response, gt)
        for response, gt in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    rewards = torch.tensor([row["reward"] for row in rows], dtype=torch.float32)
    metadata = {}
    for key in rows[0].keys():
        metadata[f"{key}_mean"] = float(sum(row[key] for row in rows) / len(rows))
    return rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[Tensor, Tensor, dict[str, float]]:
    groups = raw_rewards.view(-1, group_size)
    advantages = groups.clone()
    if baseline == "mean":
        advantages = advantages - groups.mean(dim=-1, keepdim=True)
    if advantage_normalizer == "std":
        advantages = advantages / (
            groups.std(dim=-1, keepdim=True, unbiased=True) + advantage_eps
        )
    elif advantage_normalizer == "mean":
        advantages = advantages / (
            groups.mean(dim=-1, keepdim=True).abs() + advantage_eps
        )
    elif advantage_normalizer != "none":
        raise ValueError("unknown advantage normalizer")
    flat = advantages.reshape_as(raw_rewards)
    return (
        flat,
        raw_rewards,
        {
            "reward_mean": float(raw_rewards.mean().item()),
            "reward_std": float(raw_rewards.std(unbiased=False).item()),
            "advantage_mean": float(flat.mean().item()),
        },
    )


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    advantages = raw_rewards_or_advantages.reshape(-1, 1).to(policy_log_probs.device)
    metadata: dict[str, Tensor] = {}
    weight = torch.ones_like(policy_log_probs)
    if importance_reweighting_method != "none":
        if old_log_probs is None:
            raise ValueError("old_log_probs required")
        log_ratio = policy_log_probs - old_log_probs.to(policy_log_probs.device)
        if importance_reweighting_method == "gspo":
            if response_mask is None:
                raise ValueError("response_mask required for gspo")
            denom = response_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            seq_log_ratio = (log_ratio * response_mask).sum(
                dim=-1, keepdim=True
            ) / denom
            ratio = seq_log_ratio.exp()
        else:
            ratio = log_ratio.exp()
        if importance_reweighting_method in {"grpo", "gspo"}:
            if cliprange is None:
                raise ValueError("cliprange required")
            clipped = ratio.clamp(1 - cliprange, 1 + cliprange)
            weight = torch.where(
                advantages >= 0,
                torch.minimum(ratio, clipped),
                torch.maximum(ratio, clipped),
            )
            metadata["clip_fraction"] = (ratio.ne(clipped)).float().mean()
        else:
            weight = ratio
        return -(weight * advantages).expand_as(policy_log_probs), metadata
    return -(weight * advantages * policy_log_probs), metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> Tensor:
    masked = per_token_policy_gradient_loss * mask
    if loss_normalization == "sequence":
        return (masked.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)).mean()
    if normalization_constant is None:
        raise ValueError("normalization_constant required")
    return masked.sum() / normalization_constant


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    denom = float(normalize_constant or response_mask.sum().clamp_min(1).item())
    loss = -masked_normalize(
        policy_log_probs, response_mask, normalize_constant=denom
    ) / (gradient_accumulation_steps**2)
    loss.backward()
    return loss.detach(), {"sft_loss": loss.detach()}


class PackedSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | os.PathLike,
        seq_length: int,
        shuffle: bool,
    ):
        docs = []
        with open(dataset_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = _format_alpaca_sft(
                    row.get("prompt", ""), row.get("response", row.get("output", ""))
                )
                docs.append(text)
        if shuffle:
            random.Random(1337).shuffle(docs)
        ids = []
        eos = tokenizer.eos_token_id
        for doc in docs:
            ids.extend(tokenizer.encode(doc, add_special_tokens=True))
            if eos is not None and (not ids or ids[-1] != eos):
                ids.append(eos)
        self.examples = []
        for start in range(0, max(0, len(ids) - seq_length), seq_length):
            chunk = ids[start : start + seq_length + 1]
            if len(chunk) == seq_length + 1:
                self.examples.append(
                    {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def _format_alpaca_sft(prompt: str, response: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )


def iterate_batches(dataset: Dataset, batch_size: int, shuffle: bool):
    indices = list(range(len(dataset)))
    if shuffle:
        random.Random(1337).shuffle(indices)
    batches = []
    for start in range(0, len(indices), batch_size):
        rows = [dataset[i] for i in indices[start : start + batch_size]]
        batches.append(
            {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        )
    return batches


def parse_mmlu_response(_mmlu_example: dict, model_output: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", model_output.upper())
    return match.group(1) if match else None


def parse_gsm8k_response(model_output: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", model_output)
    return matches[-1].replace(",", "") if matches else None


def compute_per_instance_dpo_loss(
    lm,
    lm_ref,
    tokenizer,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
):
    chosen = tokenize_prompt_and_output([prompt], [response_chosen], tokenizer)
    rejected = tokenize_prompt_and_output([prompt], [response_rejected], tokenizer)
    chosen_lp = get_response_log_probs(
        lm, chosen["input_ids"], chosen["labels"], False
    )["log_probs"]
    rejected_lp = get_response_log_probs(
        lm, rejected["input_ids"], rejected["labels"], False
    )["log_probs"]
    with torch.no_grad():
        ref_chosen_lp = get_response_log_probs(
            lm_ref, chosen["input_ids"], chosen["labels"], False
        )["log_probs"]
        ref_rejected_lp = get_response_log_probs(
            lm_ref, rejected["input_ids"], rejected["labels"], False
        )["log_probs"]
    chosen_score = (chosen_lp * chosen["response_mask"]).sum() - (
        ref_chosen_lp * chosen["response_mask"]
    ).sum()
    rejected_score = (rejected_lp * rejected["response_mask"]).sum() - (
        ref_rejected_lp * rejected["response_mask"]
    ).sum()
    return -F.logsigmoid(beta * (chosen_score - rejected_score))

"""Deterministic LongBench v2 prompt construction for Mixtral profiling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


DATASET_NAME = "zai-org/LongBench-v2"
DATASET_SPLIT = "train"


@dataclass(frozen=True)
class LongBenchSample:
    """The LongBench v2 fields used by the fixed multiple-choice template."""

    sample_id: str
    context: str
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str


def _sample_from_row(row: dict[str, Any]) -> LongBenchSample:
    return LongBenchSample(
        sample_id=str(row["_id"]),
        context=str(row["context"]),
        question=str(row["question"]),
        choice_a=str(row["choice_A"]),
        choice_b=str(row["choice_B"]),
        choice_c=str(row["choice_C"]),
        choice_d=str(row["choice_D"]),
    )


def _encode(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int | None = None) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=max_length is not None,
        max_length=max_length,
    )
    return list(encoded["input_ids"])


def _prompt_prefix_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    return _encode(tokenizer, "Context:\n")


def _prompt_suffix_ids(tokenizer: PreTrainedTokenizerBase, sample: LongBenchSample) -> list[int]:
    suffix = (
        "\n\nQuestion:\n"
        f"{sample.question}\n\n"
        f"A. {sample.choice_a}\n"
        f"B. {sample.choice_b}\n"
        f"C. {sample.choice_c}\n"
        f"D. {sample.choice_d}\n\n"
        "Answer:\n"
    )
    return _encode(tokenizer, suffix)


def build_prompt_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    sample: LongBenchSample,
    total_length: int,
) -> torch.Tensor:
    """Build an exact-length prompt by truncating only context token IDs.

    The static prompt prefix/suffix is tokenized once per construction. The
    context token IDs are then sliced to the remaining budget, so every shorter
    prompt uses a nested prefix of the same LongBench context.
    """
    prefix_ids = _prompt_prefix_ids(tokenizer)
    suffix_ids = _prompt_suffix_ids(tokenizer, sample)
    context_budget = total_length - len(prefix_ids) - len(suffix_ids)
    if context_budget < 1:
        raise ValueError(
            f"prompt template leaves no context tokens for sample {sample.sample_id} "
            f"at target length {total_length}"
        )
    context_ids = _encode(tokenizer, sample.context, max_length=context_budget)
    if len(context_ids) != context_budget:
        raise ValueError(
            f"LongBench sample {sample.sample_id} has only {len(context_ids)} context tokens, "
            f"needs {context_budget}"
        )
    input_ids = torch.tensor([prefix_ids + context_ids + suffix_ids], dtype=torch.long)
    if input_ids.shape[-1] != total_length:
        raise AssertionError(
            f"constructed prompt length {input_ids.shape[-1]} does not equal {total_length}"
        )
    return input_ids


def load_deterministic_samples(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sample_seed: int,
    required_prompt_length: int,
    dataset_name: str = DATASET_NAME,
) -> list[LongBenchSample]:
    """Select seed-stable samples that can construct the longest prompt."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least one")
    dataset = load_dataset(dataset_name, split=DATASET_SPLIT)
    indices = list(range(len(dataset)))
    random.Random(sample_seed).shuffle(indices)

    selected: list[LongBenchSample] = []
    for index in indices:
        sample = _sample_from_row(dataset[index])
        try:
            build_prompt_input_ids(tokenizer, sample, required_prompt_length)
        except ValueError:
            continue
        selected.append(sample)
        if len(selected) == num_samples:
            return selected
    raise ValueError(
        f"only found {len(selected)} LongBench samples with at least "
        f"{required_prompt_length} prompt tokens; requested {num_samples}"
    )

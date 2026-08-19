"""Deterministic long contiguous token spans from streaming PG-19."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


DEFAULT_DATASET = "emozilla/pg19"


@dataclass(frozen=True)
class PG19Sample:
    sample_id: str
    input_ids: torch.Tensor


def _text_from_row(row: dict[str, object]) -> str:
    for field in ("text", "content"):
        value = row.get(field)
        if isinstance(value, str):
            return value
    raise KeyError(f"PG-19 row has no supported text field; found {sorted(row)}")


def load_deterministic_pg19_samples(
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_samples: int,
    sample_seed: int,
    required_tokens: int,
    dataset_name: str = DEFAULT_DATASET,
    split: str = "train",
    max_scan: int = 1_024,
) -> list[PG19Sample]:
    """Select seed-stable contiguous spans suitable for teacher-forced LM.

    The dataset is streamed, shuffled with the requested seed, and only enough
    rows are inspected to find long documents. Each selected document gets a
    deterministic in-document start offset, so repeated runs use identical
    token IDs without downloading the complete corpus.
    """
    if num_samples < 1 or required_tokens < 2 or max_scan < 1:
        raise ValueError("num_samples, required_tokens, and max_scan must be positive")
    dataset = load_dataset(dataset_name, split=split, streaming=True).shuffle(
        seed=sample_seed,
        buffer_size=1_024,
    )
    selected: list[PG19Sample] = []
    for row_index, row in enumerate(dataset):
        if row_index >= max_scan:
            break
        text = _text_from_row(row)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) < required_tokens:
            continue
        max_start = len(token_ids) - required_tokens
        start = random.Random(f"pg19:{sample_seed}:{row_index}").randint(0, max_start)
        sample_id = str(row.get("id", f"pg19-{row_index}"))
        selected.append(
            PG19Sample(
                sample_id=f"{sample_id}@{start}",
                input_ids=torch.tensor([token_ids[start : start + required_tokens]], dtype=torch.long),
            )
        )
        if len(selected) == num_samples:
            return selected
    raise ValueError(
        f"found only {len(selected)} PG-19 samples with at least {required_tokens} tokens "
        f"after scanning {max_scan} streamed rows"
    )

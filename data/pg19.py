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
    decode_tokens: int


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
    prefill_tokens: int,
    max_decode_tokens: int,
    dataset_name: str = DEFAULT_DATASET,
    split: str = "test",
    max_scan: int = 1_024,
    allow_shorter_decode: bool = True,
) -> list[PG19Sample]:
    """Select seed-stable PG-19 spans suitable for teacher-forced LM.

    Documents are streamed and shuffled with the requested seed.  When
    ``allow_shorter_decode`` is true, every document with a valid prefix is
    retained and its continuation is capped by its remaining length. This lets
    corpus PPL cover all 100 PG-19 test books without inventing padding or
    crossing document boundaries. Long documents still receive exactly
    ``max_decode_tokens`` scored tokens.
    """
    if num_samples < 1 or prefill_tokens < 1 or max_decode_tokens < 1 or max_scan < 1:
        raise ValueError("num_samples, prefill_tokens, max_decode_tokens, and max_scan must be positive")
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
        minimum_tokens = prefill_tokens + 2  # prefix, one input, and one target
        if len(token_ids) < minimum_tokens:
            continue
        available_decode_tokens = len(token_ids) - prefill_tokens - 1
        if not allow_shorter_decode and available_decode_tokens < max_decode_tokens:
            continue
        decode_tokens = min(max_decode_tokens, available_decode_tokens)
        span_tokens = prefill_tokens + decode_tokens + 1
        max_start = len(token_ids) - span_tokens
        start = random.Random(f"pg19:{sample_seed}:{row_index}").randint(0, max_start)
        sample_id = str(row.get("id", f"pg19-{row_index}"))
        selected.append(
            PG19Sample(
                sample_id=f"{sample_id}@{start}",
                input_ids=torch.tensor([token_ids[start : start + span_tokens]], dtype=torch.long),
                decode_tokens=decode_tokens,
            )
        )
        if len(selected) == num_samples:
            return selected
    raise ValueError(
        f"found only {len(selected)} PG-19 samples with at least {minimum_tokens} tokens after scanning {max_scan} "
        f"streamed rows; requested {num_samples}"
    )

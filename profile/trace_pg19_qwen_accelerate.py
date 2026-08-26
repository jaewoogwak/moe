#!/usr/bin/env python3
"""Trace Qwen1.5-MoE routing during greedy PG19 decoding.

Run this script with ``accelerate launch --num_processes 1``.  The original
BF16 model is dispatched with Accelerate's ``device_map='auto'``: CUDA holds
as much of the model as fits under ``--gpu-memory`` and the remaining weights
stay on CPU.  The CSV deliberately excludes prefill routing; every row is one
true cached, one-token decode forward.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


DEFAULT_MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_DATASET = "emozilla/pg19"


class RouterTracer:
    """Capture the Qwen router's selected top-k experts for every MoE layer."""

    def __init__(self, model: Any, top_k: int) -> None:
        self._top_k = top_k
        self._pending: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._handles: list[Any] = []
        for layer_id, layer in enumerate(model.model.layers):
            if not hasattr(layer.mlp, "gate"):
                raise TypeError(f"Layer {layer_id} does not expose a Qwen MoE router gate.")
            self._handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(layer_id)))

    def _make_hook(self, layer_id: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], gate_output: Any) -> None:
            if layer_id in self._pending:
                raise RuntimeError(f"Router hook for layer {layer_id} ran more than once in one forward pass.")

            # Transformers 5 Qwen2MoeTopKRouter returns
            # (router_logits, normalized_top_k_weights, top_k_indices).  Keep a
            # logits fallback for compatible earlier implementations.
            if isinstance(gate_output, tuple) and len(gate_output) >= 3:
                router_logits, routing_weights, expert_indices = gate_output[:3]
            else:
                router_logits = gate_output
                if not isinstance(router_logits, torch.Tensor):
                    raise TypeError(f"Layer {layer_id} router did not return a tensor.")
                probabilities = F.softmax(router_logits.float(), dim=-1)
                routing_weights, expert_indices = torch.topk(probabilities, k=self._top_k, dim=-1)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

            if not isinstance(router_logits, torch.Tensor):
                raise TypeError(f"Layer {layer_id} router logits are not a tensor.")
            if router_logits.ndim != 2:
                router_logits = router_logits.reshape(-1, router_logits.shape[-1])
            if expert_indices.shape[-1] != self._top_k:
                raise RuntimeError(
                    f"Layer {layer_id} returned {expert_indices.shape[-1]} experts; expected {self._top_k}."
                )
            probabilities = F.softmax(router_logits.float(), dim=-1)
            selected_probabilities = probabilities.gather(dim=-1, index=expert_indices)
            self._pending[layer_id] = (
                expert_indices.detach().to(device="cpu", dtype=torch.int16),
                routing_weights.detach().to(device="cpu", dtype=torch.float32),
                selected_probabilities.detach().to(device="cpu", dtype=torch.float32),
            )

        return hook

    def begin(self) -> None:
        if self._pending:
            raise RuntimeError("Unconsumed router trace from a previous model forward.")

    def consume(self, sequence_length: int) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        expected = set(range(len(self._handles)))
        if set(self._pending) != expected:
            raise RuntimeError(f"Expected router traces for layers {sorted(expected)}, got {sorted(self._pending)}.")
        traces = self._pending
        self._pending = {}
        for layer_id, (experts, weights, probabilities) in traces.items():
            expected_shape = (sequence_length, self._top_k)
            if experts.shape != expected_shape or weights.shape != expected_shape or probabilities.shape != expected_shape:
                raise RuntimeError(
                    f"Layer {layer_id} trace shapes are {tuple(experts.shape)}, {tuple(weights.shape)}, "
                    f"{tuple(probabilities.shape)}; expected {expected_shape}."
                )
        return traces

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub ID or local path for the base Qwen MoE model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--document-index", type=int, default=0)
    parser.add_argument("--text-column", default=None, help="PG19 text column; defaults to text, then content.")
    parser.add_argument("--prefill-tokens", type=int, default=4096)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument(
        "--gpu-memory",
        default="22GiB",
        help="Maximum CUDA allocation given to Accelerate device_map=auto (default: 22GiB).",
    )
    parser.add_argument("--cpu-memory", default="900GiB", help="CPU RAM limit passed to Accelerate device mapping.")
    parser.add_argument("--output", type=Path, default=Path("results/pg19_qwen_decode_expert_trace.csv"))
    parser.add_argument("--add-special-tokens", action="store_true")
    args = parser.parse_args()
    if args.document_index < 0:
        parser.error("--document-index must be non-negative")
    if args.prefill_tokens < 1 or args.decode_tokens < 1:
        parser.error("--prefill-tokens and --decode-tokens must be positive")
    return args


def load_document(args: argparse.Namespace) -> str:
    dataset = load_dataset(args.dataset, split=args.split, streaming=True, revision=args.dataset_revision)
    row = next(itertools.islice(iter(dataset), args.document_index, args.document_index + 1), None)
    if row is None:
        raise IndexError(f"Document {args.document_index} is outside PG19 split {args.split!r}.")
    columns = (args.text_column,) if args.text_column else ("text", "content")
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"No non-empty text column found; tried {columns}, available columns: {sorted(row)}")


def token_fields(tokenizer: Any, token_id: int) -> tuple[str, str]:
    return tokenizer.convert_ids_to_tokens(token_id), tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def write_decode_rows(
    writer: csv.writer,
    tokenizer: Any,
    decode_step: int,
    input_token: torch.Tensor,
    next_token: torch.Tensor,
    traces: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    top_k: int,
) -> None:
    input_token_id = int(input_token.item())
    next_token_id = int(next_token.item())
    token_piece, token_text = token_fields(tokenizer, input_token_id)
    for layer_id, (expert_indices, routing_weights, probabilities) in traces.items():
        row: list[Any] = [decode_step, input_token_id, token_piece, token_text, next_token_id, layer_id]
        row.extend(int(expert_indices[0, rank]) for rank in range(top_k))
        row.extend(f"{float(routing_weights[0, rank]):.8f}" for rank in range(top_k))
        row.extend(f"{float(probabilities[0, rank]):.8f}" for rank in range(top_k))
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("This trace is a single-sequence experiment; launch with --num_processes 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    print("Loading PG19 document...", flush=True)
    document = load_document(args)
    print(f"Loading original BF16 Qwen model through Accelerate dispatch: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_memory, "cpu": args.cpu_memory},
        low_cpu_mem_usage=True,
    )
    model.eval()
    config = model.config
    num_layers = len(model.model.layers)
    num_experts = getattr(config, "num_experts", None)
    top_k = int(config.num_experts_per_tok)
    if num_experts is None:
        raise TypeError("This model does not expose Qwen's config.num_experts.")
    if top_k != 4:
        raise ValueError(f"Expected Qwen1.5-MoE top-4 routing, got top-{top_k}.")

    token_ids = tokenizer(document, add_special_tokens=args.add_special_tokens, return_attention_mask=False)["input_ids"]
    if len(token_ids) < args.prefill_tokens:
        raise ValueError(
            f"Document {args.document_index} has {len(token_ids)} tokens; need {args.prefill_tokens}. "
            "Choose another document index."
        )
    input_device = model.get_input_embeddings().weight.device
    prompt_ids = torch.tensor([token_ids[: args.prefill_tokens]], dtype=torch.long, device=input_device)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")

    print(f"Prefilling {prompt_ids.shape[1]} PG19 context tokens across {num_layers} MoE layers...", flush=True)
    kv_cache = DynamicCache(config=config)
    with torch.inference_mode():
        prefill_output = model(
            input_ids=prompt_ids,
            past_key_values=kv_cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    next_input_token = prefill_output.logits[:, -1:].argmax(dim=-1).to(input_device)

    tracer = RouterTracer(model, top_k=top_k)
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            header = ["decode_step", "input_token_id", "input_token_piece", "input_token_text", "next_token_id", "layer"]
            header += [f"top{rank}_expert" for rank in range(1, top_k + 1)]
            header += [f"top{rank}_routing_weight" for rank in range(1, top_k + 1)]
            header += [f"top{rank}_router_probability" for rank in range(1, top_k + 1)]
            writer.writerow(header)
            for decode_step in range(args.decode_tokens):
                tracer.begin()
                with torch.inference_mode():
                    output = model(
                        input_ids=next_input_token,
                        past_key_values=kv_cache,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                traces = tracer.consume(sequence_length=1)
                next_token = output.logits[:, -1:].argmax(dim=-1).to(input_device)
                write_decode_rows(writer, tokenizer, decode_step, next_input_token, next_token, traces, top_k)
                next_input_token = next_token
                if (decode_step + 1) % 32 == 0:
                    print(f"Traced {decode_step + 1}/{args.decode_tokens} decode steps", flush=True)
        temporary_path.replace(output_path)
    finally:
        tracer.close()

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_dtype": "bfloat16",
        "quantization": "none",
        "loader": "accelerate device_map=auto",
        "hf_device_map": getattr(model, "hf_device_map", None),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "document_index": args.document_index,
        "document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "num_layers": num_layers,
        "experts_per_layer": int(num_experts),
        "experts_per_token": top_k,
        "arguments": vars(args) | {"output": str(output_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({args.decode_tokens * num_layers:,} rows).")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

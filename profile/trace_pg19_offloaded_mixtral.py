#!/usr/bin/env python3
"""Trace per-decode-step Mixtral expert routing with CPU-offloaded experts.

The model is loaded in original BF16 precision on CPU.  Non-expert modules run
on CUDA; all 256 expert weight pairs live in pinned host memory and are staged
through the repository's fixed-per-layer GPU LRU cache on demand.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offload.expert_cache import FIXED_PER_LAYER_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import replace_with_cached_offloaded_experts


DEFAULT_MODEL = "mistralai/Mixtral-8x7B-v0.1"
DEFAULT_DATASET = "emozilla/pg19"
NUM_MIXTRAL_LAYERS = 32


class RouterTracer:
    """Capture the router's selected top-2 experts for each Mixtral layer."""

    def __init__(self, model: Any) -> None:
        self._pending: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._handles: list[Any] = []
        for layer_id, layer in enumerate(model.model.layers):
            if not hasattr(layer.mlp, "gate"):
                raise TypeError(f"Layer {layer_id} does not expose a Mixtral router gate.")
            self._handles.append(layer.mlp.gate.register_forward_hook(self._make_hook(layer_id)))

    def _make_hook(self, layer_id: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], gate_output: Any) -> None:
            if layer_id in self._pending:
                raise RuntimeError(f"Router hook for layer {layer_id} ran more than once in one forward pass.")

            # Transformers 5 returns (router_logits, normalized_top_k_weights,
            # top_k_indices). Older releases expose only logits, so retain a
            # fallback that derives the same top-2 selection from softmax.
            if isinstance(gate_output, tuple) and len(gate_output) >= 3:
                router_logits, routing_weights, expert_indices = gate_output[:3]
            else:
                router_logits = gate_output
                if not isinstance(router_logits, torch.Tensor):
                    raise TypeError(f"Layer {layer_id} router did not return a tensor.")
                probabilities = F.softmax(router_logits.float(), dim=-1)
                routing_weights, expert_indices = torch.topk(probabilities, k=2, dim=-1)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

            if not isinstance(router_logits, torch.Tensor):
                raise TypeError(f"Layer {layer_id} router logits are not a tensor.")
            if router_logits.ndim != 2:
                router_logits = router_logits.reshape(-1, router_logits.shape[-1])
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
            expected_shape = (sequence_length, 2)
            if experts.shape != expected_shape or weights.shape != expected_shape or probabilities.shape != expected_shape:
                raise RuntimeError(
                    f"Layer {layer_id} trace shapes are "
                    f"{tuple(experts.shape)}, {tuple(weights.shape)}, {tuple(probabilities.shape)}; "
                    f"expected {expected_shape}."
                )
        return traces

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hub ID or local path for the base Mixtral model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--document-index", type=int, default=0)
    parser.add_argument("--text-column", default=None, help="PG19 text column; defaults to text, then content.")
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=4096,
        help="Number of PG19 document tokens used as the fixed decoding context.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=256,
        help="Number of greedy one-token decode forwards to trace.",
    )
    parser.add_argument("--cache-slots", type=int, default=NUM_MIXTRAL_LAYERS)
    parser.add_argument("--output", type=Path, default=Path("results/pg19_expert_trace.csv"))
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens. Disabled by default to trace only document tokens.",
    )
    args = parser.parse_args()
    if args.document_index < 0:
        parser.error("--document-index must be non-negative")
    if args.prefill_tokens < 1 or args.decode_tokens < 1:
        parser.error("--prefill-tokens and --decode-tokens must be positive")
    if args.cache_slots < NUM_MIXTRAL_LAYERS:
        parser.error(f"fixed_per_layer_lru needs at least {NUM_MIXTRAL_LAYERS} cache slots")
    return args


def load_document(args: argparse.Namespace) -> str:
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
        revision=args.dataset_revision,
    )
    row = next(itertools.islice(iter(dataset), args.document_index, args.document_index + 1), None)
    if row is None:
        raise IndexError(f"Document index {args.document_index} is outside PG19 split {args.split!r}.")
    columns = (args.text_column,) if args.text_column else ("text", "content")
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"No non-empty text column found; tried {columns}, available columns: {sorted(row)}")


def move_non_expert_modules_to_cuda(model: Any) -> None:
    """Match the repository's latency runtime while retaining experts on CPU."""
    model.model.embed_tokens.to("cuda")
    model.model.norm.to("cuda")
    model.model.rotary_emb.to("cuda")
    model.lm_head.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.to("cuda")
        layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm.to("cuda")
        layer.mlp.gate.to("cuda")


def load_offloaded_model(args: argparse.Namespace) -> tuple[Any, Any, GPUExpertCache]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    print(f"Loading original BF16 model onto CPU: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    host_store = HostExpertStore(model)
    move_non_expert_modules_to_cuda(model)
    expert_cache = GPUExpertCache(
        host_store,
        capacity_slots=args.cache_slots,
        device="cuda",
        cache_policy=FIXED_PER_LAYER_LRU,
    )
    replace_with_cached_offloaded_experts(model, expert_cache)
    if not host_store.all_experts_pinned() or not host_store.pageable_expert_storage_released:
        raise AssertionError("Failed to construct the pinned CPU expert store.")
    return model, tokenizer, expert_cache


def token_fields(tokenizer: Any, token_id: int) -> tuple[str, str]:
    return (
        tokenizer.convert_ids_to_tokens(token_id),
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False),
    )


def write_decode_rows(
    writer: csv.writer,
    tokenizer: Any,
    decode_step: int,
    input_token: torch.Tensor,
    next_token: torch.Tensor,
    traces: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> None:
    input_token_id = int(input_token.item())
    next_token_id = int(next_token.item())
    token_piece, token_text = token_fields(tokenizer, input_token_id)
    for layer_id, (expert_indices, routing_weights, probabilities) in traces.items():
        writer.writerow(
            (
                decode_step,
                input_token_id,
                token_piece,
                token_text,
                next_token_id,
                layer_id,
                int(expert_indices[0, 0]),
                int(expert_indices[0, 1]),
                f"{float(routing_weights[0, 0]):.8f}",
                f"{float(routing_weights[0, 1]):.8f}",
                f"{float(probabilities[0, 0]):.8f}",
                f"{float(probabilities[0, 1]):.8f}",
            )
        )


def main() -> None:
    args = parse_args()
    print("Loading PG19 document...", flush=True)
    document = load_document(args)
    model, tokenizer, expert_cache = load_offloaded_model(args)
    token_ids = tokenizer(document, add_special_tokens=args.add_special_tokens, return_attention_mask=False)["input_ids"]
    if len(token_ids) < args.prefill_tokens:
        raise ValueError(
            f"Document {args.document_index} has {len(token_ids)} tokens; need {args.prefill_tokens}. "
            "Choose another document index."
        )
    prompt_ids = torch.tensor([token_ids[: args.prefill_tokens]], dtype=torch.long, device="cuda")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    print(
        f"Prefilling {prompt_ids.shape[1]} PG19 context tokens with "
        f"{expert_cache.capacity_slots} fixed-per-layer cache slots...",
        flush=True,
    )
    kv_cache = DynamicCache(config=model.config)
    with torch.inference_mode():
        prefill_output = model(
            input_ids=prompt_ids,
            past_key_values=kv_cache,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
    next_input_token = prefill_output.logits[:, -1:].argmax(dim=-1)
    expert_cache.reset_stats()

    # Attach after prefill so the CSV contains only true one-token decode calls.
    tracer = RouterTracer(model)
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "decode_step",
                    "input_token_id",
                    "input_token_piece",
                    "input_token_text",
                    "next_token_id",
                    "layer",
                    "top1_expert",
                    "top2_expert",
                    "top1_routing_weight",
                    "top2_routing_weight",
                    "top1_router_probability",
                    "top2_router_probability",
                )
            )
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
                next_token = output.logits[:, -1:].argmax(dim=-1)
                write_decode_rows(writer, tokenizer, decode_step, next_input_token, next_token, traces)
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
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "document_index": args.document_index,
        "document_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest(),
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "num_layers": len(model.model.layers),
        "experts_per_layer": model.config.num_local_experts,
        "experts_per_token": model.config.num_experts_per_tok,
        "cache_policy": FIXED_PER_LAYER_LRU,
        "cache_slots": expert_cache.capacity_slots,
        "per_layer_cache_capacities": list(expert_cache.layer_capacities or ()),
        "cache_stats": asdict(expert_cache.stats()),
        "arguments": vars(args) | {"output": str(output_path)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({args.decode_tokens * len(model.model.layers):,} rows).")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

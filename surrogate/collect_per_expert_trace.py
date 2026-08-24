#!/usr/bin/env python3
"""Collect routed inputs and exact unweighted BF16 targets for one Mixtral MoE layer."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pg19 import DEFAULT_DATASET, PG19Sample, load_deterministic_pg19_samples
from offload.expert_cache import GLOBAL_LAYER_BALANCED_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import CachedOffloadedMixtralExperts, replace_with_cached_offloaded_experts


DEFAULT_MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
HIDDEN_SIZE = 4096
NUM_EXPERTS = 8
SPLITS = ("train", "val", "test")


def move_non_expert_modules_to_cuda(model: torch.nn.Module) -> None:
    """Match the existing exact offloaded runtime placement."""
    model.model.embed_tokens.to("cuda")
    model.model.norm.to("cuda")
    model.model.rotary_emb.to("cuda")
    model.lm_head.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.to("cuda")
        layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm.to("cuda")
        layer.mlp.gate.to("cuda")


def exact_expert_forward(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    """Execute the exact BF16 Mixtral SwiGLU expert without router scaling."""
    gate, up = F.linear(hidden_states, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


@dataclass
class ExpertTraceFragments:
    """CPU fragments accumulated for one expert across document prefills."""

    inputs: list[torch.Tensor] = field(default_factory=list)
    router_weights: list[torch.Tensor] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    token_positions: list[torch.Tensor] = field(default_factory=list)

    def extend(self, other: "ExpertTraceFragments") -> None:
        """Append one document's routed rows while preserving their order."""
        self.inputs.extend(other.inputs)
        self.router_weights.extend(other.router_weights)
        self.sample_ids.extend(other.sample_ids)
        self.token_positions.extend(other.token_positions)


class TargetLayerTraceHook:
    """Collect one document's target-layer routed inputs via a pre-forward hook."""

    def __init__(self, module: CachedOffloadedMixtralExperts) -> None:
        self.num_experts = module.num_experts
        self.active_sample_id: str | None = None
        self.expected_token_count = 0
        self.fragments = self._empty_fragments()
        self.handle = module.register_forward_pre_hook(self._forward_pre_hook)

    def _empty_fragments(self) -> list[ExpertTraceFragments]:
        return [ExpertTraceFragments() for _ in range(self.num_experts)]

    def begin_document(self, sample_id: str, token_count: int) -> None:
        """Enable capture for exactly one document prefill."""
        if self.active_sample_id is not None:
            raise RuntimeError("target-layer trace hook is already active")
        self.active_sample_id = sample_id
        self.expected_token_count = token_count
        self.fragments = self._empty_fragments()

    def finish_document(self) -> list[ExpertTraceFragments]:
        """Disable capture and return the complete routed rows for the document."""
        if self.active_sample_id is None:
            raise RuntimeError("target-layer trace hook is inactive")
        self.active_sample_id = None
        self.expected_token_count = 0
        return self.fragments

    def close(self) -> None:
        """Remove the registered PyTorch hook."""
        self.handle.remove()

    def _forward_pre_hook(self, _: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        if self.active_sample_id is None:
            return
        if len(inputs) != 3 or not all(isinstance(value, torch.Tensor) for value in inputs):
            raise TypeError("expected hidden_states, top_k_index, and top_k_weights tensor inputs")
        hidden_states, top_k_index, top_k_weights = inputs
        flattened_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        flattened_indices = top_k_index.reshape(flattened_hidden_states.shape[0], -1)
        flattened_weights = top_k_weights.reshape(flattened_hidden_states.shape[0], -1)
        if flattened_hidden_states.shape != (self.expected_token_count, HIDDEN_SIZE):
            raise AssertionError("target-layer hidden states do not align with the active document")
        if flattened_indices.shape != flattened_weights.shape:
            raise AssertionError("router indices and weights have different shapes")
        if flattened_indices.shape[0] != self.expected_token_count:
            raise AssertionError("router rows do not align with target-layer hidden states")

        for expert_id in range(self.num_experts):
            token_indices, top_k_slots = torch.where(flattened_indices == expert_id)
            if token_indices.numel() == 0:
                continue
            fragments = self.fragments[expert_id]
            fragments.inputs.append(
                flattened_hidden_states[token_indices].detach().to("cpu", dtype=torch.bfloat16).clone()
            )
            fragments.router_weights.append(
                flattened_weights[token_indices, top_k_slots].detach().to("cpu", dtype=torch.float32).clone()
            )
            fragments.sample_ids.extend([self.active_sample_id] * token_indices.numel())
            fragments.token_positions.append(token_indices.detach().to("cpu", dtype=torch.int32).clone())


class OffloadedTraceRuntime:
    """Exact BF16 Mixtral execution plus trace collection for one target layer."""

    def __init__(self, model_path: str, layer_id: int, cache_slots: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for exact BF16 offloaded trace collection")
        if cache_slots < NUM_EXPERTS:
            raise ValueError("--cache-slots must be at least 8")

        print(f"Loading model on CPU: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.host_store = HostExpertStore(self.model)
        move_non_expert_modules_to_cuda(self.model)
        self.expert_cache = GPUExpertCache(
            self.host_store,
            capacity_slots=cache_slots,
            cache_policy=GLOBAL_LAYER_BALANCED_LRU,
        )
        replace_with_cached_offloaded_experts(self.model, self.expert_cache)
        target_executor = self.model.model.layers[layer_id].mlp.experts
        if not isinstance(target_executor, CachedOffloadedMixtralExperts):
            raise AssertionError("failed to install the cached target expert executor")
        self.trace_hook = TargetLayerTraceHook(target_executor)
        self.layer_id = layer_id

    @torch.inference_mode()
    def trace_document(self, sample_id: str, input_ids: torch.Tensor) -> list[ExpertTraceFragments]:
        """Run one cache-free document prefill and return its routed target-layer inputs."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("trace collection requires exactly one document per prefill forward")
        self.expert_cache.clear()
        self.trace_hook.begin_document(sample_id, int(input_ids.shape[1]))
        try:
            output = self.model(input_ids=input_ids.to("cuda"), use_cache=False, logits_to_keep=1)
            if not torch.isfinite(output.logits).all():
                raise FloatingPointError(f"non-finite logits while tracing {sample_id}")
        finally:
            document_fragments = self.trace_hook.finish_document()
        return document_fragments

    @torch.inference_mode()
    def compute_exact_targets(self, inputs: torch.Tensor, expert_id: int, batch_size: int) -> torch.Tensor:
        """Recompute unweighted exact BF16 expert outputs through the shared cache."""
        if inputs.dtype != torch.bfloat16 or inputs.device.type != "cpu":
            raise ValueError("trace inputs must be CPU BF16 tensors")
        gate_up, down = self.expert_cache.get(self.layer_id, expert_id)
        output_batches: list[torch.Tensor] = []
        for start_index in range(0, inputs.shape[0], batch_size):
            batch_inputs = inputs[start_index : start_index + batch_size].to("cuda")
            batch_outputs = exact_expert_forward(batch_inputs, gate_up, down)
            output_batches.append(batch_outputs.to("cpu", dtype=torch.bfloat16).clone())
        return torch.cat(output_batches, dim=0)

    def close(self) -> None:
        """Release the trace hook before the runtime is discarded."""
        self.trace_hook.close()


def parse_args() -> argparse.Namespace:
    """Parse collection configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-documents", type=int, default=28)
    parser.add_argument("--train-documents", type=int, default=20)
    parser.add_argument("--val-documents", type=int, default=4)
    parser.add_argument("--test-documents", type=int, default=4)
    parser.add_argument("--tokens-per-document", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-max-scan", type=int, default=1024)
    parser.add_argument("--cache-slots", type=int, default=8)
    parser.add_argument("--target-batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate document split and runtime settings."""
    if not 0 <= args.layer < HostExpertStore.NUM_LAYERS:
        raise ValueError("--layer must be in [0, 31]")
    if min(args.tokens_per_document, args.target_batch_size, args.cpu_threads) < 1:
        raise ValueError("token, target-batch-size, and cpu-threads arguments must be positive")
    split_counts = (args.train_documents, args.val_documents, args.test_documents)
    if min(split_counts) < 1 or sum(split_counts) != args.num_documents:
        raise ValueError("document split counts must be positive and sum to --num-documents")


def document_splits(samples: list[PG19Sample], args: argparse.Namespace) -> dict[str, list[PG19Sample]]:
    """Create the required deterministic document-level train/validation/test split."""
    train_end = args.train_documents
    validation_end = train_end + args.val_documents
    return {
        "train": samples[:train_end],
        "val": samples[train_end:validation_end],
        "test": samples[validation_end:],
    }


def save_expert_trace(
    path: Path,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    fragments: ExpertTraceFragments,
) -> None:
    """Write one expert and split trace with aligned route metadata."""
    router_weights = torch.cat(fragments.router_weights)
    token_positions = torch.cat(fragments.token_positions)
    row_count = inputs.shape[0]
    if targets.shape != inputs.shape or len(fragments.sample_ids) != row_count:
        raise AssertionError("trace targets or sample IDs do not align with inputs")
    if router_weights.shape[0] != row_count or token_positions.shape[0] != row_count:
        raise AssertionError("trace router metadata does not align with inputs")
    torch.save(
        {
            "x": inputs,
            "y": targets,
            "router_weight": router_weights,
            "sample_ids": fragments.sample_ids,
            "token_positions": token_positions,
        },
        path,
    )


def current_commit() -> str:
    """Return the current project commit for output metadata."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir or PROJECT_ROOT / "results" / "surrogate_per_expert" / f"layer_{args.layer}"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty: {output_dir}; use --overwrite")
    for split in SPLITS:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = load_deterministic_pg19_samples(
        tokenizer,
        num_samples=args.num_documents,
        sample_seed=args.seed,
        prefill_tokens=args.tokens_per_document,
        max_decode_tokens=1,
        dataset_name=args.dataset,
        split="train",
        max_scan=args.dataset_max_scan,
        allow_shorter_decode=False,
    )
    splits = document_splits(samples, args)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise AssertionError("deterministic PG19 selection produced duplicate document IDs")

    split_fragments = {split: [ExpertTraceFragments() for _ in range(NUM_EXPERTS)] for split in SPLITS}
    runtime = OffloadedTraceRuntime(args.model, args.layer, args.cache_slots)
    try:
        for split, split_samples in splits.items():
            for document_number, sample in enumerate(split_samples, start=1):
                input_ids = sample.input_ids[:, : args.tokens_per_document]
                document_fragments = runtime.trace_document(sample.sample_id, input_ids)
                for expert_id, fragments in enumerate(document_fragments):
                    split_fragments[split][expert_id].extend(fragments)
                print(f"Traced {split} document {document_number}/{len(split_samples)}: {sample.sample_id}")

        sample_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
        for split in SPLITS:
            for expert_id, fragments in enumerate(split_fragments[split]):
                if not fragments.inputs:
                    raise RuntimeError(f"no routed inputs for expert {expert_id} in {split} documents")
                inputs = torch.cat(fragments.inputs, dim=0)
                targets = runtime.compute_exact_targets(inputs, expert_id, args.target_batch_size)
                save_expert_trace(output_dir / split / f"expert_{expert_id}.pt", inputs, targets, fragments)
                sample_counts[split][f"expert_{expert_id}"] = int(inputs.shape[0])
                print(f"{split:5s} E{expert_id}: N={inputs.shape[0]:,}, x/y={tuple(inputs.shape)}")
    finally:
        runtime.close()

    metadata: dict[str, Any] = {
        "surrogate_trace_version": 1,
        "git_commit": current_commit(),
        "model_path": args.model,
        "layer": args.layer,
        "dataset": args.dataset,
        "dataset_split": "train",
        "seed": args.seed,
        "tokens_per_document": args.tokens_per_document,
        "document_counts": {split: len(split_samples) for split, split_samples in splits.items()},
        "document_ids": {split: [sample.sample_id for sample in split_samples] for split, split_samples in splits.items()},
        "expert_sample_counts": sample_counts,
        "trace_dtype": "bfloat16",
        "target_definition": "unweighted exact BF16 Mixtral expert output",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved traces to: {output_dir}")


if __name__ == "__main__":
    main()

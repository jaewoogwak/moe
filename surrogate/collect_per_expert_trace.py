#!/usr/bin/env python3
"""Collect routed inputs and exact unweighted BF16 expert outputs for one layer."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data.pg19 import DEFAULT_DATASET, load_deterministic_pg19_samples
from offload.expert_cache import GLOBAL_LAYER_BALANCED_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import CachedOffloadedMixtralExperts, replace_with_cached_offloaded_experts

MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
SPLITS = ("train", "val", "test")
H = 4096


def move_non_experts_to_cuda(model: torch.nn.Module) -> None:
    model.model.embed_tokens.to("cuda")
    model.model.norm.to("cuda")
    model.model.rotary_emb.to("cuda")
    model.lm_head.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.to("cuda")
        layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm.to("cuda")
        layer.mlp.gate.to("cuda")


def exact_expert_forward(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """Exact Mixtral computation; deliberately excludes router-weight scaling."""
    gate, up = F.linear(x, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


@dataclass
class Fragments:
    x: list[torch.Tensor] = field(default_factory=list)
    router_weight: list[torch.Tensor] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)
    token_positions: list[torch.Tensor] = field(default_factory=list)

    def extend(self, other: "Fragments") -> None:
        self.x.extend(other.x)
        self.router_weight.extend(other.router_weight)
        self.sample_ids.extend(other.sample_ids)
        self.token_positions.extend(other.token_positions)


class RoutedInputHook:
    """A target CachedOffloadedMixtralExperts pre-hook, scoped to one document."""
    def __init__(self, module: CachedOffloadedMixtralExperts) -> None:
        self.nexperts = module.num_experts
        self.sample_id: str | None = None
        self.tokens = 0
        self.parts = [Fragments() for _ in range(self.nexperts)]
        self.handle = module.register_forward_pre_hook(self._forward_pre_hook)

    def begin(self, sample_id: str, tokens: int) -> None:
        if self.sample_id is not None:
            raise RuntimeError("trace hook already active")
        self.sample_id, self.tokens = sample_id, tokens
        self.parts = [Fragments() for _ in range(self.nexperts)]

    def finish(self) -> list[Fragments]:
        if self.sample_id is None:
            raise RuntimeError("trace hook is inactive")
        self.sample_id, self.tokens = None, 0
        return self.parts

    def close(self) -> None:
        self.handle.remove()

    def _forward_pre_hook(self, _: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        if self.sample_id is None:
            return
        if len(inputs) != 3 or not all(isinstance(item, torch.Tensor) for item in inputs):
            raise TypeError("expected hidden_states, top_k_index, top_k_weights tensor inputs")
        hidden, indices, weights = inputs
        hidden = hidden.reshape(-1, hidden.shape[-1])
        indices, weights = indices.reshape(hidden.shape[0], -1), weights.reshape(hidden.shape[0], -1)
        if hidden.shape != (self.tokens, H) or indices.shape != weights.shape or indices.shape[0] != self.tokens:
            raise AssertionError("target expert hook inputs do not align with this prefill document")
        for expert in range(self.nexperts):
            token, topk = torch.where(indices == expert)
            if not token.numel():
                continue
            part = self.parts[expert]
            part.x.append(hidden[token].detach().to("cpu", torch.bfloat16).clone())
            part.router_weight.append(weights[token, topk].detach().to("cpu", torch.float32).clone())
            part.sample_ids.extend([self.sample_id] * token.numel())
            part.token_positions.append(token.detach().to("cpu", torch.int32).clone())


class Runtime:
    def __init__(self, model_path: str, layer: int, cache_slots: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if cache_slots < 8:
            raise ValueError("--cache-slots must be at least 8")
        print(f"Loading model on CPU: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
        self.model.eval()
        self.store = HostExpertStore(self.model)
        move_non_experts_to_cuda(self.model)
        self.cache = GPUExpertCache(self.store, cache_slots, cache_policy=GLOBAL_LAYER_BALANCED_LRU)
        replace_with_cached_offloaded_experts(self.model, self.cache)
        executor = self.model.model.layers[layer].mlp.experts
        if not isinstance(executor, CachedOffloadedMixtralExperts):
            raise AssertionError("failed to install cached expert executor")
        self.hook, self.layer = RoutedInputHook(executor), layer

    @torch.inference_mode()
    def trace(self, sample_id: str, ids: torch.Tensor) -> list[Fragments]:
        if ids.shape[0] != 1:
            raise ValueError("one document must be traced in each prefill forward")
        self.cache.clear()
        self.hook.begin(sample_id, ids.shape[1])
        try:
            output = self.model(input_ids=ids.to("cuda"), use_cache=False, logits_to_keep=1)
            if not torch.isfinite(output.logits).all():
                raise FloatingPointError(f"non-finite logits for {sample_id}")
        finally:
            parts = self.hook.finish()
        return parts

    @torch.inference_mode()
    def targets(self, x: torch.Tensor, expert: int, batch: int) -> torch.Tensor:
        gate_up, down = self.cache.get(self.layer, expert)  # reuse exact cache-owned weights
        ys = []
        for start in range(0, x.shape[0], batch):
            y = exact_expert_forward(x[start : start + batch].to("cuda"), gate_up, down)
            ys.append(y.to("cpu", torch.bfloat16).clone())
        return torch.cat(ys)

    def close(self) -> None:
        self.hook.close()


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layer", type=int, default=16)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--num-documents", type=int, default=28)
    p.add_argument("--train-documents", type=int, default=20)
    p.add_argument("--val-documents", type=int, default=4)
    p.add_argument("--test-documents", type=int, default=4)
    p.add_argument("--tokens-per-document", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-max-scan", type=int, default=1024)
    p.add_argument("--cache-slots", type=int, default=8)
    p.add_argument("--target-batch-size", type=int, default=64)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    a = args()
    if not 0 <= a.layer < 32 or min(a.tokens_per_document, a.target_batch_size, a.cpu_threads) < 1:
        raise ValueError("invalid layer or non-positive size argument")
    if min(a.train_documents, a.val_documents, a.test_documents) < 1 or a.train_documents + a.val_documents + a.test_documents != a.num_documents:
        raise ValueError("document splits must be positive and sum to --num-documents")
    out = a.output_dir or ROOT / "results" / "surrogate_per_expert" / f"layer_{a.layer}"
    if out.exists() and any(out.iterdir()) and not a.overwrite:
        raise FileExistsError(f"non-empty output directory: {out}; use --overwrite")
    for split in SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.cpu_threads)
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    samples = load_deterministic_pg19_samples(tokenizer, num_samples=a.num_documents, sample_seed=a.seed,
        prefill_tokens=a.tokens_per_document, max_decode_tokens=1, dataset_name=a.dataset, split="train",
        max_scan=a.dataset_max_scan, allow_shorter_decode=False)
    cuts = (a.train_documents, a.train_documents + a.val_documents)
    groups = {"train": samples[:cuts[0]], "val": samples[cuts[0]:cuts[1]], "test": samples[cuts[1]:]}
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise AssertionError("duplicate document IDs")
    grouped = {name: [Fragments() for _ in range(8)] for name in SPLITS}
    runtime = Runtime(a.model, a.layer, a.cache_slots)
    try:
        for split, docs in groups.items():
            for n, sample in enumerate(docs, 1):
                parts = runtime.trace(sample.sample_id, sample.input_ids[:, :a.tokens_per_document])
                for expert, part in enumerate(parts): grouped[split][expert].extend(part)
                print(f"Traced {split} document {n}/{len(docs)}: {sample.sample_id}")
        counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
        for split in SPLITS:
            for expert, part in enumerate(grouped[split]):
                x = torch.cat(part.x) if part.x else torch.empty((0, H), dtype=torch.bfloat16)
                if not x.shape[0]: raise RuntimeError(f"no routed inputs for E{expert} in {split}")
                y = runtime.targets(x, expert, a.target_batch_size)
                weights = torch.cat(part.router_weight); positions = torch.cat(part.token_positions)
                if len(part.sample_ids) != x.shape[0] or weights.shape[0] != x.shape[0] or positions.shape[0] != x.shape[0] or y.shape != x.shape:
                    raise AssertionError("trace metadata or exact output shape mismatch")
                torch.save({"x": x, "y": y, "router_weight": weights, "sample_ids": part.sample_ids,
                            "token_positions": positions}, out / split / f"expert_{expert}.pt")
                counts[split][f"expert_{expert}"] = x.shape[0]
                print(f"{split:5} E{expert}: N={x.shape[0]:,}, x/y={tuple(x.shape)}")
    finally:
        runtime.close()
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError: commit = "unknown"
    metadata: dict[str, Any] = {"surrogate_trace_version": 1, "git_commit": commit, "model_path": a.model, "layer": a.layer,
        "dataset": a.dataset, "dataset_split": "train", "seed": a.seed, "tokens_per_document": a.tokens_per_document,
        "document_counts": {k: len(v) for k, v in groups.items()}, "document_ids": {k: [s.sample_id for s in v] for k, v in groups.items()},
        "expert_sample_counts": counts, "trace_dtype": "bfloat16", "target_definition": "unweighted exact BF16 Mixtral expert output",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved traces to: {out}")

if __name__ == "__main__": main()

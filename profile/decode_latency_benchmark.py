"""Decode-latency benchmark for cached CPU-to-GPU Mixtral expert offloading."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offload.expert_cache import ExpertCacheStats, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import CachedOffloadedMixtralExperts, replace_with_cached_offloaded_experts
from data.longbench_v2 import DATASET_NAME, LongBenchSample, build_prompt_input_ids, load_deterministic_samples


MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
DEFAULT_CONTEXTS = (4096, 8192, 16384, 24576, 32768)
GPU_BUDGET_GIB = 22.0
RUNTIME_RESERVE_GIB = 1.0
DEFAULT_WARMUP_TOKENS = 16
DEFAULT_MEASURE_TOKENS = 64
SAMPLE_SELECTION_PROMPT_LENGTH = 32768 - DEFAULT_WARMUP_TOKENS - DEFAULT_MEASURE_TOKENS


@dataclass
class TokenResult:
    sample_id: str
    requested_context_length: int
    measured_start_kv_length: int
    measured_end_kv_length: int
    decode_token: int
    total_ms: float
    attention_ms: float
    router_ms: float
    expert_h2d_ms: float
    host_staging_ms: float
    expert_compute_ms: float
    other_ms: float
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    expert_accesses: int
    h2d_bytes: int
    kv_cache_bytes: int
    expert_cache_allocated_bytes: int
    expert_cache_resident_bytes: int
    expert_cache_capacity_slots: int
    expert_cache_resident_slots: int


class TokenProfiler:
    """Collect default-stream CUDA event ranges for one decode forward."""

    def __init__(self, expert_cache: GPUExpertCache, expected_expert_accesses: int) -> None:
        self.expert_cache = expert_cache
        self.expected_expert_accesses = expected_expert_accesses
        self.active = False
        self._events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._cache_before: ExpertCacheStats | None = None

    def begin(self) -> None:
        self.active = True
        self._events.clear()
        self._cache_before = self.expert_cache.stats()

    @contextmanager
    def cuda_section(self, name: str) -> Iterator[None]:
        if not self.active:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._events[name].append((start, end))

    def begin_section(self, name: str) -> tuple[str, torch.cuda.Event] | None:
        if not self.active:
            return None
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        return name, start

    def end_section(self, section: tuple[str, torch.cuda.Event] | None) -> None:
        if section is None:
            return
        name, start = section
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events[name].append((start, end))

    def finish(
        self,
        sample_id: str,
        requested_context_length: int,
        measured_start_kv_length: int,
        measured_end_kv_length: int,
        decode_token: int,
        total_ms: float,
        kv_cache_bytes: int,
    ) -> TokenResult:
        if self._cache_before is None:
            raise RuntimeError("begin() must be called before finish()")
        cache_after = self.expert_cache.stats()

        def elapsed(name: str) -> float:
            return sum(start.elapsed_time(end) for start, end in self._events[name])

        attention_ms = elapsed("attention")
        router_ms = elapsed("router")
        h2d_ms = elapsed("expert_h2d")
        expert_compute_ms = elapsed("expert_compute")
        h2d_bytes = cache_after.h2d_bytes - self._cache_before.h2d_bytes
        host_staging_ms = cache_after.host_staging_ms - self._cache_before.host_staging_ms
        component_sum = attention_ms + router_ms + h2d_ms + expert_compute_ms
        misses = cache_after.misses - self._cache_before.misses
        hits = cache_after.hits - self._cache_before.hits
        expert_accesses = hits + misses
        if h2d_bytes != misses * self.expert_cache.expert_size_bytes:
            raise AssertionError("cache miss H2D bytes do not match one BF16 expert per miss")
        if expert_accesses != self.expected_expert_accesses:
            raise RuntimeError(
                f"expected {self.expected_expert_accesses} expert accesses for batch-1 decode, "
                f"observed {expert_accesses}"
            )
        if self.expert_cache.allocated_bytes != (
            self.expert_cache.capacity_slots * self.expert_cache.expert_size_bytes
        ):
            raise AssertionError("expert-cache allocated bytes do not match its slot capacity")
        if self.expert_cache.resident_bytes != (
            self.expert_cache.resident_slots * self.expert_cache.expert_size_bytes
        ):
            raise AssertionError("expert-cache resident bytes do not match its occupied slots")
        if self.expert_cache.resident_slots > self.expert_cache.capacity_slots:
            raise AssertionError("expert-cache residency exceeds its slot capacity")

        other_ms = total_ms - component_sum
        if other_ms < -1.0:
            print(
                f"WARNING: latency component sum exceeds total by {-other_ms:.2f} ms "
                f"at requested context {requested_context_length}, token {decode_token}"
            )
        self.active = False
        return TokenResult(
            sample_id=sample_id,
            requested_context_length=requested_context_length,
            measured_start_kv_length=measured_start_kv_length,
            measured_end_kv_length=measured_end_kv_length,
            decode_token=decode_token,
            total_ms=total_ms,
            attention_ms=attention_ms,
            router_ms=router_ms,
            expert_h2d_ms=h2d_ms,
            host_staging_ms=host_staging_ms,
            expert_compute_ms=expert_compute_ms,
            other_ms=other_ms,
            cache_hits=hits,
            cache_misses=misses,
            cache_evictions=cache_after.evictions - self._cache_before.evictions,
            expert_accesses=expert_accesses,
            h2d_bytes=h2d_bytes,
            kv_cache_bytes=kv_cache_bytes,
            expert_cache_allocated_bytes=self.expert_cache.allocated_bytes,
            expert_cache_resident_bytes=self.expert_cache.resident_bytes,
            expert_cache_capacity_slots=self.expert_cache.capacity_slots,
            expert_cache_resident_slots=self.expert_cache.resident_slots,
        )


def attach_module_timers(model: torch.nn.Module, profiler: TokenProfiler) -> list[torch.utils.hooks.RemovableHandle]:
    """Time unchanged HF attention and router modules using forward hooks."""
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(module: torch.nn.Module, section_name: str) -> None:
        stack: list[tuple[str, torch.cuda.Event] | None] = []

        def pre_hook(_: torch.nn.Module, __: tuple[object, ...]) -> None:
            stack.append(profiler.begin_section(section_name))

        def post_hook(_: torch.nn.Module, __: tuple[object, ...], output: object) -> object:
            profiler.end_section(stack.pop())
            return output

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    for layer in model.model.layers:
        attach(layer.self_attn, "attention")
        attach(layer.mlp.gate, "router")
        assert isinstance(layer.mlp.experts, CachedOffloadedMixtralExperts)
        layer.mlp.experts.profiler = profiler
    return handles


def move_non_expert_modules_to_cuda(model: torch.nn.Module) -> None:
    """Move all inference modules except the original expert tensors to CUDA."""
    model.model.embed_tokens.to("cuda")
    model.model.norm.to("cuda")
    model.model.rotary_emb.to("cuda")
    model.lm_head.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.to("cuda")
        layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm.to("cuda")
        layer.mlp.gate.to("cuda")


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def kv_cache_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if isinstance(keys, torch.Tensor):
            total += tensor_bytes(keys)
        if isinstance(values, torch.Tensor):
            total += tensor_bytes(values)
    return total


def model_gpu_bytes(model: torch.nn.Module) -> int:
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type != "cuda":
            continue
        key = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
        if key not in seen:
            seen.add(key)
            total += key[1]
    return total


def validate_placement(model: torch.nn.Module, store: HostExpertStore) -> None:
    for layer_id in range(store.NUM_LAYERS):
        for expert_id in range(store.NUM_EXPERTS_PER_LAYER):
            gate_up, down = store.get(layer_id, expert_id)
            assert gate_up.device.type == "cpu" and down.device.type == "cpu"
            assert gate_up.dtype == torch.bfloat16 and down.dtype == torch.bfloat16
    assert model.model.embed_tokens.weight.device.type == "cuda"
    assert model.lm_head.weight.device.type == "cuda"
    for layer in model.model.layers:
        assert layer.self_attn.q_proj.weight.device.type == "cuda"
        assert layer.mlp.gate.weight.device.type == "cuda"


def expert_slots_for_context(
    gpu_budget_bytes: int,
    non_expert_model_bytes: int,
    kv_bytes: int,
    runtime_reserve_bytes: int,
    expert_size_bytes: int,
) -> int:
    available = gpu_budget_bytes - non_expert_model_bytes - kv_bytes - runtime_reserve_bytes
    slots = available // expert_size_bytes
    if slots < 1:
        raise RuntimeError("GPU budget cannot hold one BF16 expert at this context length")
    return slots


NUMERIC_RESULT_FIELDS = (
    "total_ms",
    "attention_ms",
    "router_ms",
    "expert_h2d_ms",
    "host_staging_ms",
    "expert_compute_ms",
    "other_ms",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "expert_accesses",
    "h2d_bytes",
    "kv_cache_bytes",
    "expert_cache_allocated_bytes",
    "expert_cache_resident_bytes",
    "expert_cache_capacity_slots",
    "expert_cache_resident_slots",
)


def average_results(results: list[TokenResult]) -> dict[str, float]:
    count = len(results)
    if count == 0:
        raise ValueError("no token results")
    return {field: sum(getattr(row, field) for row in results) / count for field in NUMERIC_RESULT_FIELDS}


class DecodeBenchmark:
    def __init__(self, model_path: str, gpu_budget_gib: float, runtime_reserve_gib: float) -> None:
        self.model_path = model_path
        self.gpu_budget_bytes = int(gpu_budget_gib * 1024**3)
        self.runtime_reserve_bytes = int(runtime_reserve_gib * 1024**3)

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
        self.expert_cache = GPUExpertCache(self.host_store, capacity_slots=1)
        replace_with_cached_offloaded_experts(self.model, self.expert_cache)
        validate_placement(self.model, self.host_store)
        self.non_expert_model_bytes = model_gpu_bytes(self.model)
        expected_accesses = self.model.config.num_hidden_layers * self.model.config.num_experts_per_tok
        self.profiler = TokenProfiler(self.expert_cache, expected_expert_accesses=expected_accesses)
        self.hooks = attach_module_timers(self.model, self.profiler)

    @torch.inference_mode()
    def run_context(
        self,
        sample_id: str,
        context_length: int,
        warmup_tokens: int,
        measure_tokens: int,
        prompt_input_ids: torch.Tensor,
    ) -> list[TokenResult]:
        max_positions = self.model.config.max_position_embeddings
        if context_length > max_positions:
            raise ValueError(
                f"requested context {context_length} exceeds native context window {max_positions}"
            )
        measured_end_kv_length = context_length
        measured_start_kv_length = measured_end_kv_length - measure_tokens
        if measured_start_kv_length < 1:
            raise ValueError("requested measured-token count leaves no valid prefill length")
        prefill_length = measured_start_kv_length - warmup_tokens
        if prefill_length < 1:
            raise ValueError("warmup_tokens must be smaller than the measured start KV length")
        if prompt_input_ids.shape != (1, prefill_length):
            raise AssertionError(
                f"input_ids shape {tuple(prompt_input_ids.shape)} does not equal (1, {prefill_length})"
            )

        print(
            f"Context plan: sample={sample_id}, requested={context_length}, prefill_length={prefill_length}, "
            f"measured_start_kv_length={measured_start_kv_length}, "
            f"measured_end_kv_length={measured_end_kv_length}"
        )
        self.expert_cache.clear()
        self.expert_cache.set_capacity_slots(1)
        prompt = prompt_input_ids.to(device="cuda")
        cache = DynamicCache(config=self.model.config)
        prefill_output = self.model(input_ids=prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)

        prefill_kv_bytes = kv_cache_bytes(cache)
        bytes_per_kv_token = prefill_kv_bytes // prefill_length
        expected_start_kv_bytes = prefill_kv_bytes + bytes_per_kv_token * warmup_tokens
        slots = expert_slots_for_context(
            self.gpu_budget_bytes,
            self.non_expert_model_bytes,
            expected_start_kv_bytes,
            self.runtime_reserve_bytes,
            self.expert_cache.expert_size_bytes,
        )
        self.expert_cache.set_capacity_slots(slots)
        self.expert_cache.clear()

        next_token = prefill_output.logits[:, -1:].argmax(dim=-1)
        for _ in range(warmup_tokens):
            output = self.model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            next_token = output.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

        if cache.get_seq_length() != measured_start_kv_length:
            raise AssertionError(
                f"KV length after warmup is {cache.get_seq_length()}, expected {measured_start_kv_length}"
            )
        kv_before = kv_cache_bytes(cache)
        if kv_before != expected_start_kv_bytes:
            raise AssertionError(
                f"actual warmup KV bytes {kv_before} differ from expected {expected_start_kv_bytes}"
            )
        self.expert_cache.reset_stats()

        memory_before = torch.cuda.memory_allocated()
        resident_cache_before = self.expert_cache.allocated_bytes
        results: list[TokenResult] = []
        for token_index in range(measure_tokens):
            torch.cuda.synchronize()
            self.profiler.begin()
            start = time.perf_counter()
            output = self.model(input_ids=next_token, past_key_values=cache, use_cache=True, logits_to_keep=1)
            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - start) * 1000
            if not torch.isfinite(output.logits).all():
                raise AssertionError("decode produced NaN or Inf logits")
            next_token = output.logits[:, -1:].argmax(dim=-1)
            if cache.get_seq_length() > max_positions:
                raise AssertionError(
                    f"measured decode exceeded max positions: {cache.get_seq_length()} > {max_positions}"
            )
            results.append(
                self.profiler.finish(
                    sample_id=sample_id,
                    requested_context_length=context_length,
                    measured_start_kv_length=measured_start_kv_length,
                    measured_end_kv_length=measured_end_kv_length,
                    decode_token=token_index,
                    total_ms=total_ms,
                    kv_cache_bytes=kv_cache_bytes(cache),
                )
            )

        memory_after = torch.cuda.memory_allocated()
        kv_after = kv_cache_bytes(cache)
        resident_cache_after = self.expert_cache.allocated_bytes
        expected_growth = (kv_after - kv_before) + (resident_cache_after - resident_cache_before)
        actual_growth = memory_after - memory_before
        allocator_tolerance = 128 * 1024**2
        print(
            f"GPU memory: before={memory_before:,}, after={memory_after:,}, "
            f"KV before/after={kv_before:,}/{kv_after:,}, "
            f"cache before/after={resident_cache_before:,}/{resident_cache_after:,}"
        )
        if actual_growth > expected_growth + allocator_tolerance:
            raise AssertionError(
                "unexpected GPU memory growth: "
                f"actual={actual_growth:,}, expected={expected_growth:,}, "
                f"tolerance={allocator_tolerance:,}"
            )
        return results


def _write_json_csv(rows: list[dict[str, object]], json_path: Path, csv_path: Path) -> None:
    with json_path.open("w") as handle:
        json.dump(rows, handle, indent=2)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(results: list[TokenResult], output_dir: Path, metadata: dict[str, object]) -> None:
    if not results:
        raise ValueError("no token results to save")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_csv(
        [asdict(row) for row in results],
        output_dir / "decode_tokens.json",
        output_dir / "decode_tokens.csv",
    )
    with (output_dir / "decode_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    grouped: dict[tuple[str, int], list[TokenResult]] = defaultdict(list)
    for row in results:
        grouped[(row.sample_id, row.requested_context_length)].append(row)
    sample_summaries: list[dict[str, object]] = []
    for (sample_id, context), rows in sorted(grouped.items()):
        summary = average_results(rows)
        summary.update(
            sample_id=sample_id,
            requested_context_length=context,
            measured_start_kv_length=rows[0].measured_start_kv_length,
            measured_end_kv_length=rows[0].measured_end_kv_length,
            measured_tokens=len(rows),
        )
        summary["hit_rate"] = summary["cache_hits"] / max(1.0, summary["cache_hits"] + summary["cache_misses"])
        sample_summaries.append(summary)
    _write_json_csv(
        sample_summaries,
        output_dir / "decode_sample_context_summary.json",
        output_dir / "decode_sample_context_summary.csv",
    )

    context_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for summary in sample_summaries:
        context_groups[int(summary["requested_context_length"])].append(summary)
    summaries: list[dict[str, object]] = []
    for context, rows in sorted(context_groups.items()):
        if len({int(row["measured_start_kv_length"]) for row in rows}) != 1:
            raise AssertionError("sample summaries disagree on measured start KV length")
        if len({int(row["measured_end_kv_length"]) for row in rows}) != 1:
            raise AssertionError("sample summaries disagree on measured end KV length")
        summary: dict[str, object] = {
            "requested_context_length": context,
            "measured_start_kv_length": rows[0]["measured_start_kv_length"],
            "measured_end_kv_length": rows[0]["measured_end_kv_length"],
            "num_samples": len(rows),
        }
        for field in NUMERIC_RESULT_FIELDS:
            values = [float(row[field]) for row in rows]
            summary[field] = mean(values)
            summary[f"{field}_std"] = pstdev(values)
        hit_rates = [float(row["hit_rate"]) for row in rows]
        summary["hit_rate"] = mean(hit_rates)
        summary["hit_rate_std"] = pstdev(hit_rates)
        summaries.append(summary)
    _write_json_csv(summaries, output_dir / "decode_summary.json", output_dir / "decode_summary.csv")

    print()
    print(
        "Requested | Samples | Measured KV Range | KV GiB | Cache Capacity Slots | Resident Slots | "
        "Cache Allocated GiB | Cache Resident GiB | Hit Rate | Accesses/token | H2D GB/token | "
        "Host Staging ms | Attention ms | Router ms | H2D ms | Expert Compute ms | Other ms | TPOT ms"
    )
    for row in summaries:
        print(
            f"{int(row['requested_context_length']):9d} | "
            f"{int(row['num_samples']):7d} | "
            f"{int(row['measured_start_kv_length']):5d}->{int(row['measured_end_kv_length']):5d} | "
            f"{row['kv_cache_bytes'] / 1024**3:6.2f} | "
            f"{row['expert_cache_capacity_slots']:20.1f} | "
            f"{row['expert_cache_resident_slots']:14.1f} | "
            f"{row['expert_cache_allocated_bytes'] / 1024**3:19.2f} | "
            f"{row['expert_cache_resident_bytes'] / 1024**3:18.2f} | "
            f"{row['hit_rate']:8.3f} | "
            f"{row['expert_accesses']:14.1f} | "
            f"{row['h2d_bytes'] / 1e9:12.3f} | "
            f"{row['host_staging_ms']:15.2f} | "
            f"{row['attention_ms']:12.2f} | "
            f"{row['router_ms']:9.2f} | "
            f"{row['expert_h2d_ms']:6.2f} | "
            f"{row['expert_compute_ms']:17.2f} | "
            f"{row['other_ms']:8.2f} | "
            f"{row['total_ms']:7.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    parser.add_argument("--measure-tokens", type=int, default=DEFAULT_MEASURE_TOKENS)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--gpu-budget-gib", type=float, default=GPU_BUDGET_GIB)
    parser.add_argument("--runtime-reserve-gib", type=float, default=RUNTIME_RESERVE_GIB)
    parser.add_argument("--output-dir", type=Path, default=Path("results/decode_latency"))
    args = parser.parse_args()

    if not args.contexts:
        raise ValueError("at least one context length is required")
    if max(args.contexts) > 32768:
        raise ValueError("requested context exceeds Mixtral's native 32768-token context window")
    longest_prefill_length = max(args.contexts) - args.measure_tokens - args.warmup_tokens
    if longest_prefill_length < 1:
        raise ValueError("warmup and measured tokens leave no prefill tokens")
    selection_prompt_length = max(longest_prefill_length, SAMPLE_SELECTION_PROMPT_LENGTH)

    print(f"Loading Mixtral tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = load_deterministic_samples(
        tokenizer,
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        required_prompt_length=selection_prompt_length,
    )
    print(f"LongBench v2 samples (seed={args.sample_seed}): {[sample.sample_id for sample in samples]}")

    benchmark = DecodeBenchmark(args.model, args.gpu_budget_gib, args.runtime_reserve_gib)
    all_results: list[TokenResult] = []
    for sample in samples:
        for context_length in args.contexts:
            prefill_length = context_length - args.measure_tokens - args.warmup_tokens
            prompt_input_ids = build_prompt_input_ids(tokenizer, sample, prefill_length)
            repeated_input_ids = build_prompt_input_ids(tokenizer, sample, prefill_length)
            if not torch.equal(prompt_input_ids, repeated_input_ids):
                raise AssertionError(f"prompt construction is not deterministic for sample {sample.sample_id}")
            if prompt_input_ids.shape[-1] != prefill_length:
                raise AssertionError(
                    f"input_ids length {prompt_input_ids.shape[-1]} does not equal {prefill_length}"
                )
            print(f"Running sample {sample.sample_id}, context length {context_length}")
            all_results.extend(
                benchmark.run_context(
                    sample.sample_id,
                    context_length,
                    args.warmup_tokens,
                    args.measure_tokens,
                    prompt_input_ids,
                )
            )
    metadata = {
        "model": args.model,
        "dataset": DATASET_NAME,
        "sample_seed": args.sample_seed,
        "sample_ids": [sample.sample_id for sample in samples],
        "contexts": args.contexts,
        "warmup_tokens": args.warmup_tokens,
        "measure_tokens": args.measure_tokens,
        "sample_selection_prompt_length": selection_prompt_length,
    }
    save_results(all_results, args.output_dir, metadata)


if __name__ == "__main__":
    main()

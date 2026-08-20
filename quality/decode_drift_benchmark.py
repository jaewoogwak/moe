"""Teacher-forced Mixtral expert fake-quantization drift benchmark.

The benchmark deliberately keeps one model instance. It uses BF16 experts for
prefill and the full FP trajectory, restores the same prefill KV snapshot, then
fake-quantizes only pinned expert weights and lets the quantized trajectory
grow its own KV cache under exactly the same ground-truth input tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pg19 import DEFAULT_DATASET, PG19Sample, load_deterministic_pg19_samples
from offload.expert_cache import FIXED_PER_LAYER_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import replace_with_cached_offloaded_experts
from quant.fake_quant import QuantizationStats, fake_quantize_expert_store_


DEFAULT_MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
TOKEN_FIELDS = (
    "sample_id",
    "quant_bits",
    "decode_position",
    "fp_nll",
    "q_nll",
    "delta_nll",
    "routing_drift_mean",
    "router_logit_rel_l2_mean",
    "router_logit_cosine_distance_mean",
    "hidden_rel_l2_mean",
    "hidden_cosine_distance_mean",
    "logit_kl",
)
LAYER_FIELDS = (
    "sample_id",
    "quant_bits",
    "decode_position",
    "layer_id",
    "routing_drift",
    "fp_top1",
    "fp_top2",
    "q_top1",
    "q_top2",
    "router_margin_fp",
    "router_logit_rel_l2",
    "router_logit_cosine_distance",
    "hidden_rel_l2",
    "hidden_cosine_distance",
)
CORPUS_FIELDS = (
    "quant_bits",
    "completed_samples",
    "scored_tokens",
    "mean_fp_nll",
    "fp_ppl",
    "mean_q_nll",
    "q_ppl",
    "mean_delta_nll",
    "ppl_ratio",
)


@dataclass(frozen=True)
class RouterTrace:
    top1: int
    top2: int
    margin: float
    logits_cpu: torch.Tensor


@dataclass
class StepTrace:
    routes: list[RouterTrace]
    hidden: list[torch.Tensor | None]


@dataclass
class FPTokenTrace:
    nll: float
    logits_cpu: torch.Tensor
    step: StepTrace


class TraceHooks:
    """Collect compact routing traces and strided post-layer hidden states."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.num_layers = len(model.model.layers)
        self.active = False
        self.capture_hidden = False
        self.routes: list[RouterTrace | None] = []
        self.hidden: list[torch.Tensor | None] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer_id, layer in enumerate(model.model.layers):
            self.handles.append(layer.mlp.gate.register_forward_hook(self._router_hook(layer_id)))
            self.handles.append(layer.register_forward_hook(self._layer_hook(layer_id)))

    def _router_hook(self, layer_id: int):
        def hook(_: torch.nn.Module, __: tuple[object, ...], output: object) -> None:
            if not self.active:
                return
            router_logits, _, router_indices = output  # MixtralTopKRouter API in Transformers 5.15.
            logits = router_logits.reshape(-1, router_logits.shape[-1])[-1].float()
            top_indices = router_indices.reshape(-1, router_indices.shape[-1])[-1]
            if top_indices.numel() != 2:
                raise AssertionError("Mixtral quality benchmark expects top-2 routing")
            ordered_logits = torch.topk(logits, k=3).values
            self.routes[layer_id] = RouterTrace(
                top1=int(top_indices[0].item()),
                top2=int(top_indices[1].item()),
                margin=float((ordered_logits[1] - ordered_logits[2]).item()),
                logits_cpu=logits.detach().to(device="cpu", dtype=torch.float32).clone(),
            )

        return hook

    def _layer_hook(self, layer_id: int):
        def hook(_: torch.nn.Module, __: tuple[object, ...], output: object) -> None:
            if self.active and self.capture_hidden:
                if not isinstance(output, torch.Tensor):
                    raise TypeError("MixtralDecoderLayer forward must return a hidden-state tensor")
                self.hidden[layer_id] = output[:, -1, :].detach().to(device="cpu", dtype=torch.bfloat16).clone()

        return hook

    def begin(self, *, capture_hidden: bool) -> None:
        self.active = True
        self.capture_hidden = capture_hidden
        self.routes = [None] * self.num_layers
        self.hidden = [None] * self.num_layers

    def finish(self) -> StepTrace:
        self.active = False
        if any(route is None for route in self.routes):
            raise AssertionError("did not capture router output for every Mixtral layer")
        return StepTrace(
            routes=[route for route in self.routes if route is not None],
            hidden=list(self.hidden),
        )

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def move_non_expert_modules_to_cuda(model: torch.nn.Module) -> None:
    """Match the offloaded latency runtime without importing its benchmark."""
    model.model.embed_tokens.to("cuda")
    model.model.norm.to("cuda")
    model.model.rotary_emb.to("cuda")
    model.lm_head.to("cuda")
    for layer in model.model.layers:
        layer.self_attn.to("cuda")
        layer.input_layernorm.to("cuda")
        layer.post_attention_layernorm.to("cuda")
        layer.mlp.gate.to("cuda")


def snapshot_dynamic_cache(cache: DynamicCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Clone a DynamicCache onto independent GPU tensors for repeated restores."""
    snapshot: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_id, layer in enumerate(cache.layers):
        keys, values = getattr(layer, "keys", None), getattr(layer, "values", None)
        if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
            raise ValueError(f"DynamicCache layer {layer_id} is not initialized")
        snapshot.append((keys.clone(), values.clone()))
    return tuple(snapshot)


def snapshot_to_cpu(
    snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Move an immutable BF16 KV snapshot to CPU for sample checkpointing."""
    return tuple(
        (
            keys.detach().to(device="cpu", dtype=torch.bfloat16).clone(),
            values.detach().to(device="cpu", dtype=torch.bfloat16).clone(),
        )
        for keys, values in snapshot
    )


def snapshot_to_cuda(
    snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Upload a checkpointed BF16 KV snapshot before DynamicCache restoration."""
    return tuple(
        (
            keys.to(device="cuda", non_blocking=True),
            values.to(device="cuda", non_blocking=True),
        )
        for keys, values in snapshot
    )


def restore_dynamic_cache(
    config: transformers.PretrainedConfig,
    snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> DynamicCache:
    """Restore an independent DynamicCache using the installed DDP-data API."""
    restored = DynamicCache(
        ddp_cache_data=((keys.clone(), values.clone()) for keys, values in snapshot),
        config=config,
    )
    if restored.get_seq_length() != snapshot[0][0].shape[-2]:
        raise AssertionError("restored DynamicCache sequence length differs from its snapshot")
    for layer_id, (keys, values) in enumerate(snapshot):
        restored_layer = restored.layers[layer_id]
        if restored_layer.keys.shape != keys.shape or restored_layer.values.shape != values.shape:
            raise AssertionError(f"restored DynamicCache layer {layer_id} shape differs from its snapshot")
        if not torch.equal(restored_layer.keys, keys) or not torch.equal(restored_layer.values, values):
            raise AssertionError(f"restored DynamicCache layer {layer_id} values differ from its snapshot")
    return restored


def _nll(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.float(), target.reshape(-1), reduction="mean").item())


def _logit_kl(fp_logits_cpu: torch.Tensor, q_logits: torch.Tensor) -> float:
    fp_logits = fp_logits_cpu.to(device=q_logits.device, dtype=torch.float32, non_blocking=True)
    fp_log_probs = F.log_softmax(fp_logits, dim=-1)
    fp_probs = fp_log_probs.exp()
    q_log_probs = F.log_softmax(q_logits.float(), dim=-1)
    return float((fp_probs * (fp_log_probs - q_log_probs)).sum().item())


def _hidden_divergence(
    fp_hidden: torch.Tensor | None,
    q_hidden: torch.Tensor | None,
) -> tuple[float, float]:
    if fp_hidden is None or q_hidden is None:
        return math.nan, math.nan
    fp = fp_hidden.float().reshape(-1)
    q = q_hidden.float().reshape(-1)
    relative_l2 = torch.linalg.vector_norm(q - fp) / torch.linalg.vector_norm(fp).clamp_min(1e-12)
    cosine_distance = 1.0 - F.cosine_similarity(fp, q, dim=0, eps=1e-12)
    return float(relative_l2.item()), float(cosine_distance.item())


def _route_drift(fp: RouterTrace, q: RouterTrace) -> float:
    return 1.0 - len({fp.top1, fp.top2}.intersection({q.top1, q.top2})) / 2.0


def _router_logit_divergence(fp: RouterTrace, q: RouterTrace) -> tuple[float, float]:
    fp_logits = fp.logits_cpu.float()
    q_logits = q.logits_cpu.float()
    relative_l2 = torch.linalg.vector_norm(q_logits - fp_logits) / torch.linalg.vector_norm(fp_logits).clamp_min(1e-12)
    cosine_distance = 1.0 - F.cosine_similarity(fp_logits, q_logits, dim=0, eps=1e-12)
    return float(relative_l2.item()), max(0.0, float(cosine_distance.item()))


def _assert_finite(logits: torch.Tensor, label: str) -> None:
    if not torch.isfinite(logits).all():
        raise FloatingPointError(f"NaN/Inf logits during {label}")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_fp_checkpoint(
    path: Path,
    snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    fp_trace: list[FPTokenTrace],
) -> None:
    """Atomically persist BF16-only state needed for the later Q phase."""
    payload = {
        "snapshot": snapshot_to_cpu(snapshot),
        "fp_trace": [
            {
                "nll": token.nll,
                "logits_cpu": token.logits_cpu,
                "routes": [
                    {
                        "top1": route.top1,
                        "top2": route.top2,
                        "margin": route.margin,
                        "logits_cpu": route.logits_cpu,
                    }
                    for route in token.step.routes
                ],
                "hidden": token.step.hidden,
            }
            for token in fp_trace
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_fp_checkpoint(
    path: Path,
) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], list[FPTokenTrace]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    snapshot = tuple((keys, values) for keys, values in payload["snapshot"])
    trace: list[FPTokenTrace] = []
    for token in payload["fp_trace"]:
        routes = [
            RouterTrace(
                top1=int(route["top1"]),
                top2=int(route["top2"]),
                margin=float(route["margin"]),
                logits_cpu=route["logits_cpu"],
            )
            for route in token["routes"]
        ]
        trace.append(
            FPTokenTrace(
                nll=float(token["nll"]),
                logits_cpu=token["logits_cpu"],
                step=StepTrace(routes=routes, hidden=list(token["hidden"])),
            )
        )
    return snapshot, trace


class SampleCSVWriter:
    """Write one sample to a temp file and atomically publish it when complete."""

    def __init__(self, path: Path, fieldnames: tuple[str, ...]) -> None:
        self.path = path
        self.temporary = path.with_suffix(path.suffix + ".tmp")
        self.handle = self.temporary.open("w", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)

    def commit(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        self.handle.close()
        if self.temporary.exists():
            self.temporary.unlink()


def _rebuild_combined_csv(
    output_path: Path,
    sample_paths: Iterable[Path],
    fieldnames: tuple[str, ...],
) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for sample_path in sample_paths:
            with sample_path.open(newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output_path)


def _summary_rows(token_path: Path) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    with token_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault((row["sample_id"], row["quant_bits"]), []).append(row)
    rows: list[dict[str, object]] = []
    for (sample_id, quant_bits), values in sorted(groups.items()):
        def average(field: str) -> float:
            numeric = [float(row[field]) for row in values if not math.isnan(float(row[field]))]
            return sum(numeric) / len(numeric) if numeric else math.nan

        mean_delta_nll = average("delta_nll")
        rows.append(
            {
                "sample_id": sample_id,
                "quant_bits": quant_bits,
                "decode_tokens": len(values),
                "mean_fp_nll": average("fp_nll"),
                "mean_q_nll": average("q_nll"),
                "mean_delta_nll": mean_delta_nll,
                "ppl_ratio": math.exp(mean_delta_nll),
                "mean_routing_drift": average("routing_drift_mean"),
                "mean_router_logit_rel_l2": average("router_logit_rel_l2_mean"),
                "mean_router_logit_cosine_distance": average("router_logit_cosine_distance_mean"),
                "mean_hidden_rel_l2": average("hidden_rel_l2_mean"),
                "mean_logit_kl": average("logit_kl"),
            }
        )
    return rows


def _write_summary(output_dir: Path) -> None:
    rows = _summary_rows(output_dir / "token_metrics.csv")
    path = output_dir / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    fields = tuple(rows[0]) if rows else ("sample_id",)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_corpus_summary(output_dir: Path) -> dict[str, object]:
    """Aggregate token NLL before exponentiating, as required for corpus PPL."""
    token_path = output_dir / "token_metrics.csv"
    with token_path.open(newline="") as handle:
        values = list(csv.DictReader(handle))
    if not values:
        raise ValueError("cannot calculate corpus PPL from an empty token_metrics.csv")
    quant_bits = {row["quant_bits"] for row in values}
    if len(quant_bits) != 1:
        raise ValueError(f"corpus summary expects one quantization setting, got {sorted(quant_bits)}")
    sample_ids = {row["sample_id"] for row in values}
    mean_fp_nll = sum(float(row["fp_nll"]) for row in values) / len(values)
    mean_q_nll = sum(float(row["q_nll"]) for row in values) / len(values)
    mean_delta_nll = mean_q_nll - mean_fp_nll
    row: dict[str, object] = {
        "quant_bits": quant_bits.pop(),
        "completed_samples": len(sample_ids),
        "scored_tokens": len(values),
        "mean_fp_nll": mean_fp_nll,
        "fp_ppl": math.exp(mean_fp_nll),
        "mean_q_nll": mean_q_nll,
        "q_ppl": math.exp(mean_q_nll),
        "mean_delta_nll": mean_delta_nll,
        "ppl_ratio": math.exp(mean_delta_nll),
    }
    path = output_dir / "corpus_summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_FIELDS)
        writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return row


class QualityRuntime:
    def __init__(self, model_path: str, cache_slots: int) -> None:
        if cache_slots < HostExpertStore.NUM_LAYERS:
            raise ValueError(f"fixed-per-layer cache needs at least {HostExpertStore.NUM_LAYERS} slots")
        print(f"Loading model on CPU: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.store = HostExpertStore(self.model)
        move_non_expert_modules_to_cuda(self.model)
        self.cache = GPUExpertCache(
            self.store,
            capacity_slots=HostExpertStore.NUM_LAYERS,
            cache_policy=FIXED_PER_LAYER_LRU,
        )
        replace_with_cached_offloaded_experts(self.model, self.cache)
        self.cache.set_capacity_slots(cache_slots)
        self.cache.clear()
        if not self.store.all_experts_pinned():
            raise AssertionError("quality runtime requires fully pinned host experts")
        print(f"Pinned host experts: {self.store.total_size_bytes() / 1024**3:.2f} GiB")
        print(f"Expert GPU cache slots: {self.cache.capacity_slots}")
        self.hooks = TraceHooks(self.model)

    @torch.inference_mode()
    def prefill_snapshot(self, input_ids: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        cache = DynamicCache(config=self.model.config)
        self.cache.clear()
        output = self.model(input_ids=input_ids.to("cuda"), past_key_values=cache, use_cache=True, logits_to_keep=1)
        _assert_finite(output.logits, "BF16 prefill")
        snapshot = snapshot_dynamic_cache(cache)
        if cache.get_seq_length() != input_ids.shape[-1]:
            raise AssertionError("prefill KV length differs from prompt length")
        # Validate the concrete Transformers 5.15 restore path before using it.
        restore_dynamic_cache(self.model.config, snapshot)
        return snapshot

    @torch.inference_mode()
    def collect_fp_trace(
        self,
        sequence: torch.Tensor,
        snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        prefill_tokens: int,
        decode_tokens: int,
        hidden_stride: int,
    ) -> list[FPTokenTrace]:
        cache = restore_dynamic_cache(self.model.config, snapshot)
        self.cache.clear()
        traces: list[FPTokenTrace] = []
        for position in range(1, decode_tokens + 1):
            input_ids = sequence[:, prefill_tokens + position - 1 : prefill_tokens + position].to("cuda")
            target = sequence[:, prefill_tokens + position].to("cuda")
            self.hooks.begin(capture_hidden=(position - 1) % hidden_stride == 0)
            output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            step = self.hooks.finish()
            logits = output.logits[:, -1, :]
            _assert_finite(logits, f"BF16 decode position {position}")
            traces.append(
                FPTokenTrace(
                    nll=_nll(logits, target),
                    logits_cpu=logits[0].detach().to(device="cpu", dtype=torch.bfloat16).clone(),
                    step=step,
                )
            )
        return traces

    @torch.inference_mode()
    def deterministic_replay(
        self,
        sequence: torch.Tensor,
        snapshot: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        prefill_tokens: int,
        tokens: int,
    ) -> list[tuple[float, tuple[tuple[int, int], ...]]]:
        cache = restore_dynamic_cache(self.model.config, snapshot)
        self.cache.clear()
        result: list[tuple[float, tuple[tuple[int, int], ...]]] = []
        for position in range(1, tokens + 1):
            input_ids = sequence[:, prefill_tokens + position - 1 : prefill_tokens + position].to("cuda")
            target = sequence[:, prefill_tokens + position].to("cuda")
            self.hooks.begin(capture_hidden=False)
            output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            step = self.hooks.finish()
            logits = output.logits[:, -1, :]
            result.append((_nll(logits, target), tuple((route.top1, route.top2) for route in step.routes)))
        return result

    @torch.inference_mode()
    def collect_quantized_rows(
        self,
        *,
        sample_id: str,
        quant_bits: int,
        sequence: torch.Tensor,
        snapshot_cpu: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        prefill_tokens: int,
        fp_trace: list[FPTokenTrace],
        hidden_stride: int,
        token_writer: SampleCSVWriter,
        layer_writer: SampleCSVWriter,
    ) -> dict[str, float]:
        snapshot = snapshot_to_cuda(snapshot_cpu)
        cache = restore_dynamic_cache(self.model.config, snapshot)
        if cache.get_seq_length() != prefill_tokens:
            raise AssertionError("quantized trajectory did not restore the BF16 prefill KV length")
        self.cache.clear()  # Discard all BF16 cache residency before the Q trajectory.
        maximums = {
            "abs_delta_nll": 0.0,
            "routing_drift": 0.0,
            "router_logit_rel_l2": 0.0,
            "hidden_rel_l2": 0.0,
            "logit_kl": 0.0,
        }
        for position, fp in enumerate(fp_trace, start=1):
            input_ids = sequence[:, prefill_tokens + position - 1 : prefill_tokens + position].to("cuda")
            target = sequence[:, prefill_tokens + position].to("cuda")
            if not torch.equal(input_ids.cpu(), sequence[:, prefill_tokens + position - 1 : prefill_tokens + position]):
                raise AssertionError("quantized trajectory input IDs changed")
            self.hooks.begin(capture_hidden=(position - 1) % hidden_stride == 0)
            output = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            q_step = self.hooks.finish()
            q_logits = output.logits[:, -1, :]
            _assert_finite(q_logits, f"quantized decode position {position}")
            q_nll = _nll(q_logits, target)
            route_drifts: list[float] = []
            router_logit_l2: list[float] = []
            router_logit_cosine: list[float] = []
            hidden_l2: list[float] = []
            hidden_cosine: list[float] = []
            for layer_id, (fp_route, q_route) in enumerate(zip(fp.step.routes, q_step.routes)):
                route_drift = _route_drift(fp_route, q_route)
                router_rel_l2, router_cosine_distance = _router_logit_divergence(fp_route, q_route)
                if layer_id == 0 and route_drift != 0.0:
                    raise AssertionError(
                        "layer-0 routing drift must be exactly zero for expert-only quantization: "
                        f"sample_id={sample_id}, decode_position={position}, "
                        f"fp_top2=({fp_route.top1}, {fp_route.top2}), "
                        f"q_top2=({q_route.top1}, {q_route.top2})"
                    )
                if layer_id == 0 and router_rel_l2 > 1e-7:
                    raise AssertionError(
                        "layer-0 router logits must be unchanged for expert-only quantization: "
                        f"sample_id={sample_id}, decode_position={position}, "
                        f"relative_l2={router_rel_l2:.3e}"
                    )
                rel_l2, cosine_distance = _hidden_divergence(fp.step.hidden[layer_id], q_step.hidden[layer_id])
                route_drifts.append(route_drift)
                router_logit_l2.append(router_rel_l2)
                router_logit_cosine.append(router_cosine_distance)
                if not math.isnan(rel_l2):
                    hidden_l2.append(rel_l2)
                    hidden_cosine.append(cosine_distance)
                layer_writer.write(
                    {
                        "sample_id": sample_id,
                        "quant_bits": quant_bits,
                        "decode_position": position,
                        "layer_id": layer_id,
                        "routing_drift": route_drift,
                        "fp_top1": fp_route.top1,
                        "fp_top2": fp_route.top2,
                        "q_top1": q_route.top1,
                        "q_top2": q_route.top2,
                        "router_margin_fp": fp_route.margin,
                        "router_logit_rel_l2": router_rel_l2,
                        "router_logit_cosine_distance": router_cosine_distance,
                        "hidden_rel_l2": rel_l2,
                        "hidden_cosine_distance": cosine_distance,
                    }
                )
            logit_kl = _logit_kl(fp.logits_cpu, q_logits[0])
            delta_nll = q_nll - fp.nll
            maximums["abs_delta_nll"] = max(maximums["abs_delta_nll"], abs(delta_nll))
            maximums["routing_drift"] = max(maximums["routing_drift"], max(route_drifts))
            maximums["router_logit_rel_l2"] = max(maximums["router_logit_rel_l2"], max(router_logit_l2))
            maximums["hidden_rel_l2"] = max(
                maximums["hidden_rel_l2"], max(hidden_l2) if hidden_l2 else 0.0
            )
            maximums["logit_kl"] = max(maximums["logit_kl"], abs(logit_kl))
            token_writer.write(
                {
                    "sample_id": sample_id,
                    "quant_bits": quant_bits,
                    "decode_position": position,
                    "fp_nll": fp.nll,
                    "q_nll": q_nll,
                    "delta_nll": delta_nll,
                    "routing_drift_mean": sum(route_drifts) / len(route_drifts),
                    "router_logit_rel_l2_mean": sum(router_logit_l2) / len(router_logit_l2),
                    "router_logit_cosine_distance_mean": sum(router_logit_cosine) / len(router_logit_cosine),
                    "hidden_rel_l2_mean": sum(hidden_l2) / len(hidden_l2) if hidden_l2 else math.nan,
                    "hidden_cosine_distance_mean": sum(hidden_cosine) / len(hidden_cosine) if hidden_cosine else math.nan,
                    "logit_kl": logit_kl,
                }
            )
            if cache.get_seq_length() != prefill_tokens + position:
                raise AssertionError(
                    f"quantized KV length {cache.get_seq_length()} does not equal {prefill_tokens + position}"
                )
        return maximums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quant-bits", type=int, choices=(3, 4, 8, 16), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--quant-row-chunk-size", type=int, default=128)
    parser.add_argument("--prefill-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=8192)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--hidden-stride", type=int, default=32)
    parser.add_argument("--sanity-tokens", type=int, default=16)
    parser.add_argument("--cache-slots", type=int, default=48)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-max-scan", type=int, default=1024)
    parser.add_argument(
        "--require-full-decode",
        action="store_true",
        help="Skip documents that cannot supply the requested --decode-tokens (instead of scoring their shorter suffix).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return "unknown"


RESUME_FIELDS = (
    "model_path",
    "quant_bits",
    "group_size",
    "prefill_tokens",
    "decode_tokens",
    "dataset",
    "dataset_split",
    "sample_seed",
    "hidden_stride",
    "cache_slots",
    "require_full_decode",
)


def experiment_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_path": args.model,
        "quant_bits": args.quant_bits,
        "group_size": args.group_size,
        "prefill_tokens": args.prefill_tokens,
        "decode_tokens": args.decode_tokens,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "sample_seed": args.sample_seed,
        "hidden_stride": args.hidden_stride,
        "cache_slots": args.cache_slots,
        "require_full_decode": args.require_full_decode,
    }


def validate_resume_metadata(metadata: dict[str, Any], args: argparse.Namespace) -> None:
    expected = experiment_config(args)
    if metadata.get("quality_drift_version") != 3:
        raise ValueError(
            "output directory contains results from an older/incompatible quality benchmark; "
            "choose a new --output-dir"
        )
    for field in RESUME_FIELDS:
        if metadata.get(field) != expected[field]:
            raise ValueError(
                f"cannot resume mixed experiment in {args.output_dir}: "
                f"metadata {field}={metadata.get(field)!r}, requested {expected[field]!r}"
            )


def assert_bf16_sanity(maximums: dict[str, float], sample_id: str) -> None:
    tolerances = {
        "abs_delta_nll": 1e-6,
        "routing_drift": 0.0,
        "router_logit_rel_l2": 1e-7,
        "hidden_rel_l2": 1e-7,
        "logit_kl": 1e-6,
    }
    for field, tolerance in tolerances.items():
        if maximums[field] > tolerance:
            raise AssertionError(
                f"BF16 sanity failed for {sample_id}: {field}={maximums[field]:.3e} exceeds {tolerance:.3e}"
            )


def main() -> None:
    args = parse_args()
    if args.prefill_tokens + args.decode_tokens + 1 > 32768:
        raise ValueError("prefill_tokens + decode_tokens + 1 exceeds Mixtral's 32768-token context window")
    if args.hidden_stride < 1 or args.sanity_tokens < 1:
        raise ValueError("hidden_stride and sanity_tokens must be positive")
    if args.sanity_tokens > args.decode_tokens:
        raise ValueError("sanity_tokens cannot exceed decode_tokens")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.sample_seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists():
        metadata: dict[str, Any] = json.loads(metadata_path.read_text())
        validate_resume_metadata(metadata, args)
    else:
        metadata = {
            "quality_drift_version": 3,
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "quantization_method": "groupwise_symmetric_rtn_fake_quant_bf16_reconstruction",
            **experiment_config(args),
            "bf16_completed_samples": [],
            "quantized_completed_samples": [],
        }

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = load_deterministic_pg19_samples(
        tokenizer,
        num_samples=args.num_samples,
        sample_seed=args.sample_seed,
        prefill_tokens=args.prefill_tokens,
        max_decode_tokens=args.decode_tokens,
        dataset_name=args.dataset,
        split=args.dataset_split,
        max_scan=args.dataset_max_scan,
        allow_shorter_decode=not args.require_full_decode,
    )
    sample_ids = [sample.sample_id for sample in samples]
    existing_sample_ids = metadata.get("sample_ids")
    if existing_sample_ids is not None and existing_sample_ids != sample_ids:
        raise ValueError("deterministic sample selection differs from existing output metadata")
    metadata["sample_ids"] = sample_ids
    decode_lengths = [sample.decode_tokens for sample in samples]
    print(f"PG-19 samples (seed={args.sample_seed}): {metadata['sample_ids']}")
    print(
        f"Continuation tokens/sample: min={min(decode_lengths)}, max={max(decode_lengths)}, "
        f"total={sum(decode_lengths)}"
    )

    runtime = QualityRuntime(args.model, args.cache_slots)
    try:
        # Phase 1: every selected sample gets BF16 prefill and BF16 trace
        # before the shared HostExpertStore is ever quantized.
        bf16_completed: list[dict[str, Any]] = list(metadata.get("bf16_completed_samples", []))
        bf16_ids = {str(row["sample_id"]) for row in bf16_completed}
        for sample_index, sample in enumerate(samples):
            if sample.sample_id in bf16_ids:
                print(f"Skipping completed BF16 phase: {sample.sample_id}")
                continue
            sequence = sample.input_ids
            snapshot = runtime.prefill_snapshot(sequence[:, : args.prefill_tokens])
            replay_a = runtime.deterministic_replay(
                sequence, snapshot, args.prefill_tokens, args.sanity_tokens
            )
            replay_b = runtime.deterministic_replay(
                sequence, snapshot, args.prefill_tokens, args.sanity_tokens
            )
            if replay_a != replay_b:
                raise AssertionError("BF16 deterministic replay differs from the same initial KV snapshot")
            print(f"BF16 KV replay sanity: PASS ({args.sanity_tokens} teacher-forced tokens)")

            fp_trace = runtime.collect_fp_trace(
                sequence, snapshot, args.prefill_tokens, sample.decode_tokens, args.hidden_stride
            )
            fp_path = checkpoints / f"sample_{sample_index:04d}_bf16.pt"
            save_fp_checkpoint(fp_path, snapshot, fp_trace)
            bf16_completed.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_index": sample_index,
                    "fp_checkpoint": str(fp_path.relative_to(args.output_dir)),
                    "prefill_kv_length": args.prefill_tokens,
                    "decode_tokens": sample.decode_tokens,
                }
            )
            bf16_ids.add(sample.sample_id)
            metadata["bf16_completed_samples"] = bf16_completed
            _atomic_json(metadata_path, metadata)
            print(f"Checkpointed BF16 phase for {sample.sample_id}")
            del fp_trace, snapshot

        if len(bf16_completed) != len(samples):
            raise AssertionError("quantized phase cannot start before all BF16 sample traces are complete")

        # Phase 2 uses the same model instance. Quantize exactly once after all
        # BF16 snapshots/traces exist, then restore each sample's own BF16 KV.
        quantized_completed: list[dict[str, Any]] = list(metadata.get("quantized_completed_samples", []))
        quantized_ids = {str(row["sample_id"]) for row in quantized_completed}
        if len(quantized_ids) < len(samples):
            if args.quant_bits == 16:
                quant_stats = QuantizationStats()
                print("Quantization condition: BF16 reference (no weight perturbation)")
            else:
                print(f"Fake-quantizing all pinned experts once to W{args.quant_bits}A16")
                quant_stats = fake_quantize_expert_store_(
                    runtime.store,
                    bits=args.quant_bits,
                    group_size=args.group_size,
                    row_chunk_size=args.quant_row_chunk_size,
                )
                print("Quantization stats:", quant_stats.as_dict())
            metadata["quantization_stats"] = quant_stats.as_dict()
            _atomic_json(metadata_path, metadata)

        for sample_index, sample in enumerate(samples):
            if sample.sample_id in quantized_ids:
                print(f"Skipping completed quantized phase: {sample.sample_id}")
                continue
            fp_record = next(row for row in bf16_completed if row["sample_id"] == sample.sample_id)
            snapshot_cpu, fp_trace = load_fp_checkpoint(args.output_dir / fp_record["fp_checkpoint"])
            if len(fp_trace) != sample.decode_tokens:
                raise AssertionError("FP checkpoint decode length differs from the selected PG-19 sample")
            token_path = checkpoints / f"sample_{sample_index:04d}_tokens.csv"
            layer_path = checkpoints / f"sample_{sample_index:04d}_layers.csv"
            token_writer = SampleCSVWriter(token_path, TOKEN_FIELDS)
            layer_writer = SampleCSVWriter(layer_path, LAYER_FIELDS)
            try:
                maximums = runtime.collect_quantized_rows(
                    sample_id=sample.sample_id,
                    quant_bits=args.quant_bits,
                    sequence=sample.input_ids,
                    snapshot_cpu=snapshot_cpu,
                    prefill_tokens=args.prefill_tokens,
                    fp_trace=fp_trace,
                    hidden_stride=args.hidden_stride,
                    token_writer=token_writer,
                    layer_writer=layer_writer,
                )
                token_writer.commit()
                layer_writer.commit()
            except BaseException:
                token_writer.abort()
                layer_writer.abort()
                raise
            if args.quant_bits == 16:
                assert_bf16_sanity(maximums, sample.sample_id)
            quantized_completed.append(
                {
                    "sample_id": sample.sample_id,
                    "sample_index": sample_index,
                    "token_checkpoint": str(token_path.relative_to(args.output_dir)),
                    "layer_checkpoint": str(layer_path.relative_to(args.output_dir)),
                    "trajectory_maximums": maximums,
                }
            )
            quantized_ids.add(sample.sample_id)
            _rebuild_combined_csv(
                args.output_dir / "token_metrics.csv",
                [args.output_dir / row["token_checkpoint"] for row in quantized_completed],
                TOKEN_FIELDS,
            )
            _rebuild_combined_csv(
                args.output_dir / "layer_metrics.csv",
                [args.output_dir / row["layer_checkpoint"] for row in quantized_completed],
                LAYER_FIELDS,
            )
            _write_summary(args.output_dir)
            corpus = _write_corpus_summary(args.output_dir)
            metadata["quantized_completed_samples"] = quantized_completed
            metadata["corpus_summary"] = corpus
            _atomic_json(metadata_path, metadata)
            print(
                f"Checkpointed quantized phase for {sample.sample_id}; "
                f"corpus continuation PPL: FP={corpus['fp_ppl']:.4f}, "
                f"W{args.quant_bits}A16={corpus['q_ppl']:.4f}, ratio={corpus['ppl_ratio']:.4f}"
            )
            del snapshot_cpu, fp_trace
    finally:
        runtime.hooks.close()


if __name__ == "__main__":
    main()

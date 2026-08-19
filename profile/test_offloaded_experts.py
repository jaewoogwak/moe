"""Correctness test for the naive offloaded Mixtral expert executor."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offload.host_expert_store import HostExpertStore
from offload.offloaded_experts import OffloadedMixtralExperts


MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
TEST_CASES = ((0, 0), (0, 1), (10, 3), (31, 7))
ATOL = 1e-2
RTOL = 1e-2


def reference_expert_output(
    weights: tuple[torch.Tensor, torch.Tensor],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run one original CPU expert after temporarily copying its weights to CUDA."""
    gate_up_gpu = weights[0].to("cuda")
    down_gpu = weights[1].to("cuda")

    gate, up = F.linear(hidden_states, gate_up_gpu).chunk(2, dim=-1)
    output = F.linear(F.silu(gate) * up, down_gpu)

    # The returned output does not retain either copied weight tensor.
    del gate_up_gpu, down_gpu
    return output


def routed_single_expert(
    executor: OffloadedMixtralExperts,
    hidden_states: torch.Tensor,
    expert_id: int,
) -> torch.Tensor:
    top_k_index = torch.tensor([[expert_id]], dtype=torch.long, device="cuda")
    top_k_weights = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    return executor(hidden_states, top_k_index, top_k_weights)


def assert_reusable_buffer(
    executor: OffloadedMixtralExperts,
    hidden_states: torch.Tensor,
    expert_id: int,
) -> None:
    # The warm-up output is released before the allocation baseline is sampled.
    warmup_output = routed_single_expert(executor, hidden_states, expert_id)
    del warmup_output
    torch.cuda.synchronize()

    allocated_before = torch.cuda.memory_allocated()
    for _ in range(3):
        output = routed_single_expert(executor, hidden_states, expert_id)
        del output
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()

    print(f"GPU allocated before repeated execution: {allocated_before:,} bytes")
    print(f"GPU allocated after repeated execution:  {allocated_after:,} bytes")
    assert allocated_after == allocated_before, "repeated executions allocated new live GPU memory"


def main() -> None:
    model_path = os.environ.get("MIXTRAL_MODEL", MODEL)
    print(f"Loading Mixtral on CPU from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    # Keep only the selected original views for reference; HostExpertStore
    # releases model-owned pageable expert parameters after making pinned copies.
    reference_weights = {
        (layer_id, expert_id): (
            model.model.layers[layer_id].mlp.experts.gate_up_proj[expert_id],
            model.model.layers[layer_id].mlp.experts.down_proj[expert_id],
        )
        for layer_id, expert_id in TEST_CASES
    }
    store = HostExpertStore(model)
    assert store.all_experts_pinned()

    # This is intentionally a standalone layer-0 executor; do not replace the
    # full model before expert-level correctness has been established.
    hidden_states = torch.randn((1, 4096), dtype=torch.bfloat16, device="cuda")
    executors: dict[int, OffloadedMixtralExperts] = {}

    for layer_id, expert_id in TEST_CASES:
        if layer_id not in executors:
            executors[layer_id] = OffloadedMixtralExperts(
                layer_id=layer_id,
                host_store=store,
                device="cuda",
            )
        executor = executors[layer_id]

        reference = reference_expert_output(reference_weights[(layer_id, expert_id)], hidden_states)
        offloaded = routed_single_expert(executor, hidden_states, expert_id)
        torch.cuda.synchronize()

        difference = (reference.float() - offloaded.float()).abs()
        matches = torch.allclose(reference, offloaded, atol=ATOL, rtol=RTOL)
        print(f"Layer {layer_id}, expert {expert_id}")
        print(f"  max absolute difference:  {difference.max().item():.6f}")
        print(f"  mean absolute difference: {difference.mean().item():.6f}")
        print(f"  torch.allclose: {matches}")
        print(f"  output shape: {tuple(offloaded.shape)}")
        print(f"  output dtype: {offloaded.dtype}")
        assert matches, f"offloaded output differs for layer {layer_id}, expert {expert_id}"

        if (layer_id, expert_id) == (0, 0):
            buffer_mib = executor.buffer_size_bytes() / 1024**2
            print(f"Reusable GPU buffer size: {buffer_mib:.2f} MiB")
            assert_reusable_buffer(executor, hidden_states, expert_id)

        del reference, offloaded, difference

    print("Offloaded expert correctness: PASS")
    print("Reusable GPU buffer: PASS")
    print("No repeated GPU allocation growth: PASS")


if __name__ == "__main__":
    main()

"""Correctness smoke test for direct pinned-host Mixtral expert cache misses."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from offload.expert_cache import GLOBAL_LAYER_BALANCED_LRU, GPUExpertCache
from offload.host_expert_store import HostExpertStore


MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"
LAYER_ID = 0
EXPERT_ID = 0


def expert_forward(
    hidden_states: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    gate, up = F.linear(hidden_states, gate_up).chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, down)


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

    # Keep one original expert view solely as a bit-exact reference. The
    # HostExpertStore releases all model-owned pageable expert parameters.
    original_experts = model.model.layers[LAYER_ID].mlp.experts
    original_gate_up = original_experts.gate_up_proj[EXPERT_ID]
    original_down = original_experts.down_proj[EXPERT_ID]
    store = HostExpertStore(model)
    pinned_gate_up, pinned_down = store.get(LAYER_ID, EXPERT_ID)

    assert store.all_experts_pinned()
    assert pinned_gate_up.is_pinned() and pinned_down.is_pinned()
    assert torch.equal(pinned_gate_up, original_gate_up)
    assert torch.equal(pinned_down, original_down)
    print("Pinned store bit-exactness: PASS")
    print("All stored experts pinned: PASS")

    hidden_states = torch.randn((1, 4096), dtype=torch.bfloat16, device="cuda")
    reference_gate_up = original_gate_up.to("cuda")
    reference_down = original_down.to("cuda")
    reference = expert_forward(hidden_states, reference_gate_up, reference_down)
    del reference_gate_up, reference_down
    torch.cuda.synchronize()

    cache = GPUExpertCache(
        store,
        capacity_slots=1,
        cache_policy=GLOBAL_LAYER_BALANCED_LRU,
    )
    cached_gate_up, cached_down = cache.get(LAYER_ID, EXPERT_ID)
    cached_output = expert_forward(hidden_states, cached_gate_up, cached_down)
    torch.cuda.synchronize()
    assert torch.equal(reference, cached_output)
    assert cache.stats().misses == 1
    assert cache.stats().host_staging_ms == 0.0
    print("Pinned-host cache-miss expert output: PASS")

    del cached_output
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    for _ in range(3):
        gate_up, down = cache.get(LAYER_ID, EXPERT_ID)
        output = expert_forward(hidden_states, gate_up, down)
        del output
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated()
    assert allocated_after == allocated_before
    print("Repeated cache use has no GPU allocated-memory growth: PASS")


if __name__ == "__main__":
    main()

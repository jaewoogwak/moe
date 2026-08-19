"""Smoke test for the fully pinned HostExpertStore."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM

from offload.host_expert_store import HostExpertStore


MODEL_PATH = os.environ.get("MIXTRAL_MODEL", "/workspace/models/Mixtral-8x7B-Instruct-v0.1")


def format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB ({num_bytes:,} bytes)"


def main() -> None:
    print(f"Loading Mixtral on CPU from: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    )

    original_experts = model.model.layers[0].mlp.experts
    original_gate_up = original_experts.gate_up_proj[0]
    original_down = original_experts.down_proj[0]
    store = HostExpertStore(model)
    gate_up, down = store.get(0, 0)

    assert gate_up.device.type == "cpu"
    assert down.device.type == "cpu"
    assert gate_up.is_pinned()
    assert down.is_pinned()
    assert store.all_experts_pinned()
    assert torch.equal(gate_up, original_gate_up)
    assert torch.equal(down, original_down)
    assert store.pageable_expert_storage_released
    assert not hasattr(original_experts, "gate_up_proj")
    assert not hasattr(original_experts, "down_proj")

    print("Expert (layer=0, expert=0)")
    print(f"  gate_up: shape={tuple(gate_up.shape)}, dtype={gate_up.dtype}, device={gate_up.device}")
    print(f"  down:    shape={tuple(down.shape)}, dtype={down.dtype}, device={down.device}")
    print(f"  gate_up pinned: {gate_up.is_pinned()}")
    print(f"  down pinned:    {down.is_pinned()}")
    print(f"  gate_up shares storage with original: {original_gate_up.data_ptr() == gate_up.data_ptr()}")
    print(f"  down shares storage with original:    {original_down.data_ptr() == down.data_ptr()}")
    print(f"Stored experts: {store.num_experts()}")
    print(f"Expert size: {format_bytes(store.expert_size_bytes())}")
    print(f"Total size:  {format_bytes(store.total_size_bytes())}")
    print("All stored experts are pinned: passed")
    print("Bit-exact equality with original tensors: passed")
    print("Original pageable expert parameters released: passed")


if __name__ == "__main__":
    main()

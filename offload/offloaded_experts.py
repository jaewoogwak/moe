"""Mixtral expert executors backed by CPU-resident or cached GPU weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from offload.expert_cache import GPUExpertCache
    from offload.host_expert_store import HostExpertStore


class OffloadedMixtralExperts(nn.Module):
    """Naive single-buffer CPU-to-GPU expert executor."""

    GATE_UP_SHAPE = (28672, 4096)
    DOWN_SHAPE = (4096, 14336)
    DTYPE = torch.bfloat16

    def __init__(
        self,
        layer_id: int,
        host_store: "HostExpertStore",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError("layer_id must be an integer")
        if not 0 <= layer_id < host_store.NUM_LAYERS:
            raise IndexError(f"layer_id must be in [0, {host_store.NUM_LAYERS - 1}], got {layer_id}")

        self.layer_id = layer_id
        self.host_store = host_store
        self.num_experts = host_store.NUM_EXPERTS_PER_LAYER
        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("OffloadedMixtralExperts requires a CUDA device")

        self.register_buffer(
            "gate_up_buffer",
            torch.empty(self.GATE_UP_SHAPE, dtype=self.DTYPE, device=target_device),
            persistent=False,
        )
        self.register_buffer(
            "down_buffer",
            torch.empty(self.DOWN_SHAPE, dtype=self.DTYPE, device=target_device),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_id = int(expert_idx_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_id])
            current_state = hidden_states[token_idx]
            gate_up_cpu, down_cpu = self.host_store.get(self.layer_id, expert_id)
            self.gate_up_buffer.copy_(gate_up_cpu, non_blocking=True)
            self.down_buffer.copy_(down_cpu, non_blocking=True)
            gate, up = F.linear(current_state, self.gate_up_buffer).chunk(2, dim=-1)
            current_hidden_states = F.linear(F.silu(gate) * up, self.down_buffer)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    def buffer_size_bytes(self) -> int:
        return (
            self.gate_up_buffer.numel() * self.gate_up_buffer.element_size()
            + self.down_buffer.numel() * self.down_buffer.element_size()
        )


class CachedOffloadedMixtralExperts(nn.Module):
    """Current MixtralExperts semantics backed by a shared GPUExpertCache."""

    def __init__(self, layer_id: int, expert_cache: "GPUExpertCache") -> None:
        super().__init__()
        self.layer_id = layer_id
        self.expert_cache = expert_cache
        self.num_experts = expert_cache.host_store.NUM_EXPERTS_PER_LAYER
        self.profiler: object | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Mirror installed Transformers MixtralExperts.forward exactly."""
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_id = int(expert_idx_tensor[0].item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_id])
            current_state = hidden_states[token_idx]
            gate_up_gpu, down_gpu = self.expert_cache.get(
                self.layer_id,
                expert_id,
                profiler=self.profiler,
            )
            if self.profiler is None:
                gate, up = F.linear(current_state, gate_up_gpu).chunk(2, dim=-1)
                current_hidden_states = F.linear(F.silu(gate) * up, down_gpu)
            else:
                with self.profiler.cuda_section("expert_compute"):
                    gate, up = F.linear(current_state, gate_up_gpu).chunk(2, dim=-1)
                    current_hidden_states = F.linear(F.silu(gate) * up, down_gpu)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


def replace_with_offloaded_experts(
    model: torch.nn.Module,
    host_store: "HostExpertStore",
) -> torch.nn.Module:
    """Replace all original experts with naive per-layer staging-buffer executors."""
    for layer_id, layer in enumerate(model.model.layers):
        layer.mlp.experts = OffloadedMixtralExperts(layer_id, host_store, device="cuda")
    return model


def replace_with_cached_offloaded_experts(
    model: torch.nn.Module,
    expert_cache: "GPUExpertCache",
) -> torch.nn.Module:
    """Replace all original experts with executors sharing one GPU expert cache."""
    layers = model.model.layers
    if len(layers) != expert_cache.host_store.NUM_LAYERS:
        raise ValueError("model and HostExpertStore layer counts differ")
    for layer_id, layer in enumerate(layers):
        layer.mlp.experts = CachedOffloadedMixtralExperts(layer_id, expert_cache)
    return model

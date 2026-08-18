"""A simple BF16 GPU expert cache for Mixtral offloading experiments."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import torch

if TYPE_CHECKING:
    from offload.host_expert_store import HostExpertStore


ExpertKey = Tuple[int, int]
GPUExpertWeights = Tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class ExpertCacheStats:
    hits: int
    misses: int
    evictions: int
    h2d_bytes: int


class GPUExpertCache:
    """Cache BF16 Mixtral experts on one CUDA device.

    The cache is LRU within the most represented layer. This layer-aware
    variant prevents one layer with many recently used experts from evicting
    the only resident experts of every other layer.
    """

    def __init__(
        self,
        host_store: "HostExpertStore",
        capacity_slots: int,
        device: str | torch.device = "cuda",
    ) -> None:
        if capacity_slots < 1:
            raise ValueError("capacity_slots must be at least one")

        self.host_store = host_store
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPUExpertCache requires a CUDA device")

        self.capacity_slots = capacity_slots
        self._entries: OrderedDict[ExpertKey, GPUExpertWeights] = OrderedDict()
        self._layer_counts: Counter[int] = Counter()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._h2d_bytes = 0

    @property
    def expert_size_bytes(self) -> int:
        return self.host_store.expert_size_bytes()

    @property
    def resident_slots(self) -> int:
        return len(self._entries)

    @property
    def resident_bytes(self) -> int:
        return self.resident_slots * self.expert_size_bytes

    def stats(self) -> ExpertCacheStats:
        return ExpertCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            h2d_bytes=self._h2d_bytes,
        )

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._h2d_bytes = 0

    def clear(self) -> None:
        self._entries.clear()
        self._layer_counts.clear()

    def set_capacity_slots(self, capacity_slots: int) -> None:
        if capacity_slots < 1:
            raise ValueError("capacity_slots must be at least one")
        self.capacity_slots = capacity_slots
        while self.resident_slots > self.capacity_slots:
            self._evict_one()

    def get(
        self,
        layer_id: int,
        expert_id: int,
        profiler: object | None = None,
    ) -> GPUExpertWeights:
        key = (layer_id, expert_id)
        resident = self._entries.get(key)
        if resident is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return resident

        self._misses += 1
        if self.resident_slots >= self.capacity_slots:
            self._evict_one()

        gate_up_cpu, down_cpu = self.host_store.get(layer_id, expert_id)
        if profiler is None:
            gate_up_gpu, down_gpu = self._copy_to_gpu(gate_up_cpu, down_cpu)
        else:
            with profiler.cuda_section("expert_h2d"):
                gate_up_gpu, down_gpu = self._copy_to_gpu(gate_up_cpu, down_cpu)

        self._entries[key] = (gate_up_gpu, down_gpu)
        self._layer_counts[layer_id] += 1
        self._h2d_bytes += self.expert_size_bytes
        return gate_up_gpu, down_gpu

    def _copy_to_gpu(
        self,
        gate_up_cpu: torch.Tensor,
        down_cpu: torch.Tensor,
    ) -> GPUExpertWeights:
        gate_up_gpu = torch.empty_like(gate_up_cpu, device=self.device)
        down_gpu = torch.empty_like(down_cpu, device=self.device)
        gate_up_gpu.copy_(gate_up_cpu, non_blocking=True)
        down_gpu.copy_(down_cpu, non_blocking=True)
        return gate_up_gpu, down_gpu

    def _evict_one(self) -> None:
        if not self._entries:
            return

        largest_layer_size = max(self._layer_counts.values())
        victim_key = next(
            key for key in self._entries if self._layer_counts[key[0]] == largest_layer_size
        )
        del self._entries[victim_key]
        self._layer_counts[victim_key[0]] -= 1
        if self._layer_counts[victim_key[0]] == 0:
            del self._layer_counts[victim_key[0]]
        self._evictions += 1

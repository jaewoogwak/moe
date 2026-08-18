"""A fixed-slot BF16 GPU expert cache for Mixtral offloading experiments."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Deque, Tuple

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
    host_staging_ms: float


class GPUExpertCache:
    """Layer-aware LRU cache with preallocated GPU expert slots."""

    GATE_UP_SHAPE = (28672, 4096)
    DOWN_SHAPE = (4096, 14336)

    def __init__(
        self,
        host_store: "HostExpertStore",
        capacity_slots: int,
        device: str | torch.device = "cuda",
    ) -> None:
        self.host_store = host_store
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPUExpertCache requires a CUDA device")

        self._entries: OrderedDict[ExpertKey, int] = OrderedDict()
        self._layer_counts: Counter[int] = Counter()
        self._slots: list[GPUExpertWeights] = []
        self._free_slots: Deque[int] = deque()
        self._pinned_ready_event: torch.cuda.Event | None = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._h2d_bytes = 0
        self._host_staging_ms = 0.0

        # There is exactly one shared pinned staging expert.
        self.pinned_gate_up = torch.empty(
            self.GATE_UP_SHAPE,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self.pinned_down = torch.empty(
            self.DOWN_SHAPE,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self.set_capacity_slots(capacity_slots)

    @property
    def expert_size_bytes(self) -> int:
        return self.host_store.expert_size_bytes()

    @property
    def capacity_slots(self) -> int:
        return len(self._slots)

    @property
    def resident_slots(self) -> int:
        return len(self._entries)

    @property
    def allocated_bytes(self) -> int:
        return self.capacity_slots * self.expert_size_bytes

    @property
    def resident_bytes(self) -> int:
        return self.resident_slots * self.expert_size_bytes

    def stats(self) -> ExpertCacheStats:
        return ExpertCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            h2d_bytes=self._h2d_bytes,
            host_staging_ms=self._host_staging_ms,
        )

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._h2d_bytes = 0
        self._host_staging_ms = 0.0

    def clear(self) -> None:
        """Clear key metadata while retaining every preallocated GPU slot."""
        self._entries.clear()
        self._layer_counts.clear()
        self._free_slots = deque(range(self.capacity_slots))

    def set_capacity_slots(self, capacity_slots: int) -> None:
        """Allocate exactly capacity_slots reusable CUDA expert slots."""
        if capacity_slots < 1:
            raise ValueError("capacity_slots must be at least one")
        if capacity_slots == self.capacity_slots:
            self.clear()
            return

        # Resizing happens only outside measured decode. Existing tensors are
        # released here; eviction during decode only remaps metadata.
        self._entries.clear()
        self._layer_counts.clear()
        self._slots = [
            (
                torch.empty(self.GATE_UP_SHAPE, dtype=torch.bfloat16, device=self.device),
                torch.empty(self.DOWN_SHAPE, dtype=torch.bfloat16, device=self.device),
            )
            for _ in range(capacity_slots)
        ]
        self._free_slots = deque(range(capacity_slots))

    def get(
        self,
        layer_id: int,
        expert_id: int,
        profiler: object | None = None,
    ) -> GPUExpertWeights:
        key = (layer_id, expert_id)
        slot_id = self._entries.get(key)
        if slot_id is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._slots[slot_id]

        self._misses += 1
        slot_id = self._free_slots.popleft() if self._free_slots else self._evict_one()
        gate_up_cpu, down_cpu = self.host_store.get(layer_id, expert_id)

        # A pinned source cannot be overwritten until its previous async copy
        # has completed. This preserves one shared staging pair and a serialized
        # default-stream baseline without allocating a second staging expert.
        if self._pinned_ready_event is not None:
            self._pinned_ready_event.synchronize()

        staging_start = perf_counter()
        self.pinned_gate_up.copy_(gate_up_cpu)
        self.pinned_down.copy_(down_cpu)
        self._host_staging_ms += (perf_counter() - staging_start) * 1000

        gate_up_gpu, down_gpu = self._slots[slot_id]
        if profiler is None:
            gate_up_gpu.copy_(self.pinned_gate_up, non_blocking=True)
            down_gpu.copy_(self.pinned_down, non_blocking=True)
        else:
            # This event range includes only pinned CPU -> fixed GPU-slot copies.
            with profiler.cuda_section("expert_h2d"):
                gate_up_gpu.copy_(self.pinned_gate_up, non_blocking=True)
                down_gpu.copy_(self.pinned_down, non_blocking=True)

        self._pinned_ready_event = torch.cuda.Event()
        self._pinned_ready_event.record()
        self._entries[key] = slot_id
        self._layer_counts[layer_id] += 1
        self._h2d_bytes += self.expert_size_bytes
        return gate_up_gpu, down_gpu

    def _evict_one(self) -> int:
        """Evict metadata only and return the victim's reusable slot ID."""
        largest_layer_size = max(self._layer_counts.values())
        victim_key = next(
            key for key in self._entries if self._layer_counts[key[0]] == largest_layer_size
        )
        slot_id = self._entries.pop(victim_key)
        self._layer_counts[victim_key[0]] -= 1
        if self._layer_counts[victim_key[0]] == 0:
            del self._layer_counts[victim_key[0]]
        self._evictions += 1
        return slot_id

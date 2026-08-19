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

FIXED_PER_LAYER_LRU = "fixed_per_layer_lru"
GLOBAL_LAYER_BALANCED_LRU = "global_layer_balanced_lru"
VALID_CACHE_POLICIES = (FIXED_PER_LAYER_LRU, GLOBAL_LAYER_BALANCED_LRU)


@dataclass(frozen=True)
class ExpertCacheStats:
    hits: int
    misses: int
    evictions: int
    h2d_bytes: int
    host_staging_ms: float


class GPUExpertCache:
    """Preallocated BF16 cache with selectable fixed or global layer-aware LRU."""

    GATE_UP_SHAPE = (28672, 4096)
    DOWN_SHAPE = (4096, 14336)

    def __init__(
        self,
        host_store: "HostExpertStore",
        capacity_slots: int,
        device: str | torch.device = "cuda",
        cache_policy: str = FIXED_PER_LAYER_LRU,
    ) -> None:
        self.host_store = host_store
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPUExpertCache requires a CUDA device")
        if cache_policy not in VALID_CACHE_POLICIES:
            raise ValueError(f"cache_policy must be one of {VALID_CACHE_POLICIES}, got {cache_policy!r}")
        self.cache_policy = cache_policy

        # State retained unchanged for the legacy global layer-balanced policy.
        self._entries: OrderedDict[ExpertKey, int] = OrderedDict()
        self._layer_counts: Counter[int] = Counter()
        self._free_slots: Deque[int] = deque()

        # Fixed per-layer policy state. A slot's layer owner never changes
        # between set_capacity_slots calls, which occur outside measured decode.
        self._layer_capacities: tuple[int, ...] = ()
        self._per_layer_entries: list[OrderedDict[int, int]] = []
        self._per_layer_free_slots: list[Deque[int]] = []
        self._slot_owner_layers: list[int] = []
        self._slots_by_layer: list[tuple[int, ...]] = []

        self._slots: list[GPUExpertWeights] = []
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
        if self.cache_policy == FIXED_PER_LAYER_LRU:
            return sum(len(entries) for entries in self._per_layer_entries)
        return len(self._entries)

    @property
    def allocated_bytes(self) -> int:
        return self.capacity_slots * self.expert_size_bytes

    @property
    def resident_bytes(self) -> int:
        return self.resident_slots * self.expert_size_bytes

    @property
    def layer_capacities(self) -> tuple[int, ...] | None:
        """Fixed quotas for the selected policy, or None for global LRU."""
        if self.cache_policy != FIXED_PER_LAYER_LRU:
            return None
        return self._layer_capacities

    @staticmethod
    def fixed_layer_capacities(total_slots: int, num_layers: int) -> tuple[int, ...]:
        """Evenly distribute slots across depth without front-loading extras."""
        if total_slots < 1:
            raise ValueError("total_slots must be at least one")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        base, remainder = divmod(total_slots, num_layers)
        capacities = tuple(
            base + ((layer_id + 1) * remainder) // num_layers - (layer_id * remainder) // num_layers
            for layer_id in range(num_layers)
        )
        if sum(capacities) != total_slots or max(capacities) - min(capacities) > 1:
            raise AssertionError("invalid fixed per-layer cache allocation")
        return capacities

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
        """Clear residency/LRU metadata while retaining all GPU slot tensors."""
        self._entries.clear()
        self._layer_counts.clear()
        if self.cache_policy == FIXED_PER_LAYER_LRU:
            for entries in self._per_layer_entries:
                entries.clear()
            self._per_layer_free_slots = [deque(slot_ids) for slot_ids in self._slots_by_layer]
            self._free_slots.clear()
            self._assert_fixed_per_layer_invariants()
        else:
            self._free_slots = deque(range(self.capacity_slots))

    def set_capacity_slots(self, capacity_slots: int) -> None:
        """Resize reusable CUDA slots in place without a duplicate full cache."""
        if capacity_slots < 1:
            raise ValueError("capacity_slots must be at least one")
        current_capacity = self.capacity_slots
        if capacity_slots == current_capacity:
            self.clear()
            return

        # Resizing happens only outside measured decode. Discard residency but
        # retain physical tensors that remain within the requested capacity.
        if current_capacity:
            self.clear()
        if capacity_slots > current_capacity:
            self._slots.extend(
                (
                    torch.empty(self.GATE_UP_SHAPE, dtype=torch.bfloat16, device=self.device),
                    torch.empty(self.DOWN_SHAPE, dtype=torch.bfloat16, device=self.device),
                )
                for _ in range(capacity_slots - current_capacity)
            )
        else:
            del self._slots[capacity_slots:]
            # Tail tensors have no references after deletion. This is outside
            # measured decode and returns cached allocator blocks before a
            # later resize may require a large allocation.
            torch.cuda.empty_cache()

        if self.capacity_slots != capacity_slots:
            raise AssertionError("in-place cache resize did not reach requested capacity")
        if self.cache_policy == FIXED_PER_LAYER_LRU:
            self._rebuild_fixed_per_layer_slots()
        else:
            self._free_slots = deque(range(capacity_slots))

    def get(
        self,
        layer_id: int,
        expert_id: int,
        profiler: object | None = None,
    ) -> GPUExpertWeights:
        self._validate_indices(layer_id, expert_id)
        if self.cache_policy == FIXED_PER_LAYER_LRU:
            slot_id = self._get_fixed_per_layer_slot(layer_id, expert_id)
        else:
            slot_id = self._get_global_layer_balanced_slot(layer_id, expert_id)
        if slot_id is not None:
            return self._slots[slot_id]

        gate_up_cpu, down_cpu = self.host_store.get(layer_id, expert_id)
        if self._pinned_ready_event is not None:
            self._pinned_ready_event.synchronize()

        staging_start = perf_counter()
        self.pinned_gate_up.copy_(gate_up_cpu)
        self.pinned_down.copy_(down_cpu)
        self._host_staging_ms += (perf_counter() - staging_start) * 1000

        gate_up_gpu, down_gpu = self._slots[self._pending_slot_id]
        if profiler is None:
            gate_up_gpu.copy_(self.pinned_gate_up, non_blocking=True)
            down_gpu.copy_(self.pinned_down, non_blocking=True)
        else:
            with profiler.cuda_section("expert_h2d"):
                gate_up_gpu.copy_(self.pinned_gate_up, non_blocking=True)
                down_gpu.copy_(self.pinned_down, non_blocking=True)

        self._pinned_ready_event = torch.cuda.Event()
        self._pinned_ready_event.record()
        self._install_pending_entry(layer_id, expert_id)
        self._h2d_bytes += self.expert_size_bytes
        return gate_up_gpu, down_gpu

    def _get_fixed_per_layer_slot(self, layer_id: int, expert_id: int) -> int | None:
        entries = self._per_layer_entries[layer_id]
        slot_id = entries.get(expert_id)
        if slot_id is not None:
            self._hits += 1
            entries.move_to_end(expert_id)
            return slot_id

        self._misses += 1
        if self._per_layer_free_slots[layer_id]:
            self._pending_slot_id = self._per_layer_free_slots[layer_id].popleft()
        else:
            if self._layer_capacities[layer_id] == 0:
                raise RuntimeError(
                    f"fixed per-layer policy has zero slots for active layer {layer_id}; "
                    "use at least one slot per Mixtral layer"
                )
            victim_expert_id, self._pending_slot_id = entries.popitem(last=False)
            if self._slot_owner_layers[self._pending_slot_id] != layer_id:
                raise AssertionError("fixed per-layer eviction selected a slot from another layer")
            if victim_expert_id == expert_id:
                raise AssertionError("cache miss cannot evict the requested expert")
            self._evictions += 1
        return None

    def _get_global_layer_balanced_slot(self, layer_id: int, expert_id: int) -> int | None:
        key = (layer_id, expert_id)
        slot_id = self._entries.get(key)
        if slot_id is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return slot_id

        self._misses += 1
        self._pending_slot_id = self._free_slots.popleft() if self._free_slots else self._evict_one_global()
        return None

    def _install_pending_entry(self, layer_id: int, expert_id: int) -> None:
        if self.cache_policy == FIXED_PER_LAYER_LRU:
            if self._slot_owner_layers[self._pending_slot_id] != layer_id:
                raise AssertionError("fixed per-layer miss attempted to install into another layer's slot")
            entries = self._per_layer_entries[layer_id]
            if len(entries) >= self._layer_capacities[layer_id]:
                raise AssertionError("fixed per-layer cache exceeds its assigned quota")
            entries[expert_id] = self._pending_slot_id
            self._assert_fixed_per_layer_invariants()
        else:
            key = (layer_id, expert_id)
            self._entries[key] = self._pending_slot_id
            self._layer_counts[layer_id] += 1

    def _evict_one_global(self) -> int:
        """Preserve the legacy global layer-balanced LRU eviction behavior."""
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

    def _rebuild_fixed_per_layer_slots(self) -> None:
        num_layers = self.host_store.NUM_LAYERS
        self._layer_capacities = self.fixed_layer_capacities(self.capacity_slots, num_layers)
        slots_by_layer: list[tuple[int, ...]] = []
        slot_owner_layers: list[int] = []
        slot_id = 0
        for layer_id, capacity in enumerate(self._layer_capacities):
            layer_slots = tuple(range(slot_id, slot_id + capacity))
            slots_by_layer.append(layer_slots)
            slot_owner_layers.extend([layer_id] * capacity)
            slot_id += capacity
        self._slots_by_layer = slots_by_layer
        self._slot_owner_layers = slot_owner_layers
        self._per_layer_entries = [OrderedDict() for _ in range(num_layers)]
        self._per_layer_free_slots = [deque(slot_ids) for slot_ids in slots_by_layer]
        self._free_slots.clear()
        self._assert_fixed_per_layer_invariants()

    def _assert_fixed_per_layer_invariants(self) -> None:
        if self.cache_policy != FIXED_PER_LAYER_LRU:
            return
        if len(self._layer_capacities) != self.host_store.NUM_LAYERS:
            raise AssertionError("fixed per-layer capacities do not cover every layer")
        if sum(self._layer_capacities) != self.capacity_slots:
            raise AssertionError("fixed per-layer capacities do not sum to total slots")
        if len(self._slot_owner_layers) != self.capacity_slots:
            raise AssertionError("every physical slot must have exactly one layer owner")
        for layer_id, entries in enumerate(self._per_layer_entries):
            if len(entries) > self._layer_capacities[layer_id]:
                raise AssertionError("layer cache residency exceeds its fixed quota")
            expected_slots = set(self._slots_by_layer[layer_id])
            if any(slot_id not in expected_slots for slot_id in entries.values()):
                raise AssertionError("layer cache entry owns a slot assigned to another layer")
            if any(slot_id not in expected_slots for slot_id in self._per_layer_free_slots[layer_id]):
                raise AssertionError("layer free list contains another layer's slot")
            if len(entries) + len(self._per_layer_free_slots[layer_id]) != self._layer_capacities[layer_id]:
                raise AssertionError("layer quota does not match resident plus free slots")

    def _validate_indices(self, layer_id: int, expert_id: int) -> None:
        if not 0 <= layer_id < self.host_store.NUM_LAYERS:
            raise IndexError(f"layer_id must be in [0, {self.host_store.NUM_LAYERS - 1}], got {layer_id}")
        if not 0 <= expert_id < self.host_store.NUM_EXPERTS_PER_LAYER:
            raise IndexError(
                f"expert_id must be in [0, {self.host_store.NUM_EXPERTS_PER_LAYER - 1}], got {expert_id}"
            )

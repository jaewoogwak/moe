"""GPU-free structural tests for in-place GPUExpertCache resizing."""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from types import SimpleNamespace
from unittest.mock import patch

from offload.expert_cache import FIXED_PER_LAYER_LRU, GPUExpertCache


def make_cache() -> GPUExpertCache:
    """Build cache metadata without allocating real CUDA tensors."""
    cache = object.__new__(GPUExpertCache)
    cache.host_store = SimpleNamespace(NUM_LAYERS=32, NUM_EXPERTS_PER_LAYER=8)
    cache.device = "cuda"
    cache.cache_policy = FIXED_PER_LAYER_LRU
    cache._entries = OrderedDict()
    cache._layer_counts = Counter()
    cache._free_slots = deque()
    cache._layer_capacities = ()
    cache._per_layer_entries = []
    cache._per_layer_free_slots = []
    cache._slot_owner_layers = []
    cache._slots_by_layer = []
    cache._slots = []
    return cache


def assert_resize_invariants(cache: GPUExpertCache, capacity: int) -> None:
    assert cache.capacity_slots == capacity
    assert cache.resident_slots == 0
    assert cache.layer_capacities is not None
    assert sum(cache.layer_capacities) == capacity
    assert len(cache._slot_owner_layers) == capacity
    flattened_slot_ids = [slot_id for slots in cache._slots_by_layer for slot_id in slots]
    assert sorted(flattened_slot_ids) == list(range(capacity))
    assert len(flattened_slot_ids) == len(set(flattened_slot_ids))
    cache._assert_fixed_per_layer_invariants()


def main() -> None:
    cache = make_cache()
    allocated_slot_objects: list[object] = []

    def fake_empty(*_: object, **__: object) -> object:
        slot = object()
        allocated_slot_objects.append(slot)
        return slot

    with patch("offload.expert_cache.torch.empty", side_effect=fake_empty), patch(
        "offload.expert_cache.torch.cuda.empty_cache"
    ) as empty_cache:
        cache.set_capacity_slots(32)
        assert_resize_invariants(cache, 32)
        original_slots = list(cache._slots)
        assert len(allocated_slot_objects) == 64

        cache.set_capacity_slots(53)
        assert_resize_invariants(cache, 53)
        assert cache._slots[:32] == original_slots
        # Each expert slot owns two tensors. Growing 32 -> 53 therefore
        # allocates 21 * 2 tensors, rather than another complete 53-slot cache.
        assert len(allocated_slot_objects) == 64 + 21 * 2

        for capacity in (48, 42, 36, 32):
            cache.set_capacity_slots(capacity)
            assert_resize_invariants(cache, capacity)
        assert empty_cache.call_count == 4

    print("in-place resize transitions and quota invariants: PASS")


if __name__ == "__main__":
    main()

"""CPU-resident storage for Mixtral expert weights.

The store retains views into a CPU-loaded Hugging Face Mixtral model. It does
not copy weights or move tensors between devices.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


ExpertWeights = Tuple[torch.Tensor, torch.Tensor]


class HostExpertStore:
    """Index CPU-resident Mixtral expert weights by layer and expert ID.

    The tensors stored here are views of the model parameters, so the model must
    remain alive for as long as this store is used.
    """

    NUM_LAYERS = 32
    NUM_EXPERTS_PER_LAYER = 8
    GATE_UP_SHAPE = (28672, 4096)
    DOWN_SHAPE = (4096, 14336)
    DTYPE = torch.bfloat16

    def __init__(self, model: torch.nn.Module) -> None:
        self._experts: Dict[Tuple[int, int], ExpertWeights] = {}
        self._validate_model(model)

        for layer_id, layer in enumerate(model.model.layers):
            experts = layer.mlp.experts
            gate_up_proj = experts.gate_up_proj
            down_proj = experts.down_proj

            for expert_id in range(self.NUM_EXPERTS_PER_LAYER):
                # Tensor indexing returns a view, so this does not duplicate
                # checkpoint data.
                self._experts[(layer_id, expert_id)] = (
                    gate_up_proj[expert_id],
                    down_proj[expert_id],
                )

        gate_up, down = self._experts[(0, 0)]
        self._expert_size_bytes = (gate_up.numel() + down.numel()) * gate_up.element_size()

    def get(self, layer_id: int, expert_id: int) -> ExpertWeights:
        """Return the CPU BF16 gate-up and down-projection tensors for an expert."""
        self._validate_indices(layer_id, expert_id)
        return self._experts[(layer_id, expert_id)]

    def num_experts(self) -> int:
        """Return the total number of stored experts."""
        return len(self._experts)

    def expert_size_bytes(self) -> int:
        """Return the size of one expert's two weight tensors in bytes."""
        return self._expert_size_bytes

    def total_size_bytes(self) -> int:
        """Return the total size of all stored expert weights in bytes."""
        return self.num_experts() * self.expert_size_bytes()

    @classmethod
    def _validate_indices(cls, layer_id: int, expert_id: int) -> None:
        if isinstance(layer_id, bool) or not isinstance(layer_id, int):
            raise TypeError("layer_id must be an integer")
        if isinstance(expert_id, bool) or not isinstance(expert_id, int):
            raise TypeError("expert_id must be an integer")
        if not 0 <= layer_id < cls.NUM_LAYERS:
            raise IndexError(f"layer_id must be in [0, {cls.NUM_LAYERS - 1}], got {layer_id}")
        if not 0 <= expert_id < cls.NUM_EXPERTS_PER_LAYER:
            raise IndexError(
                f"expert_id must be in [0, {cls.NUM_EXPERTS_PER_LAYER - 1}], got {expert_id}"
            )

    @classmethod
    def _validate_model(cls, model: torch.nn.Module) -> None:
        try:
            layers = model.model.layers
        except AttributeError as error:
            raise TypeError("model must be a Hugging Face Mixtral model") from error

        if len(layers) != cls.NUM_LAYERS:
            raise ValueError(f"expected {cls.NUM_LAYERS} Mixtral layers, found {len(layers)}")

        for layer_id, layer in enumerate(layers):
            try:
                experts = layer.mlp.experts
                gate_up_proj = experts.gate_up_proj
                down_proj = experts.down_proj
            except AttributeError as error:
                raise TypeError(f"layer {layer_id} does not expose Mixtral expert weights") from error

            expected_gate_up_shape = (cls.NUM_EXPERTS_PER_LAYER, *cls.GATE_UP_SHAPE)
            expected_down_shape = (cls.NUM_EXPERTS_PER_LAYER, *cls.DOWN_SHAPE)
            if tuple(gate_up_proj.shape) != expected_gate_up_shape:
                raise ValueError(
                    f"layer {layer_id} gate_up_proj has shape {tuple(gate_up_proj.shape)}, "
                    f"expected {expected_gate_up_shape}"
                )
            if tuple(down_proj.shape) != expected_down_shape:
                raise ValueError(
                    f"layer {layer_id} down_proj has shape {tuple(down_proj.shape)}, "
                    f"expected {expected_down_shape}"
                )
            if gate_up_proj.dtype != cls.DTYPE or down_proj.dtype != cls.DTYPE:
                raise TypeError(f"layer {layer_id} expert weights must be torch.bfloat16")
            if gate_up_proj.device.type != "cpu" or down_proj.device.type != "cpu":
                raise ValueError(f"layer {layer_id} expert weights must reside on CPU")

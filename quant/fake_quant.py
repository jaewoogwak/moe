"""Chunked group-wise symmetric RTN fake quantization for BF16 expert weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from offload.host_expert_store import HostExpertStore


@dataclass
class QuantizationStats:
    """Sufficient statistics over fake-quantized BF16 weights."""

    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    squared_weight_sum: float = 0.0
    max_abs_error: float = 0.0
    parameter_count: int = 0

    def update(self, original: torch.Tensor, dequantized: torch.Tensor) -> None:
        error = (dequantized - original).abs()
        self.abs_error_sum += error.sum().item()
        self.squared_error_sum += error.square().sum().item()
        self.squared_weight_sum += original.square().sum().item()
        self.max_abs_error = max(self.max_abs_error, error.max().item())
        self.parameter_count += original.numel()

    def merge(self, other: "QuantizationStats") -> None:
        self.abs_error_sum += other.abs_error_sum
        self.squared_error_sum += other.squared_error_sum
        self.squared_weight_sum += other.squared_weight_sum
        self.max_abs_error = max(self.max_abs_error, other.max_abs_error)
        self.parameter_count += other.parameter_count

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_abs_weight_error": self.abs_error_sum / max(1, self.parameter_count),
            "relative_l2_weight_error": (self.squared_error_sum / max(self.squared_weight_sum, 1e-30)) ** 0.5,
            "max_abs_weight_error": self.max_abs_error,
            "quantized_parameter_count": self.parameter_count,
        }


def _validate(bits: int, group_size: int, tensor: torch.Tensor) -> None:
    if bits not in (3, 4, 8):
        raise ValueError("bits must be one of 3, 4, or 8")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if tensor.ndim != 2:
        raise ValueError("fake quantization expects a 2D linear weight tensor")
    if tensor.shape[-1] % group_size:
        raise ValueError(
            f"input dimension {tensor.shape[-1]} must be divisible by group_size {group_size}"
        )
    if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
        raise TypeError("fake quantization expects CPU BF16 expert tensors")


@torch.inference_mode()
def fake_quantize_tensor_(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    row_chunk_size: int = 128,
) -> QuantizationStats:
    """Replace ``tensor`` with its group-wise symmetric RTN BF16 reconstruction.

    Groups are contiguous slices along the linear input dimension (the last
    dimension). Only ``row_chunk_size × input_dim`` values are promoted to FP32
    at once, avoiding a multi-GiB temporary allocation for Mixtral experts.
    """
    _validate(bits, group_size, tensor)
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")

    qmax = (1 << (bits - 1)) - 1
    stats = QuantizationStats()
    input_dim = tensor.shape[-1]
    groups_per_row = input_dim // group_size

    for row_start in range(0, tensor.shape[0], row_chunk_size):
        chunk = tensor[row_start : row_start + row_chunk_size]
        original = chunk.float()
        grouped = original.reshape(-1, groups_per_row, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).div(qmax)
        # All-zero groups are represented exactly without division by zero.
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        quantized = torch.round(grouped / scale).clamp_(-qmax, qmax)
        dequantized = (quantized * scale).reshape_as(original)
        stats.update(original, dequantized)
        chunk.copy_(dequantized.to(dtype=torch.bfloat16))
    return stats


@torch.inference_mode()
def fake_quantize_expert_store_(
    store: HostExpertStore,
    *,
    bits: int,
    group_size: int,
    row_chunk_size: int = 128,
) -> QuantizationStats:
    """Fake-quantize every pinned Mixtral expert in place."""
    if not store.all_experts_pinned():
        raise AssertionError("expert store must contain pinned CPU tensors before quantization")
    total = QuantizationStats()
    for layer_id in range(store.NUM_LAYERS):
        for expert_id in range(store.NUM_EXPERTS_PER_LAYER):
            gate_up, down = store.get(layer_id, expert_id)
            total.merge(
                fake_quantize_tensor_(
                    gate_up,
                    bits=bits,
                    group_size=group_size,
                    row_chunk_size=row_chunk_size,
                )
            )
            total.merge(
                fake_quantize_tensor_(
                    down,
                    bits=bits,
                    group_size=group_size,
                    row_chunk_size=row_chunk_size,
                )
            )
    return total

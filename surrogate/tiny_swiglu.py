"""Tiny SwiGLU surrogate architecture and deployment-size helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_SIZE = 4096
EXACT_INTERMEDIATE_SIZE = 14336


class TinySwiGLUSurrogate(nn.Module):
    """A small bias-free SwiGLU MLP with Mixtral's expert functional form."""

    def __init__(self, intermediate_size: int) -> None:
        super().__init__()
        if intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(HIDDEN_SIZE, intermediate_size, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, HIDDEN_SIZE, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply ``down(SiLU(gate(inputs)) * up(inputs))``."""
        gate = self.gate_proj(inputs)
        up = self.up_proj(inputs)
        return self.down_proj(F.silu(gate) * up)


def tiny_swiglu_parameter_count(intermediate_size: int) -> int:
    """Return parameter count for the bias-free Tiny SwiGLU architecture."""
    if intermediate_size < 1:
        raise ValueError("intermediate_size must be positive")
    return 3 * HIDDEN_SIZE * intermediate_size


def exact_expert_parameter_count() -> int:
    """Return parameter count for a bias-free Mixtral expert."""
    return 3 * HIDDEN_SIZE * EXACT_INTERMEDIATE_SIZE


def bf16_size_bytes(parameter_count: int) -> int:
    """Return deployment storage bytes for BF16 parameters."""
    if parameter_count < 0:
        raise ValueError("parameter_count must be non-negative")
    return parameter_count * torch.tensor([], dtype=torch.bfloat16).element_size()


def tiny_swiglu_size_report(intermediate_size: int) -> dict[str, float | int]:
    """Describe Tiny SwiGLU BF16 storage relative to an exact Mixtral expert."""
    parameter_count = tiny_swiglu_parameter_count(intermediate_size)
    exact_parameters = exact_expert_parameter_count()
    size_bytes = bf16_size_bytes(parameter_count)
    exact_size_bytes = bf16_size_bytes(exact_parameters)
    return {
        "parameter_count": parameter_count,
        "bf16_size_bytes": size_bytes,
        "compression_ratio": exact_size_bytes / size_bytes,
    }

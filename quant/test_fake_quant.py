"""Small deterministic unit test for chunked fake quantization."""

from __future__ import annotations

import torch

from quant.fake_quant import fake_quantize_tensor_


def quantization_error(bits: int) -> float:
    torch.manual_seed(0)
    original = torch.randn((64, 256), dtype=torch.float32).to(torch.bfloat16)
    tensor = original.clone()
    fake_quantize_tensor_(tensor, bits=bits, group_size=128, row_chunk_size=11)
    assert tensor.shape == original.shape
    assert tensor.dtype == torch.bfloat16
    return (tensor.float() - original.float()).abs().mean().item()


def main() -> None:
    errors = {bits: quantization_error(bits) for bits in (8, 4, 3)}
    print("mean absolute BF16 reconstruction error:", errors)
    assert errors[8] < errors[4] < errors[3]
    print("fake quantization unit test: PASS")


if __name__ == "__main__":
    main()

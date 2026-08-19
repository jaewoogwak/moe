"""Measure CPU staging and H2D bandwidth for one real Mixtral expert."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offload.host_expert_store import HostExpertStore


DEFAULT_MODEL = "/workspace/models/Mixtral-8x7B-Instruct-v0.1"


@dataclass(frozen=True)
class CopyResult:
    name: str
    times_ms: list[float]
    total_bytes: int

    @property
    def mean_ms(self) -> float:
        return sum(self.times_ms) / len(self.times_ms)

    @property
    def bandwidth_gbps(self) -> float:
        return self.total_bytes / (self.mean_ms / 1_000.0) / 1e9


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def empty_like_cpu(tensor: torch.Tensor, *, pinned: bool) -> torch.Tensor:
    return torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device="cpu",
        pin_memory=pinned,
    )


def print_tensor_info(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape: {tuple(tensor.shape)}")
    print(f"  stride: {tensor.stride()}")
    print(f"  contiguous: {tensor.is_contiguous()}")
    print(f"  pinned: {tensor.is_pinned()}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")


def benchmark_cpu_copy(
    name: str,
    source: tuple[torch.Tensor, torch.Tensor],
    destination: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    repeat: int,
) -> CopyResult:
    for _ in range(warmup):
        destination[0].copy_(source[0])
        destination[1].copy_(source[1])

    times_ms: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        destination[0].copy_(source[0])
        destination[1].copy_(source[1])
        times_ms.append((time.perf_counter() - start) * 1_000.0)
    return CopyResult(name, times_ms, tensor_bytes(*source))


def benchmark_pinned_h2d(
    source: tuple[torch.Tensor, torch.Tensor],
    destination: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    repeat: int,
) -> CopyResult:
    for _ in range(warmup):
        destination[0].copy_(source[0], non_blocking=True)
        destination[1].copy_(source[1], non_blocking=True)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        destination[0].copy_(source[0], non_blocking=True)
        destination[1].copy_(source[1], non_blocking=True)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return CopyResult("pinned -> GPU", times_ms, tensor_bytes(*source))


def command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return "not available"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output or f"command exited with {completed.returncode}"


def print_system_diagnostics() -> None:
    print("\nSystem/process diagnostics")
    print("  sched_getaffinity(0):", sorted(os.sched_getaffinity(0)))
    print("  torch.get_num_threads():", torch.get_num_threads())
    print("  torch.get_num_interop_threads():", torch.get_num_interop_threads())
    status = Path("/proc/self/status")
    if status.exists():
        numa_lines = [
            line.strip()
            for line in status.read_text().splitlines()
            if line.startswith(("Cpus_allowed_list:", "Mems_allowed_list:"))
        ]
        print("  /proc/self/status:", "; ".join(numa_lines) or "no CPU/NUMA affinity fields")
    print("  numactl --show:")
    print("    " + command_output(["numactl", "--show"]).replace("\n", "\n    "))
    print("  GPU PCIe:")
    print(
        "    "
        + command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id",
                "--format=csv,noheader",
            ]
        ).replace("\n", "\n    ")
    )


def print_result(result: CopyResult) -> None:
    print(
        f"{result.name:22s}: {result.mean_ms:8.2f} ms, {result.bandwidth_gbps:6.2f} GB/s "
        f"(min {min(result.times_ms):.2f}, max {max(result.times_ms):.2f} ms)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned -> GPU diagnostic")
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("--warmup must be non-negative and --repeat must be positive")

    print(f"Loading Mixtral on CPU: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    store = HostExpertStore(model)
    source = store.get(args.layer, args.expert)
    gate_up, down = source

    print(f"\nReal expert: layer={args.layer}, expert={args.expert}")
    print_tensor_info("gate_up", gate_up)
    print_tensor_info("down", down)
    expert_bytes = tensor_bytes(*source)
    print(f"total expert size: {expert_bytes:,} bytes ({expert_bytes / 1024**2:.2f} MiB)")
    print_system_diagnostics()

    # All buffers are allocated once. Their allocation and initial population
    # are deliberately outside the timed loops.
    pageable_destination = (empty_like_cpu(gate_up, pinned=False), empty_like_cpu(down, pinned=False))
    pinned_destination = (empty_like_cpu(gate_up, pinned=True), empty_like_cpu(down, pinned=True))
    pinned_source = (empty_like_cpu(gate_up, pinned=True), empty_like_cpu(down, pinned=True))
    pinned_source[0].copy_(gate_up)
    pinned_source[1].copy_(down)
    gpu_destination = (
        torch.empty_like(gate_up, device="cuda"),
        torch.empty_like(down, device="cuda"),
    )

    with torch.inference_mode():
        results = [
            benchmark_cpu_copy(
                "pageable -> pinned", source, pinned_destination, args.warmup, args.repeat
            ),
            benchmark_cpu_copy(
                "pageable -> pageable", source, pageable_destination, args.warmup, args.repeat
            ),
            benchmark_cpu_copy(
                "pinned -> pinned", pinned_source, pinned_destination, args.warmup, args.repeat
            ),
            benchmark_pinned_h2d(pinned_source, gpu_destination, args.warmup, args.repeat),
        ]

    print("\nDetailed results")
    for result in results:
        print_result(result)
    print("\nConcise comparison")
    for result in results:
        print(f"{result.name:22s}: {result.mean_ms:.2f} ms, {result.bandwidth_gbps:.2f} GB/s")


if __name__ == "__main__":
    main()

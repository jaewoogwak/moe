import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


MODEL = "/workspace/models/Mixtral-8x7B-v0.1"

LAYER_ID = 0
EXPERT_ID = 0

WARMUP = 5
REPEAT = 20


# ============================================================
# Utility
# ============================================================

def tensor_size_bytes(x):
    return x.numel() * x.element_size()


def bytes_to_mib(x):
    return x / (1024 ** 2)


def benchmark_h2d(
    gate_up_cpu,
    down_cpu,
    gate_up_gpu,
    down_gpu,
    warmup=5,
    repeat=20,
):
    """
    Measure CPU pinned memory -> GPU H2D latency.

    Important:
      - default CUDA stream
      - no separate copy stream
      - no overlap/prefetch
    """
    
    print("gate_up_cpu.device:  ", gate_up_cpu.device)
    print("down_cpu.device:  ", down_cpu.device)
    
    print("gate_up_gpu.device:  ", gate_up_gpu.device)
    print("down_gpu.device:  ", down_gpu.device)

    # Warmup
    for _ in range(warmup):
        gate_up_gpu.copy_(gate_up_cpu, non_blocking=True)
        down_gpu.copy_(down_cpu, non_blocking=True)

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(repeat):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        gate_up_gpu.copy_(
            gate_up_cpu,
            non_blocking=True,
        )

        down_gpu.copy_(
            down_cpu,
            non_blocking=True,
        )

        end.record()

        # Wait once after both copies
        end.synchronize()

        times_ms.append(start.elapsed_time(end))

    return times_ms


def benchmark_expert_compute(
    hidden_states,
    gate_up_gpu,
    down_gpu,
    warmup=5,
    repeat=20,
):
    """
    Measure one Mixtral expert SwiGLU computation:

        gate, up = Linear(x, gate_up)
        h = SiLU(gate) * up
        y = Linear(h, down)
    """

    # Warmup
    for _ in range(warmup):
        gate, up = F.linear(
            hidden_states,
            gate_up_gpu,
        ).chunk(2, dim=-1)

        intermediate = F.silu(gate) * up

        output = F.linear(
            intermediate,
            down_gpu,
        )

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(repeat):

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        gate, up = F.linear(
            hidden_states,
            gate_up_gpu,
        ).chunk(2, dim=-1)

        intermediate = F.silu(gate) * up

        output = F.linear(
            intermediate,
            down_gpu,
        )

        end.record()

        end.synchronize()

        times_ms.append(start.elapsed_time(end))

    return times_ms


# ============================================================
# 1. Load Mixtral on CPU
# ============================================================

print("=" * 70)
print("Loading Mixtral on CPU")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

model.eval()

print("Loaded.")


# ============================================================
# 2. Select one real expert from checkpoint
# ============================================================

experts = model.model.layers[LAYER_ID].mlp.experts

print()
print("=" * 70)
print(f"Layer {LAYER_ID}, Expert {EXPERT_ID}")
print("=" * 70)

print("Expert module:", type(experts))


# Actual checkpoint tensors
gate_up_src = experts.gate_up_proj[EXPERT_ID]
down_src = experts.down_proj[EXPERT_ID]

print()
print("gate_up_proj:")
print("  shape :", tuple(gate_up_src.shape))
print("  dtype :", gate_up_src.dtype)
print("  device:", gate_up_src.device)

print()
print("down_proj:")
print("  shape :", tuple(down_src.shape))
print("  dtype :", down_src.dtype)
print("  device:", down_src.device)


# ============================================================
# 3. Calculate expert size
# ============================================================

gate_up_bytes = tensor_size_bytes(gate_up_src)
down_bytes = tensor_size_bytes(down_src)

expert_bytes = gate_up_bytes + down_bytes

print()
print("=" * 70)
print("Expert Size")
print("=" * 70)

print(
    f"gate_up_proj : "
    f"{bytes_to_mib(gate_up_bytes):.2f} MiB"
)

print(
    f"down_proj    : "
    f"{bytes_to_mib(down_bytes):.2f} MiB"
)

print(
    f"Total expert : "
    f"{bytes_to_mib(expert_bytes):.2f} MiB"
)

print(
    f"Total expert : "
    f"{expert_bytes / 1e6:.2f} MB"
)


# ============================================================
# 4. Create pinned CPU expert storage
# ============================================================

print()
print("=" * 70)
print("Creating pinned CPU expert storage")
print("=" * 70)

gate_up_cpu = torch.empty(
    gate_up_src.shape,
    dtype=gate_up_src.dtype,
    device="cpu",
    pin_memory=True,
)

down_cpu = torch.empty(
    down_src.shape,
    dtype=down_src.dtype,
    device="cpu",
    pin_memory=True,
)

gate_up_cpu.copy_(gate_up_src)
down_cpu.copy_(down_src)

print("gate_up pinned:", gate_up_cpu.is_pinned())
print("down pinned   :", down_cpu.is_pinned())


# ============================================================
# 5. Create reusable GPU buffers
# ============================================================

print()
print("=" * 70)
print("Allocating GPU expert buffers")
print("=" * 70)

gate_up_gpu = torch.empty(
    gate_up_cpu.shape,
    dtype=gate_up_cpu.dtype,
    device="cuda",
)

down_gpu = torch.empty(
    down_cpu.shape,
    dtype=down_cpu.dtype,
    device="cuda",
)

print(
    "GPU memory allocated:",
    f"{torch.cuda.memory_allocated() / 1024**3:.3f} GiB",
)


# ============================================================
# 6. Benchmark H2D
# ============================================================

print()
print("=" * 70)
print("Benchmark: Expert H2D")
print("=" * 70)

h2d_times = benchmark_h2d(
    gate_up_cpu,
    down_cpu,
    gate_up_gpu,
    down_gpu,
    warmup=WARMUP,
    repeat=REPEAT,
)

mean_h2d_ms = sum(h2d_times) / len(h2d_times)

min_h2d_ms = min(h2d_times)
max_h2d_ms = max(h2d_times)

# GB/s using decimal GB, matching typical PCIe bandwidth reporting
expert_gb = expert_bytes / 1e9

mean_bandwidth = expert_gb / (mean_h2d_ms / 1000.0)


print(f"Mean H2D latency : {mean_h2d_ms:.3f} ms")
print(f"Min H2D latency  : {min_h2d_ms:.3f} ms")
print(f"Max H2D latency  : {max_h2d_ms:.3f} ms")

print(
    f"Effective H2D BW : "
    f"{mean_bandwidth:.2f} GB/s"
)


# ============================================================
# 7. Verify copied values
# ============================================================

print()
print("=" * 70)
print("Verify H2D copy")
print("=" * 70)

gate_up_match = torch.equal(
    gate_up_gpu.cpu(),
    gate_up_cpu,
)

down_match = torch.equal(
    down_gpu.cpu(),
    down_cpu,
)

print("gate_up bit-exact:", gate_up_match)
print("down bit-exact   :", down_match)


# ============================================================
# 8. Benchmark expert computation
# ============================================================

print()
print("=" * 70)
print("Benchmark: Expert Compute")
print("=" * 70)

hidden_size = gate_up_gpu.shape[1]

# One decoding token, batch size = 1
hidden_states = torch.randn(
    1,
    hidden_size,
    dtype=torch.bfloat16,
    device="cuda",
)

compute_times = benchmark_expert_compute(
    hidden_states,
    gate_up_gpu,
    down_gpu,
    warmup=WARMUP,
    repeat=REPEAT,
)

mean_compute_ms = sum(compute_times) / len(compute_times)

print(
    f"Mean expert compute : "
    f"{mean_compute_ms:.3f} ms"
)

print(
    f"Min expert compute  : "
    f"{min(compute_times):.3f} ms"
)

print(
    f"Max expert compute  : "
    f"{max(compute_times):.3f} ms"
)


# ============================================================
# 9. Summary
# ============================================================

print()
print("=" * 70)
print("Summary")
print("=" * 70)

print(
    f"Expert size         : "
    f"{bytes_to_mib(expert_bytes):.2f} MiB"
)

print(
    f"H2D latency         : "
    f"{mean_h2d_ms:.3f} ms"
)

print(
    f"H2D bandwidth       : "
    f"{mean_bandwidth:.2f} GB/s"
)

print(
    f"Expert compute      : "
    f"{mean_compute_ms:.3f} ms"
)

print(
    f"H2D / Compute ratio : "
    f"{mean_h2d_ms / mean_compute_ms:.2f}x"
)
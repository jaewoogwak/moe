#!/usr/bin/env python3
"""Evaluate Tiny SwiGLU capacities alongside available full-rank linear baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.tiny_swiglu import HIDDEN_SIZE, TinySwiGLUSurrogate, bf16_size_bytes, tiny_swiglu_size_report
from surrogate.train_per_expert_linear import NUM_EXPERTS, load_trace, trace_tensors


EPSILON = 1e-12
MetricSummary = dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/surrogate_per_expert/layer_16"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--linear-checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 4096])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(NUM_EXPERTS)))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate evaluation arguments."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tiny SwiGLU evaluation")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if min(args.hidden_sizes) < 1 or len(set(args.hidden_sizes)) != len(args.hidden_sizes):
        raise ValueError("--hidden-sizes must contain unique positive values")
    if sorted(set(args.experts)) != sorted(args.experts) or any(expert not in range(NUM_EXPERTS) for expert in args.experts):
        raise ValueError("--experts must be unique IDs in [0, 7]")


def summarize(values: list[torch.Tensor]) -> dict[str, float]:
    """Calculate mean and median over per-sample scalar values."""
    all_values = torch.cat(values).float()
    return {"mean": float(all_values.mean().item()), "median": float(all_values.median().item())}


@torch.no_grad()
def evaluate_predictor(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    input_dtype: torch.dtype,
) -> MetricSummary:
    """Evaluate one predictor with sample-wise MSE, relative L2, and cosine metrics."""
    mse_values: list[torch.Tensor] = []
    relative_l2_values: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    for start_index in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start_index : start_index + batch_size].to("cuda", dtype=input_dtype)
        batch_targets = targets[start_index : start_index + batch_size].to("cuda", dtype=torch.float32)
        predictions = predictor(batch_inputs).float()
        difference = predictions - batch_targets
        mse_values.append(difference.square().mean(dim=1).cpu())
        relative_l2_values.append(
            (torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(batch_targets, dim=1).clamp_min(EPSILON)).cpu()
        )
        cosine_values.append(F.cosine_similarity(predictions, batch_targets, dim=1, eps=EPSILON).cpu())
    return {
        "mse": summarize(mse_values),
        "relative_l2": summarize(relative_l2_values),
        "cosine_similarity": summarize(cosine_values),
    }


def load_tiny_swiglu(checkpoint_path: Path) -> TinySwiGLUSurrogate:
    """Load a validation-best Tiny SwiGLU checkpoint as a FP32 CUDA module."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("surrogate_type") != "tiny_swiglu" or checkpoint.get("hidden_size") != HIDDEN_SIZE:
        raise ValueError(f"incompatible Tiny SwiGLU checkpoint: {checkpoint_path}")
    intermediate_size = checkpoint.get("intermediate_size")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(intermediate_size, int) or not isinstance(state_dict, dict):
        raise TypeError(f"Tiny SwiGLU checkpoint lacks architecture metadata: {checkpoint_path}")
    model = TinySwiGLUSurrogate(intermediate_size).to("cuda", dtype=torch.float32)
    model.load_state_dict(state_dict)
    return model.eval()


def make_bf16_model(fp32_model: nn.Module) -> nn.Module:
    """Create a BF16 deployment copy from one FP32 checkpoint model."""
    if isinstance(fp32_model, TinySwiGLUSurrogate):
        bf16_model: nn.Module = TinySwiGLUSurrogate(fp32_model.intermediate_size)
    elif isinstance(fp32_model, nn.Linear):
        bf16_model = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True)
    else:
        raise TypeError(f"unsupported deployment model type: {type(fp32_model).__name__}")
    bf16_model = bf16_model.to("cuda", dtype=torch.bfloat16)
    bf16_model.load_state_dict({name: value.detach().to(dtype=torch.bfloat16) for name, value in fp32_model.state_dict().items()})
    if any(parameter.dtype != torch.bfloat16 for parameter in bf16_model.parameters()):
        raise AssertionError("BF16 deployment model parameters are not torch.bfloat16")
    return bf16_model.eval()


def load_full_rank_linear(checkpoint_path: Path) -> nn.Linear:
    """Load an existing full-rank linear baseline checkpoint when available."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("hidden_size") != HIDDEN_SIZE:
        raise ValueError(f"incompatible full-rank linear checkpoint: {checkpoint_path}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError(f"full-rank checkpoint lacks state_dict: {checkpoint_path}")
    model = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True, device="cuda", dtype=torch.float32)
    model.load_state_dict(state_dict)
    return model.eval()


def metric_records(
    expert_id: int,
    surrogate_type: str,
    intermediate_size: int | None,
    parameter_count: int,
    bf16_size_bytes: int,
    compression_ratio: float | None,
    precision: str,
    metrics: MetricSummary,
) -> list[dict[str, object]]:
    """Create rows with the capacity metadata required by the result schema."""
    return [
        {
            "expert": expert_id,
            "surrogate_type": surrogate_type,
            "intermediate_size": intermediate_size,
            "parameter_count": parameter_count,
            "bf16_size_bytes": bf16_size_bytes,
            "compression_ratio": compression_ratio,
            "precision": precision,
            "metric": metric_name,
            **values,
        }
        for metric_name, values in metrics.items()
    ]


def full_linear_size_report() -> dict[str, float | int]:
    """Calculate the full-rank linear baseline's BF16 deployment storage."""
    parameter_count = HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
    exact_parameters = 3 * HIDDEN_SIZE * 14336
    size_bytes = bf16_size_bytes(parameter_count)
    return {
        "parameter_count": parameter_count,
        "bf16_size_bytes": size_bytes,
        "compression_ratio": bf16_size_bytes(exact_parameters) / size_bytes,
    }


def print_capacity_table(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate the BF16 deployment quality of each architecture across experts."""
    grouped: dict[tuple[str, int | None, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["metric"] in {"relative_l2", "cosine_similarity"}:
            grouped[(str(record["surrogate_type"]), record["intermediate_size"], str(record["precision"]))].append(record)

    rows: list[dict[str, object]] = []
    for (surrogate_type, intermediate_size, precision), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or -1, item[0][2])):
        rel_l2 = [float(value["mean"]) for value in values if value["metric"] == "relative_l2"]
        cosine = [float(value["mean"]) for value in values if value["metric"] == "cosine_similarity"]
        if not rel_l2 or not cosine:
            continue
        label = surrogate_type if intermediate_size is None else f"{surrogate_type}-{intermediate_size}"
        rows.append(
            {
                "architecture": label,
                "surrogate_type": surrogate_type,
                "intermediate_size": intermediate_size,
                "precision": precision,
                "experts": len(rel_l2),
                "relative_l2_mean": sum(rel_l2) / len(rel_l2),
                "cosine_mean": sum(cosine) / len(cosine),
                "bf16_mib": float(values[0]["bf16_size_bytes"]) / 1024**2,
                "compression_ratio": values[0]["compression_ratio"],
            }
        )

    deployment_rows = [
        row
        for row in rows
        if row["precision"] == "bf16" or row["surrogate_type"] == "mean_output"
    ]
    print("Architecture | RelL2 ↓ | Cosine ↑ | BF16 MiB | Compression")
    for row in deployment_rows:
        compression = row["compression_ratio"]
        compression_text = "n/a" if compression is None else f"{float(compression):.2f}x"
        print(
            f"{row['architecture']:<22} | {row['relative_l2_mean']:.6f} | {row['cosine_mean']:.6f} | "
            f"{row['bf16_mib']:.2f} | {compression_text}"
        )
    return rows


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_dir = args.checkpoint_dir or args.trace_dir / "checkpoints_tiny_swiglu"
    linear_checkpoint_dir = args.linear_checkpoint_dir or args.trace_dir / "checkpoints"
    output_dir = args.output_dir or args.trace_dir / "evaluation_tiny_swiglu"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    linear_size = full_linear_size_report()

    for expert_id in args.experts:
        train_trace = load_trace(args.trace_dir / "train" / f"expert_{expert_id}.pt")
        test_trace = load_trace(args.trace_dir / "test" / f"expert_{expert_id}.pt")
        _, train_targets = trace_tensors(train_trace)
        test_inputs, test_targets = trace_tensors(test_trace)
        mean_output = train_targets.float().mean(dim=0, keepdim=True).to("cuda")
        zero_metrics = evaluate_predictor(test_inputs, test_targets, args.batch_size, lambda batch: torch.zeros_like(batch), torch.float32)
        mean_metrics = evaluate_predictor(
            test_inputs,
            test_targets,
            args.batch_size,
            lambda batch: mean_output.expand(batch.shape[0], -1),
            torch.float32,
        )
        records.extend(metric_records(expert_id, "zero", None, 0, 0, None, "fp32", zero_metrics))
        records.extend(metric_records(expert_id, "mean_output", None, 0, 0, None, "fp32", mean_metrics))

        linear_checkpoint = linear_checkpoint_dir / f"expert_{expert_id}.pt"
        if linear_checkpoint.is_file():
            linear_model = load_full_rank_linear(linear_checkpoint)
            linear_fp32 = evaluate_predictor(test_inputs, test_targets, args.batch_size, linear_model, torch.float32)
            linear_bf16_model = make_bf16_model(linear_model)
            linear_bf16 = evaluate_predictor(test_inputs, test_targets, args.batch_size, linear_bf16_model, torch.bfloat16)
            records.extend(metric_records(expert_id, "full_rank_linear", None, precision="fp32", metrics=linear_fp32, **linear_size))
            records.extend(metric_records(expert_id, "full_rank_linear", None, precision="bf16", metrics=linear_bf16, **linear_size))
            del linear_model, linear_bf16_model
        else:
            print(f"Full-rank linear checkpoint missing for E{expert_id}; skipping linear comparison")

        for intermediate_size in args.hidden_sizes:
            checkpoint = checkpoint_dir / f"hidden_{intermediate_size}" / f"expert_{expert_id}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing Tiny SwiGLU checkpoint: {checkpoint}")
            fp32_model = load_tiny_swiglu(checkpoint)
            fp32_metrics = evaluate_predictor(test_inputs, test_targets, args.batch_size, fp32_model, torch.float32)
            bf16_model = make_bf16_model(fp32_model)
            bf16_metrics = evaluate_predictor(test_inputs, test_targets, args.batch_size, bf16_model, torch.bfloat16)
            size = tiny_swiglu_size_report(intermediate_size)
            records.extend(metric_records(expert_id, "tiny_swiglu", intermediate_size, precision="fp32", metrics=fp32_metrics, **size))
            records.extend(metric_records(expert_id, "tiny_swiglu", intermediate_size, precision="bf16", metrics=bf16_metrics, **size))
            del fp32_model, bf16_model
        del mean_output

    fields = (
        "expert", "surrogate_type", "intermediate_size", "parameter_count", "bf16_size_bytes",
        "compression_ratio", "precision", "metric", "mean", "median",
    )
    with (output_dir / "evaluation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    capacity_rows = print_capacity_table(records)
    capacity_fields = (
        "architecture", "surrogate_type", "intermediate_size", "precision", "experts",
        "relative_l2_mean", "cosine_mean", "bf16_mib", "compression_ratio",
    )
    with (output_dir / "capacity_vs_quality.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=capacity_fields)
        writer.writeheader()
        writer.writerows(capacity_rows)
    report: dict[str, Any] = {
        "trace_dir": str(args.trace_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "linear_checkpoint_dir": str(linear_checkpoint_dir),
        "metrics": records,
        "capacity_vs_quality": capacity_rows,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved Tiny SwiGLU evaluation to: {output_dir}")


if __name__ == "__main__":
    main()

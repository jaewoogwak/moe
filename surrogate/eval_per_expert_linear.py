#!/usr/bin/env python3
"""Evaluate zero, mean-output, FP32, and BF16-cast linear expert surrogates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


HIDDEN_SIZE = 4096
NUM_EXPERTS = 8
EPSILON = 1e-12
TraceData = dict[str, object]
MetricSummary = dict[str, dict[str, float]]


def parse_args() -> argparse.Namespace:
    """Parse evaluation configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/surrogate_per_expert/layer_16"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(NUM_EXPERTS)))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate evaluation arguments."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if sorted(set(args.experts)) != sorted(args.experts):
        raise ValueError("--experts must not contain duplicates")
    if any(expert_id < 0 or expert_id >= NUM_EXPERTS for expert_id in args.experts):
        raise ValueError("--experts must contain IDs in [0, 7]")


def load_trace(path: Path) -> TraceData:
    """Load and validate one non-empty trace file."""
    trace_data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(trace_data, dict):
        raise TypeError(f"trace must be a dictionary: {path}")
    inputs = trace_data.get("x")
    targets = trace_data.get("y")
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(f"trace does not contain tensor x/y: {path}")
    if inputs.shape != targets.shape or inputs.ndim != 2 or inputs.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"trace must contain matching [N, {HIDDEN_SIZE}] tensors: {path}")
    if inputs.shape[0] == 0:
        raise ValueError(f"trace has no samples: {path}")
    return trace_data


def trace_tensors(trace_data: TraceData) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract input and exact-output tensors from a validated trace."""
    inputs = trace_data["x"]
    targets = trace_data["y"]
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise AssertionError("validated trace unexpectedly lacks tensors")
    return inputs, targets


def summarize(values: list[torch.Tensor]) -> dict[str, float]:
    """Calculate mean and median over sample-wise metric values."""
    all_values = torch.cat(values).float()
    return {"mean": float(all_values.mean().item()), "median": float(all_values.median().item())}


@torch.no_grad()
def evaluate_predictor(
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    input_dtype: torch.dtype,
) -> MetricSummary:
    """Compute sample-wise MSE, relative L2, and cosine metrics."""
    mse_values: list[torch.Tensor] = []
    relative_l2_values: list[torch.Tensor] = []
    cosine_values: list[torch.Tensor] = []
    for start_index in range(0, test_inputs.shape[0], batch_size):
        batch_inputs = test_inputs[start_index : start_index + batch_size].to("cuda", dtype=input_dtype)
        batch_targets = test_targets[start_index : start_index + batch_size].to("cuda", dtype=torch.float32)
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


def load_fp32_surrogate(checkpoint_path: Path) -> nn.Linear:
    """Load the validation-best checkpoint into a FP32 linear module."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("hidden_size") != HIDDEN_SIZE:
        raise ValueError(f"incompatible surrogate checkpoint: {checkpoint_path}")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise TypeError(f"checkpoint lacks state_dict: {checkpoint_path}")
    model = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True, device="cuda", dtype=torch.float32)
    model.load_state_dict(state_dict)
    return model.eval()


def make_bf16_surrogate(fp32_model: nn.Linear) -> nn.Linear:
    """Cast one FP32 checkpoint to a BF16 deployment surrogate."""
    bf16_model = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        bf16_model.weight.copy_(fp32_model.weight.to(dtype=torch.bfloat16))
        bf16_model.bias.copy_(fp32_model.bias.to(dtype=torch.bfloat16))
    if any(parameter.dtype != torch.bfloat16 for parameter in bf16_model.parameters()):
        raise AssertionError("BF16 deployment surrogate parameters are not torch.bfloat16")
    return bf16_model.eval()


def metric_records(expert_id: int, baseline: str, metrics: MetricSummary) -> list[dict[str, object]]:
    """Convert one baseline metric dictionary into CSV-compatible records."""
    return [
        {"expert": expert_id, "baseline": baseline, "metric": metric_name, **values}
        for metric_name, values in metrics.items()
    ]


def deployment_degradation(fp32_metrics: MetricSummary, bf16_metrics: MetricSummary) -> dict[str, float]:
    """Calculate mean-metric change introduced by a BF16 deployment cast."""
    return {
        "bf16_minus_fp32_rel_l2": bf16_metrics["relative_l2"]["mean"] - fp32_metrics["relative_l2"]["mean"],
        "bf16_minus_fp32_cosine": bf16_metrics["cosine_similarity"]["mean"] - fp32_metrics["cosine_similarity"]["mean"],
    }


def size_report() -> dict[str, float | int]:
    """Calculate BF16 deployment sizes, excluding FP32 optimization checkpoints."""
    exact_expert_parameters = 28672 * 4096 + 4096 * 14336
    linear_surrogate_parameters = HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    exact_expert_bf16_bytes = exact_expert_parameters * element_size
    linear_surrogate_bf16_bytes = linear_surrogate_parameters * element_size
    return {
        "exact_expert_parameters": exact_expert_parameters,
        "linear_surrogate_parameters": linear_surrogate_parameters,
        "exact_expert_bf16_bytes": exact_expert_bf16_bytes,
        "linear_surrogate_bf16_bytes": linear_surrogate_bf16_bytes,
        "compression_ratio": exact_expert_bf16_bytes / linear_surrogate_bf16_bytes,
    }


def print_summary(rows: list[dict[str, object]], size: dict[str, float | int]) -> None:
    """Print the per-expert FP32/BF16 deployment comparison table."""
    print("Expert | Train N | Test N | Mean RelL2 | FP32 RelL2 | BF16 RelL2 | FP32 Cos | BF16 Cos")
    for row in rows:
        print(
            f"E{row['expert']:<5} | {row['train_n']:>7,} | {row['test_n']:>6,} | "
            f"{row['mean_rel_l2']:.6f} | {row['fp32_rel_l2']:.6f} | {row['bf16_rel_l2']:.6f} | "
            f"{row['fp32_cos']:.6f} | {row['bf16_cos']:.6f}"
        )
    if rows:
        average = {
            field: sum(float(row[field]) for row in rows) / len(rows)
            for field in ("mean_rel_l2", "fp32_rel_l2", "bf16_rel_l2", "fp32_cos", "bf16_cos")
        }
        print(
            f"Mean   |         |        | {average['mean_rel_l2']:.6f} | {average['fp32_rel_l2']:.6f} | "
            f"{average['bf16_rel_l2']:.6f} | {average['fp32_cos']:.6f} | {average['bf16_cos']:.6f}"
        )
    print(
        f"Exact Mixtral expert BF16 size: {size['exact_expert_parameters']:,} parameters, "
        f"{size['exact_expert_bf16_bytes'] / 1024**2:.2f} MiB"
    )
    print(
        f"Full-rank linear surrogate BF16 size: {size['linear_surrogate_parameters']:,} parameters, "
        f"{size['linear_surrogate_bf16_bytes'] / 1024**2:.2f} MiB"
    )
    print(f"Compression ratio: {size['compression_ratio']:.2f}x")


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_dir = args.checkpoint_dir or args.trace_dir / "checkpoints"
    output_dir = args.output_dir or args.trace_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for expert_id in args.experts:
        train_trace = load_trace(args.trace_dir / "train" / f"expert_{expert_id}.pt")
        test_trace = load_trace(args.trace_dir / "test" / f"expert_{expert_id}.pt")
        _, train_targets = trace_tensors(train_trace)
        test_inputs, test_targets = trace_tensors(test_trace)
        mean_output = train_targets.float().mean(dim=0, keepdim=True).to("cuda")
        zero_metrics = evaluate_predictor(
            test_inputs, test_targets, args.batch_size, lambda batch: torch.zeros_like(batch), torch.float32
        )
        mean_metrics = evaluate_predictor(
            test_inputs,
            test_targets,
            args.batch_size,
            lambda batch: mean_output.expand(batch.shape[0], -1),
            torch.float32,
        )
        fp32_model = load_fp32_surrogate(checkpoint_dir / f"expert_{expert_id}.pt")
        fp32_metrics = evaluate_predictor(test_inputs, test_targets, args.batch_size, fp32_model, torch.float32)
        bf16_model = make_bf16_surrogate(fp32_model)
        bf16_metrics = evaluate_predictor(test_inputs, test_targets, args.batch_size, bf16_model, torch.bfloat16)
        degradation = deployment_degradation(fp32_metrics, bf16_metrics)
        del fp32_model, bf16_model, mean_output

        for baseline, metrics in (
            ("zero", zero_metrics),
            ("mean_output", mean_metrics),
            ("learned_linear_fp32", fp32_metrics),
            ("learned_linear_bf16", bf16_metrics),
        ):
            records.extend(metric_records(expert_id, baseline, metrics))
        for metric_name, value in degradation.items():
            records.append(
                {
                    "expert": expert_id,
                    "baseline": "bf16_minus_fp32",
                    "metric": metric_name,
                    "mean": value,
                    "median": value,
                }
            )
        summary_rows.append(
            {
                "expert": expert_id,
                "train_n": int(train_targets.shape[0]),
                "test_n": int(test_targets.shape[0]),
                "mean_rel_l2": mean_metrics["relative_l2"]["mean"],
                "fp32_rel_l2": fp32_metrics["relative_l2"]["mean"],
                "bf16_rel_l2": bf16_metrics["relative_l2"]["mean"],
                "fp32_cos": fp32_metrics["cosine_similarity"]["mean"],
                "bf16_cos": bf16_metrics["cosine_similarity"]["mean"],
                **degradation,
            }
        )

    with (output_dir / "evaluation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("expert", "baseline", "metric", "mean", "median"))
        writer.writeheader()
        writer.writerows(records)
    size = size_report()
    report: dict[str, Any] = {
        "trace_dir": str(args.trace_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "experts": summary_rows,
        "metrics": records,
        "size": size,
    }
    (output_dir / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print_summary(summary_rows, size)
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()

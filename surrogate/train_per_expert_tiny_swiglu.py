#!/usr/bin/env python3
"""Train independent FP32 Tiny SwiGLU surrogates for every expert and capacity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from surrogate.tiny_swiglu import HIDDEN_SIZE, TinySwiGLUSurrogate
from surrogate.train_per_expert_linear import NUM_EXPERTS, load_trace, set_deterministic_seed, trace_tensors


@dataclass
class GPUTrace:
    """One expert's FP32 train and validation trace retained on CUDA."""

    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor


def parse_args() -> argparse.Namespace:
    """Parse Tiny SwiGLU training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/surrogate_per_expert/layer_16"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(NUM_EXPERTS)))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate training arguments before loading traces."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Tiny SwiGLU training")
    if min(args.hidden_sizes) < 1 or len(set(args.hidden_sizes)) != len(args.hidden_sizes):
        raise ValueError("--hidden-sizes must contain unique positive values")
    if args.batch_size < 1 or args.learning_rate <= 0.0 or args.max_epochs < 1:
        raise ValueError("batch size, learning rate, and max epochs must be positive")
    if args.early_stopping_patience < 1 or args.min_delta < 0.0:
        raise ValueError("invalid early stopping configuration")
    if sorted(set(args.experts)) != sorted(args.experts) or any(expert not in range(NUM_EXPERTS) for expert in args.experts):
        raise ValueError("--experts must be unique IDs in [0, 7]")


def checkpoint_path(output_dir: Path, intermediate_size: int, expert_id: int) -> Path:
    """Return the capacity-specific checkpoint path."""
    return output_dir / f"hidden_{intermediate_size}" / f"expert_{expert_id}.pt"


def prepare_gpu_trace(
    train_trace: dict[str, object],
    validation_trace: dict[str, object],
) -> GPUTrace:
    """Upload one expert's BF16 CPU trace once and retain FP32 tensors on CUDA."""
    train_inputs, train_targets = trace_tensors(train_trace)
    validation_inputs, validation_targets = trace_tensors(validation_trace)
    return GPUTrace(
        train_inputs=train_inputs.to("cuda", dtype=torch.float32),
        train_targets=train_targets.to("cuda", dtype=torch.float32),
        validation_inputs=validation_inputs.to("cuda", dtype=torch.float32),
        validation_targets=validation_targets.to("cuda", dtype=torch.float32),
    )


@torch.no_grad()
def compute_validation_mse(model: nn.Module, trace: GPUTrace, batch_size: int) -> float:
    """Calculate validation MSE with one host synchronization after all batches."""
    squared_error_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    for start_index in range(0, trace.validation_inputs.shape[0], batch_size):
        batch_inputs = trace.validation_inputs[start_index : start_index + batch_size]
        batch_targets = trace.validation_targets[start_index : start_index + batch_size]
        squared_error_sum.add_((model(batch_inputs) - batch_targets).square().sum())
    validation_elements = trace.validation_targets.numel()
    return float((squared_error_sum / validation_elements).item())


def save_checkpoint(
    path: Path,
    model: TinySwiGLUSurrogate,
    expert_id: int,
    intermediate_size: int,
    best_epoch: int,
    best_validation_mse: float,
    epochs_run: int,
    train_samples: int,
    validation_samples: int,
) -> None:
    """Save the FP32 checkpoint that achieved the best validation MSE."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "expert_id": expert_id,
            "surrogate_type": "tiny_swiglu",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": intermediate_size,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "best_epoch": best_epoch,
            "best_val_mse": best_validation_mse,
            "epochs_run": epochs_run,
            "train_samples": train_samples,
            "val_samples": validation_samples,
            "optimization_dtype": "fp32",
        },
        path,
    )


def finalize_checkpoint(path: Path, epochs_run: int, early_stopped: bool) -> None:
    """Update final run metadata without replacing the saved validation-best state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["epochs_run"] = epochs_run
    checkpoint["early_stopped"] = early_stopped
    torch.save(checkpoint, path)


def train_one_configuration(
    args: argparse.Namespace,
    gpu_trace: GPUTrace,
    expert_id: int,
    intermediate_size: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Train one expert-capacity pair with validation early stopping."""
    configuration_seed = args.seed + expert_id * 10_000 + intermediate_size
    set_deterministic_seed(configuration_seed)
    model = TinySwiGLUSurrogate(intermediate_size).to("cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    sample_generator = torch.Generator(device="cuda").manual_seed(configuration_seed)
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0
    early_stopped = False
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        sample_order = torch.randperm(
            gpu_trace.train_inputs.shape[0],
            generator=sample_generator,
            device="cuda",
        )
        training_squared_error = torch.zeros((), device="cuda", dtype=torch.float32)
        for start_index in range(0, sample_order.numel(), args.batch_size):
            batch_indices = sample_order[start_index : start_index + args.batch_size]
            batch_inputs = gpu_trace.train_inputs[batch_indices]
            batch_targets = gpu_trace.train_targets[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            prediction_error = model(batch_inputs) - batch_targets
            loss = torch.mean(prediction_error.square())
            loss.backward()
            optimizer.step()
            training_squared_error.add_(prediction_error.detach().square().sum())

        model.eval()
        train_mse = float((training_squared_error / gpu_trace.train_targets.numel()).item())
        validation_mse = compute_validation_mse(model, gpu_trace, args.batch_size)
        epochs_run = epoch
        if validation_mse < best_validation_mse - args.min_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path(output_dir, intermediate_size, expert_id),
                model,
                expert_id,
                intermediate_size,
                best_epoch,
                best_validation_mse,
                epochs_run,
                int(gpu_trace.train_inputs.shape[0]),
                int(gpu_trace.validation_inputs.shape[0]),
            )
        else:
            epochs_without_improvement += 1

        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": validation_mse})
        print(f"E{expert_id} Tiny SwiGLU-{intermediate_size} epoch {epoch}/{args.max_epochs}:")
        print(f"  train_mse={train_mse:.8f}")
        print(f"  val_mse={validation_mse:.8f}")
        print(f"  best_val_mse={best_validation_mse:.8f}")
        print(f"  patience={epochs_without_improvement}/{args.early_stopping_patience}")
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            break

    finalize_checkpoint(
        checkpoint_path(output_dir, intermediate_size, expert_id),
        epochs_run,
        early_stopped,
    )
    del model, optimizer

    return {
        "expert_id": expert_id,
        "surrogate_type": "tiny_swiglu",
        "intermediate_size": intermediate_size,
        "best_epoch": best_epoch,
        "best_val_mse": best_validation_mse,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "train_samples": int(gpu_trace.train_inputs.shape[0]),
        "val_samples": int(gpu_trace.validation_inputs.shape[0]),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not (args.trace_dir / "metadata.json").is_file():
        raise FileNotFoundError(f"missing trace metadata: {args.trace_dir / 'metadata.json'}")
    output_dir = args.output_dir or args.trace_dir / "checkpoints_tiny_swiglu"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "trace_dir": str(args.trace_dir),
        "surrogate_type": "tiny_swiglu",
        "hidden_sizes": args.hidden_sizes,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "optimization_dtype": "fp32",
        "configurations": [],
    }
    for expert_id in args.experts:
        train_trace = load_trace(args.trace_dir / "train" / f"expert_{expert_id}.pt")
        validation_trace = load_trace(args.trace_dir / "val" / f"expert_{expert_id}.pt")
        gpu_trace = prepare_gpu_trace(train_trace, validation_trace)
        try:
            for intermediate_size in args.hidden_sizes:
                print(f"Training E{expert_id} Tiny SwiGLU-{intermediate_size}")
                summary["configurations"].append(
                    train_one_configuration(
                        args,
                        gpu_trace,
                        expert_id,
                        intermediate_size,
                        output_dir,
                    )
                )
        finally:
            del gpu_trace
            torch.cuda.empty_cache()
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved Tiny SwiGLU checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()

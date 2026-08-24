#!/usr/bin/env python3
"""Train independent full-rank FP32 linear surrogates for Mixtral experts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


HIDDEN_SIZE = 4096
NUM_EXPERTS = 8
TraceData = dict[str, object]


def parse_args() -> argparse.Namespace:
    """Parse training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("results/surrogate_per_expert/layer_16"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experts", type=int, nargs="+", default=list(range(NUM_EXPERTS)))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments before allocating a surrogate."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for surrogate training")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be positive")
    if args.min_delta < 0.0:
        raise ValueError("--min-delta must be non-negative")
    if sorted(set(args.experts)) != sorted(args.experts):
        raise ValueError("--experts must not contain duplicates")
    if any(expert_id < 0 or expert_id >= NUM_EXPERTS for expert_id in args.experts):
        raise ValueError("--experts must contain IDs in [0, 7]")


def set_deterministic_seed(seed: int) -> None:
    """Configure deterministic random sources for independent expert training."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_trace(path: Path) -> TraceData:
    """Load and validate one non-empty CPU BF16 trace file."""
    trace_data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(trace_data, dict):
        raise TypeError(f"trace must be a dictionary: {path}")
    inputs = trace_data.get("x")
    targets = trace_data.get("y")
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError(f"trace does not contain tensor x/y: {path}")
    if inputs.shape != targets.shape or inputs.ndim != 2 or inputs.shape[1] != HIDDEN_SIZE:
        raise ValueError(f"trace must contain matching [N, {HIDDEN_SIZE}] x/y tensors: {path}")
    if inputs.shape[0] == 0:
        raise ValueError(f"trace has no routed samples: {path}")
    if inputs.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError(f"trace tensors must reside on CPU: {path}")
    return trace_data


def trace_tensors(trace_data: TraceData) -> tuple[torch.Tensor, torch.Tensor]:
    """Return input and target tensors from a validated trace dictionary."""
    inputs = trace_data["x"]
    targets = trace_data["y"]
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise AssertionError("validated trace unexpectedly lacks tensors")
    return inputs, targets


@torch.no_grad()
def compute_mse(model: nn.Module, trace_data: TraceData, batch_size: int) -> float:
    """Calculate global MSE using FP32 inputs, targets, and model parameters."""
    inputs, targets = trace_tensors(trace_data)
    squared_error_sum = 0.0
    element_count = 0
    for start_index in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start_index : start_index + batch_size].to("cuda", dtype=torch.float32)
        batch_targets = targets[start_index : start_index + batch_size].to("cuda", dtype=torch.float32)
        predictions = model(batch_inputs)
        squared_error_sum += float((predictions - batch_targets).square().sum().item())
        element_count += batch_targets.numel()
    return squared_error_sum / element_count


def save_checkpoint(
    path: Path,
    *,
    expert_id: int,
    model: nn.Linear,
    best_epoch: int,
    best_validation_mse: float,
    train_samples: int,
    validation_samples: int,
) -> None:
    """Persist the FP32 state associated with the validation minimum."""
    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    torch.save(
        {
            "expert_id": expert_id,
            "hidden_size": HIDDEN_SIZE,
            "state_dict": state_dict,
            "best_epoch": best_epoch,
            "best_val_mse": best_validation_mse,
            "train_samples": train_samples,
            "val_samples": validation_samples,
            "optimization_dtype": "fp32",
        },
        path,
    )


def train_one_expert(args: argparse.Namespace, expert_id: int, checkpoint_dir: Path) -> dict[str, Any]:
    """Train one expert until validation early stopping selects its best epoch."""
    train_trace = load_trace(args.trace_dir / "train" / f"expert_{expert_id}.pt")
    validation_trace = load_trace(args.trace_dir / "val" / f"expert_{expert_id}.pt")
    train_inputs, train_targets = trace_tensors(train_trace)
    validation_inputs, _ = trace_tensors(validation_trace)

    model = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True, device="cuda", dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    sample_generator = torch.Generator(device="cpu").manual_seed(args.seed + expert_id)
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_run = 0
    early_stopped = False
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        sample_order = torch.randperm(train_inputs.shape[0], generator=sample_generator)
        for start_index in range(0, sample_order.numel(), args.batch_size):
            batch_indices = sample_order[start_index : start_index + args.batch_size]
            batch_inputs = train_inputs[batch_indices].to("cuda", dtype=torch.float32)
            batch_targets = train_targets[batch_indices].to("cuda", dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(batch_inputs) - batch_targets).square())
            loss.backward()
            optimizer.step()

        model.eval()
        train_mse = compute_mse(model, train_trace, args.batch_size)
        validation_mse = compute_mse(model, validation_trace, args.batch_size)
        epochs_run = epoch
        if validation_mse < best_validation_mse - args.min_delta:
            best_validation_mse = validation_mse
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / f"expert_{expert_id}.pt",
                expert_id=expert_id,
                model=model,
                best_epoch=best_epoch,
                best_validation_mse=best_validation_mse,
                train_samples=int(train_inputs.shape[0]),
                validation_samples=int(validation_inputs.shape[0]),
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": validation_mse,
                "best_val_mse": best_validation_mse,
                "patience": epochs_without_improvement,
            }
        )
        print(f"E{expert_id} epoch {epoch}/{args.max_epochs}:")
        print(f"  train_mse={train_mse:.6e}")
        print(f"  val_mse={validation_mse:.6e}")
        print(f"  best_val_mse={best_validation_mse:.6e}")
        print(f"  patience={epochs_without_improvement}/{args.early_stopping_patience}")
        if epochs_without_improvement >= args.early_stopping_patience:
            early_stopped = True
            break

    return {
        "expert_id": expert_id,
        "best_epoch": best_epoch,
        "best_val_mse": best_validation_mse,
        "epochs_run": epochs_run,
        "early_stopped": early_stopped,
        "train_samples": int(train_inputs.shape[0]),
        "val_samples": int(validation_inputs.shape[0]),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    metadata_path = args.trace_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing trace metadata: {metadata_path}")

    set_deterministic_seed(args.seed)
    checkpoint_dir = args.output_dir or args.trace_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "trace_dir": str(args.trace_dir),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
        "seed": args.seed,
        "optimization_dtype": "fp32",
        "experts": [],
    }
    for expert_id in args.experts:
        print(f"Training independent FP32 linear surrogate for expert {expert_id}")
        summary["experts"].append(train_one_expert(args, expert_id, checkpoint_dir))

    (checkpoint_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved best checkpoints to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

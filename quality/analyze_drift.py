"""Plot teacher-forced expert-quantization drift results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {8: "#4C78A8", 4: "#F58518", 3: "#E45756", 16: "#54A24B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--margin-bins", type=int, default=12)
    return parser.parse_args()


def read_rows(input_dirs: list[Path], name: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for directory in input_dirs:
        with (directory / name).open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {name}")
    return rows


def label(bits: int) -> str:
    return "BF16" if bits == 16 else f"W{bits}A16"


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(values)
    totals = np.convolve(np.where(valid, values, 0.0), np.ones(window), mode="same")
    counts = np.convolve(valid.astype(float), np.ones(window), mode="same")
    return np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)


def per_position_mean(rows: list[dict[str, str]], field: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["quant_bits"]), int(row["decode_position"]))].append(float(row[field]))
    by_bits: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for (bits, position), values in grouped.items():
        by_bits[bits][position] = values
    output: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for bits, positions in by_bits.items():
        x = np.array(sorted(positions))
        means: list[float] = []
        for position in x:
            values = np.asarray(positions[position], dtype=float)
            finite = values[np.isfinite(values)]
            means.append(float(finite.mean()) if finite.size else math.nan)
        y = np.array(means)
        output[bits] = x, y
    return output


def line_plot(
    series: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    window: int,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    for bits in sorted(series, reverse=True):
        x, y = series[bits]
        axis.plot(x, rolling_mean(y, window), label=label(bits), color=COLORS.get(bits), linewidth=2)
    axis.set_xlabel("Decode position")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_margin_flip(layer_rows: list[dict[str, str]], bins: int, output: Path) -> None:
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in layer_rows:
        grouped[int(row["quant_bits"])].append(
            (float(row["router_margin_fp"]), float(row["routing_drift"]) > 0.0)
        )
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    for bits in sorted(grouped, reverse=True):
        margins = np.array([item[0] for item in grouped[bits]])
        flips = np.array([item[1] for item in grouped[bits]], dtype=float)
        edges = np.linspace(margins.min(), margins.max(), bins + 1)
        indices = np.clip(np.digitize(margins, edges) - 1, 0, bins - 1)
        x, y = [], []
        for bin_index in range(bins):
            mask = indices == bin_index
            if mask.any():
                x.append(margins[mask].mean())
                y.append(flips[mask].mean())
        axis.plot(x, y, marker="o", label=label(bits), color=COLORS.get(bits), linewidth=1.8)
    axis.set_xlabel("FP router margin (2nd logit − 3rd logit)")
    axis.set_ylabel("Route-flip probability")
    axis.set_ylim(bottom=0)
    axis.set_title("Router Margin vs Route-Flip Probability")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_ppl_ratio(token_rows: list[dict[str, str]], window: int, output: Path) -> None:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in token_rows:
        bits = int(row["quant_bits"])
        position = int(row["decode_position"])
        start = ((position - 1) // window) * window + 1
        grouped[(bits, start)].append(float(row["delta_nll"]))
    by_bits: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (bits, start), deltas in grouped.items():
        by_bits[bits].append((start, math.exp(float(np.mean(deltas)))))
    fig, axis = plt.subplots(figsize=(8.2, 4.2))
    for bits in sorted(by_bits, reverse=True):
        values = sorted(by_bits[bits])
        axis.plot(*zip(*values), marker="o", label=label(bits), color=COLORS.get(bits), linewidth=1.8)
    axis.axhline(1.0, color="#444444", linewidth=1, linestyle="--")
    axis.set_xlabel("Decode-position window start")
    axis.set_ylabel("PPL_Q / PPL_FP")
    axis.set_title(f"PPL Ratio by {window}-Token Window")
    axis.grid(alpha=0.3, linestyle="--")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window < 1 or args.margin_bins < 1:
        raise ValueError("--window and --margin-bins must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_rows = read_rows(args.input_dirs, "token_metrics.csv")
    layer_rows = read_rows(args.input_dirs, "layer_metrics.csv")
    line_plot(
        per_position_mean(token_rows, "routing_drift_mean"),
        window=args.window,
        ylabel="Mean routing drift",
        title="Decode Position vs Routing Drift",
        output=args.output_dir / "routing_drift.png",
    )
    line_plot(
        per_position_mean(token_rows, "hidden_rel_l2_mean"),
        window=args.window,
        ylabel="Mean hidden relative L2",
        title="Decode Position vs Hidden-State Divergence",
        output=args.output_dir / "hidden_rel_l2.png",
    )
    line_plot(
        per_position_mean(token_rows, "router_logit_rel_l2_mean"),
        window=args.window,
        ylabel="Mean router-logit relative L2",
        title="Decode Position vs Router-Logit Divergence",
        output=args.output_dir / "router_logit_rel_l2.png",
    )
    line_plot(
        per_position_mean(token_rows, "logit_kl"),
        window=args.window,
        ylabel="KL(p_FP || p_Q)",
        title="Decode Position vs Logit KL Divergence",
        output=args.output_dir / "logit_kl.png",
    )
    line_plot(
        per_position_mean(token_rows, "delta_nll"),
        window=args.window,
        ylabel="NLL_Q − NLL_FP",
        title="Decode Position vs Delta NLL",
        output=args.output_dir / "delta_nll.png",
    )
    plot_margin_flip(layer_rows, args.margin_bins, args.output_dir / "margin_route_flip.png")
    plot_ppl_ratio(token_rows, args.window, args.output_dir / "ppl_ratio.png")
    print(f"Saved drift analysis plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

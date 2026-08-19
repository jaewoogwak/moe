"""Render a horizontal stacked TPOT breakdown from decode_summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


COMPONENTS = (
    ("attention_ms", "Attention", "#F6C445"),
    ("router_ms", "Router", "#F28E2B"),
    ("expert_h2d_ms", "Expert H2D", "#274C9A"),
    ("expert_compute_ms", "Expert Compute", "#79A7D3"),
    ("other_ms", "Other", "#B8B8B8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as file:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows:
        raise ValueError(f"no results in {path}")
    return sorted(rows, key=lambda row: row["requested_context_length"])


def context_label(row: dict[str, float]) -> str:
    return f"{int(row['requested_context_length']) // 1024}K"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    # Put 4K at the top, matching the comparison-oriented horizontal-bar style.
    rows = list(reversed(rows))
    labels = [context_label(row) for row in rows]
    positions = list(range(len(rows)))

    fig, axis = plt.subplots(figsize=(10.5, 4.9), constrained_layout=False)
    left = [0.0] * len(rows)
    for field, label, color in COMPONENTS:
        values = [row[field] for row in rows]
        axis.barh(
            positions,
            values,
            left=left,
            height=0.56,
            color=color,
            edgecolor="#202020",
            linewidth=0.8,
            label=label,
        )
        left = [start + value for start, value in zip(left, values)]

    maximum = max(row["total_ms"] for row in rows)
    axis.set_xlim(0, maximum * 1.15)
    for position, row in zip(positions, rows):
        axis.text(
            row["total_ms"] + maximum * 0.012,
            position,
            f"TPOT {row['total_ms']:.0f} ms",
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    axis.set_yticks(positions, labels)
    axis.set_xlabel("Latency per output token (ms)")
    axis.xaxis.set_major_locator(MultipleLocator(100))
    axis.xaxis.grid(True, color="#C9C9C9", linestyle="--", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#222222")

    axis.legend(
        ncol=len(COMPONENTS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        frameon=False,
        fontsize=9,
        handlelength=1.25,
        columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.13, right=0.95, top=0.82, bottom=0.13)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()

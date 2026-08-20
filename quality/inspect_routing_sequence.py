#!/usr/bin/env python3
"""Inspect FP versus quantized Mixtral expert selections from layer_metrics.csv.

This is an offline inspection tool: it only reads the CSV emitted by
``quality/decode_drift_benchmark.py`` and never loads a model or runs inference.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


NUM_LAYERS = 32
NUM_EXPERTS = 8
REQUIRED_COLUMNS = {
    "sample_id",
    "quant_bits",
    "decode_position",
    "layer_id",
    "routing_drift",
    "fp_top1",
    "fp_top2",
    "q_top1",
    "q_top2",
}
SUMMARY_COLUMNS = (
    "decode_position",
    "first_flip_layer",
    "last_flip_layer",
    "num_flipped_layers",
    "num_partial_flips",
    "num_full_flips",
    "num_order_swaps",
)


@dataclass(frozen=True)
class LayerSelection:
    layer_id: int
    fp_top2: tuple[int, int]
    q_top2: tuple[int, int]
    routing_drift: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Benchmark layer_metrics.csv")
    parser.add_argument("--sample-id", required=True, help="Sample ID to inspect")
    parser.add_argument("--position", type=int, help="One 1-indexed decode position to print")
    parser.add_argument("--only-flips", action="store_true", help="Hide SAME layers in single-token output")
    parser.add_argument("--summary", action="store_true", help="Summarize every decode position for the sample")
    parser.add_argument("--output-csv", type=Path, help="Write per-position summary CSV (requires --summary)")
    args = parser.parse_args()
    if not args.summary and args.position is None:
        parser.error("--position is required unless --summary is supplied")
    if args.summary and args.position is not None:
        parser.error("use either --position or --summary, not both")
    if args.output_csv is not None and not args.summary:
        parser.error("--output-csv requires --summary")
    return args


def classification(selection: LayerSelection) -> str:
    if selection.fp_top2 == selection.q_top2:
        return "SAME"
    if set(selection.fp_top2) == set(selection.q_top2):
        return "ORDER_SWAP"
    if set(selection.fp_top2).intersection(selection.q_top2):
        return "PARTIAL_FLIP"
    return "FULL_FLIP"


def expected_routing_drift(status: str) -> float:
    return {"SAME": 0.0, "ORDER_SWAP": 0.0, "PARTIAL_FLIP": 0.5, "FULL_FLIP": 1.0}[status]


def load_sample(csv_path: Path, sample_id: str) -> tuple[int, dict[int, list[LayerSelection]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"layer metrics CSV does not exist: {csv_path}")

    groups: dict[int, dict[int, LayerSelection]] = defaultdict(dict)
    available_samples: set[str] = set()
    quant_bits: set[int] = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
        for row in reader:
            row_sample_id = row["sample_id"]
            available_samples.add(row_sample_id)
            if row_sample_id != sample_id:
                continue
            try:
                position = int(row["decode_position"])
                layer_id = int(row["layer_id"])
                selection = LayerSelection(
                    layer_id=layer_id,
                    fp_top2=(int(row["fp_top1"]), int(row["fp_top2"])),
                    q_top2=(int(row["q_top1"]), int(row["q_top2"])),
                    routing_drift=float(row["routing_drift"]),
                )
                quant_bits.add(int(row["quant_bits"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid numeric value in row: {row}") from exc
            if layer_id in groups[position]:
                raise ValueError(
                    f"duplicate row for sample={sample_id}, position={position}, layer={layer_id}; "
                    "use a CSV containing one quantization configuration"
                )
            groups[position][layer_id] = selection

    if not groups:
        examples = ", ".join(sorted(available_samples)[:8])
        raise ValueError(f"sample_id '{sample_id}' not found in {csv_path}; available examples: {examples}")
    if len(quant_bits) != 1:
        raise ValueError(f"sample_id '{sample_id}' has multiple quant_bits values: {sorted(quant_bits)}")

    validated: dict[int, list[LayerSelection]] = {}
    expected_layers = set(range(NUM_LAYERS))
    for position, per_layer in sorted(groups.items()):
        layer_ids = set(per_layer)
        if layer_ids != expected_layers:
            raise AssertionError(
                f"sample={sample_id}, position={position} must contain exactly layers 0..31; "
                f"found {sorted(layer_ids)}"
            )
        ordered = [per_layer[layer_id] for layer_id in range(NUM_LAYERS)]
        for selection in ordered:
            for expert_id in (*selection.fp_top2, *selection.q_top2):
                if not 0 <= expert_id < NUM_EXPERTS:
                    raise AssertionError(
                        f"sample={sample_id}, position={position}, layer={selection.layer_id}: "
                        f"expert ID {expert_id} is outside 0..7"
                    )
            if selection.fp_top2[0] == selection.fp_top2[1] or selection.q_top2[0] == selection.q_top2[1]:
                raise AssertionError(
                    f"sample={sample_id}, position={position}, layer={selection.layer_id}: "
                    "top-2 expert IDs must be distinct"
                )
            status = classification(selection)
            if selection.layer_id == 0 and status in {"PARTIAL_FLIP", "FULL_FLIP"}:
                raise AssertionError(
                    f"layer 0 route changed under expert-only quantization: sample={sample_id}, "
                    f"position={position}, fp={selection.fp_top2}, q={selection.q_top2}"
                )
            expected = expected_routing_drift(status)
            if not math.isclose(selection.routing_drift, expected, abs_tol=1e-8):
                raise AssertionError(
                    f"routing_drift disagrees with route-overlap definition: sample={sample_id}, "
                    f"position={position}, layer={selection.layer_id}, status={status}, "
                    f"csv={selection.routing_drift}, expected={expected}"
                )
        validated[position] = ordered
    return quant_bits.pop(), validated


def summarize(position: int, selections: list[LayerSelection]) -> dict[str, int | str]:
    statuses = [classification(selection) for selection in selections]
    flipped_layers = [selection.layer_id for selection, status in zip(selections, statuses) if status in {"PARTIAL_FLIP", "FULL_FLIP"}]
    return {
        "decode_position": position,
        "first_flip_layer": flipped_layers[0] if flipped_layers else "",
        "last_flip_layer": flipped_layers[-1] if flipped_layers else "",
        "num_flipped_layers": len(flipped_layers),
        "num_partial_flips": statuses.count("PARTIAL_FLIP"),
        "num_full_flips": statuses.count("FULL_FLIP"),
        "num_order_swaps": statuses.count("ORDER_SWAP"),
    }


def print_position(sample_id: str, position: int, quant_bits: int, selections: list[LayerSelection], only_flips: bool) -> None:
    print(f"Sample: {sample_id}")
    print(f"Decode position: {position}")
    print()
    print(f"{'Layer':<8}{'FP Top-2':<12}{f'W{quant_bits} Top-2':<12}Status")
    print("-" * 44)
    for selection in selections:
        status = classification(selection)
        if only_flips and status == "SAME":
            continue
        fp = f"[{selection.fp_top2[0]},{selection.fp_top2[1]}]"
        quantized = f"[{selection.q_top2[0]},{selection.q_top2[1]}]"
        print(f"L{selection.layer_id:02d}     {fp:<12}{quantized:<12}{status}")

    values = summarize(position, selections)
    print()
    print(f"flipped layers / {NUM_LAYERS}: {values['num_flipped_layers']}")
    print(f"order-swapped layers: {values['num_order_swaps']}")
    print(f"partial-flip layers: {values['num_partial_flips']}")
    print(f"full-flip layers: {values['num_full_flips']}")
    print(f"first flipped layer: {values['first_flip_layer'] if values['first_flip_layer'] != '' else 'none'}")
    print(f"last flipped layer: {values['last_flip_layer'] if values['last_flip_layer'] != '' else 'none'}")


def write_summary(output_csv: Path, rows: list[dict[str, int | str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(sample_id: str, rows: list[dict[str, int | str]], output_csv: Path | None) -> None:
    positions_with_flips = sum(int(row["num_flipped_layers"]) > 0 for row in rows)
    print(f"Sample: {sample_id}")
    print(f"Decode positions: {len(rows)}")
    print(f"Positions with routing flips: {positions_with_flips}")
    print(f"Total partial flips: {sum(int(row['num_partial_flips']) for row in rows)}")
    print(f"Total full flips: {sum(int(row['num_full_flips']) for row in rows)}")
    print(f"Total order swaps: {sum(int(row['num_order_swaps']) for row in rows)}")
    if output_csv is not None:
        print(f"Saved per-position routing summary to: {output_csv}")
        return
    print()
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    args = parse_args()
    quant_bits, positions = load_sample(args.csv, args.sample_id)
    if args.summary:
        rows = [summarize(position, selections) for position, selections in positions.items()]
        if args.output_csv is not None:
            write_summary(args.output_csv, rows)
        print_summary(args.sample_id, rows, args.output_csv)
        return
    if args.position not in positions:
        available = f"{min(positions)}..{max(positions)}"
        raise ValueError(f"decode position {args.position} is unavailable for sample={args.sample_id}; range is {available}")
    print_position(args.sample_id, args.position, quant_bits, positions[args.position], args.only_flips)


if __name__ == "__main__":
    main()

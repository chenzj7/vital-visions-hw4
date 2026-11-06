#!/usr/bin/env python3
"""Analyze the ordered 48-video HW4 alpha-BFSK dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bfsk import decode_known_frame, encode_frame
from receiver import extract_trace, remove_slow_drift


DISTANCES_CM = [20, 40, 80, 120]
DELTA_ALPHAS = [0.1, 0.5]
BIT_RATES = [1, 2, 4, 6, 8, 10]


def build_manifest(dataset_dir: Path):
    videos = sorted(dataset_dir.glob("IMG_*.MOV"))
    expected_count = len(DISTANCES_CM) * len(DELTA_ALPHAS) * len(BIT_RATES)
    if len(videos) != expected_count:
        raise ValueError(f"Expected {expected_count} IMG_*.MOV files, found {len(videos)}.")
    manifest = []
    index = 0
    for distance in DISTANCES_CM:
        for delta_alpha in DELTA_ALPHAS:
            for bit_rate in BIT_RATES:
                manifest.append(
                    {
                        "sequence_index": index + 1,
                        "video": videos[index],
                        "distance_cm": distance,
                        "delta_alpha": delta_alpha,
                        "requested_bit_rate_bps": bit_rate,
                    }
                )
                index += 1
    return manifest


def analyze(manifest, message: str, roi, output_dir: Path):
    expected_bits = encode_frame(message)
    rows = []
    per_video_dir = output_dir / "decoded"
    per_video_dir.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        video = item["video"]
        bit_rate = float(item["requested_bit_rate_bps"])
        print(
            f"[{item['sequence_index']:02d}/{len(manifest)}] {video.name}: "
            f"d={item['distance_cm']} alpha={item['delta_alpha']} rate={bit_rate:g}",
            flush=True,
        )
        trace, fps, resolved_roi = extract_trace(video, None, 30.0, roi)
        filtered = remove_slow_drift(trace, fps, bit_rate)
        aligned = decode_known_frame(filtered, fps, bit_rate, expected_bits)
        frame = aligned["frame"]
        crc_ok = bool(frame is not None and frame["crc_ok"])
        decoded_message = "" if frame is None else str(frame["message"])
        row = {
            "sequence_index": item["sequence_index"],
            "video": video.name,
            "distance_cm": item["distance_cm"],
            "delta_alpha": item["delta_alpha"],
            "requested_bit_rate_bps": bit_rate,
            "measured_fps": float(fps),
            "duration_seconds": float(trace.size / fps),
            "frame_bits": int(aligned["total_bits"]),
            "bit_errors": int(aligned["bit_errors"]),
            "missing_bits": int(aligned["missing_bits"]),
            "ber": float(aligned["ber"]),
            "correct_bit_rate_bps": float(bit_rate * (1.0 - aligned["ber"])),
            "crc_ok": crc_ok,
            "decoded_message": decoded_message,
            "frame_start_seconds": float(aligned["start_seconds"]),
            "mean_symbol_confidence": float(aligned["mean_symbol_confidence"]),
            "roi_xywh": ",".join(str(value) for value in resolved_roi),
            "result_type": "hardware_measurement",
        }
        rows.append(row)
        (per_video_dir / f"{video.stem}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(
            f"    BER={row['ber']:.4f}, errors={row['bit_errors']}/{row['frame_bits']}, "
            f"CRC={'pass' if crc_ok else 'fail'}",
            flush=True,
        )
    return rows


def save_csv(rows, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "part3_measured_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for distance in DISTANCES_CM:
        for delta_alpha in DELTA_ALPHAS:
            subset = [
                row
                for row in rows
                if row["distance_cm"] == distance and row["delta_alpha"] == delta_alpha
            ]
            reliable = [row for row in subset if row["ber"] <= 0.01]
            best = max(reliable, key=lambda row: row["requested_bit_rate_bps"]) if reliable else None
            summary.append(
                {
                    "distance_cm": distance,
                    "delta_alpha": delta_alpha,
                    "max_rate_at_ber_le_1pct_bps": "none" if best is None else best["requested_bit_rate_bps"],
                    "ber_at_max_reliable_rate": "n/a" if best is None else f"{best['ber']:.4f}",
                    "crc_pass_count": sum(int(row["crc_ok"]) for row in subset),
                    "conditions_tested": len(subset),
                }
            )
    with (output_dir / "part3_measured_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def save_plot(rows, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {20: "#1B6CA8", 40: "#3A9D6F", 80: "#E09F3E", 120: "#C43D3D"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, delta_alpha in zip(axes, DELTA_ALPHAS):
        for distance in DISTANCES_CM:
            subset = sorted(
                [
                    row
                    for row in rows
                    if row["delta_alpha"] == delta_alpha and row["distance_cm"] == distance
                ],
                key=lambda row: row["requested_bit_rate_bps"],
            )
            axis.plot(
                [row["requested_bit_rate_bps"] for row in subset],
                [row["ber"] for row in subset],
                marker="o",
                linewidth=1.7,
                color=colors[distance],
                label=f"{distance} cm",
            )
        axis.axhline(0.01, color="0.35", linestyle="--", linewidth=1)
        axis.set(
            title=f"delta alpha = {delta_alpha}",
            xlabel="Requested raw bit rate (bit/s)",
            ylim=(-0.025, 0.58),
            xticks=BIT_RATES,
        )
    axes[0].set_ylabel("Measured bit error rate")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Measured alpha-BFSK screen-camera performance (one recording per condition)")
    fig.tight_layout()
    fig.savefig(output_dir / "part3_ber_measured.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--message", default="VITAL")
    parser.add_argument("--roi", default="100,200,250,350")
    parser.add_argument("--output-dir", type=Path, default=Path("part3/results_measured"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi_values = tuple(int(value) for value in args.roi.split(","))
    if len(roi_values) != 4:
        raise ValueError("ROI must be x,y,width,height.")
    manifest = build_manifest(args.dataset_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["sequence_index", "video", "distance_cm", "delta_alpha", "requested_bit_rate_bps"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in manifest:
            writer.writerow({**item, "video": item["video"].name})
    rows = analyze(manifest, args.message, roi_values, args.output_dir)
    save_csv(rows, args.output_dir)
    save_plot(rows, args.output_dir)
    print(f"Wrote measured results for {len(rows)} videos to {args.output_dir}")


if __name__ == "__main__":
    main()

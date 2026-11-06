#!/usr/bin/env python3
"""Run the required delta-alpha/distance BFSK grid in a simulated channel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bfsk import bit_error_rate, decode_symbols, simulate_channel


DISTANCES_CM = [20, 40, 80, 120]
DELTA_ALPHAS = [0.1, 0.5]
RAW_BIT_RATES = [1, 2, 4, 6, 8, 10]


def run_grid(trials: int, payload_bits: int, fps: float):
    rows = []
    for delta_alpha in DELTA_ALPHAS:
        for distance_cm in DISTANCES_CM:
            for requested_rate in RAW_BIT_RATES:
                trial_bers = []
                trial_goodputs = []
                effective_rate = None
                for trial in range(trials):
                    seed = int(100000 * delta_alpha + 1000 * distance_cm + 10 * requested_rate + trial)
                    rng = np.random.default_rng(seed)
                    bits = rng.integers(0, 2, size=payload_bits, dtype=np.uint8)
                    observed, parameters = simulate_channel(bits, fps, requested_rate, delta_alpha, distance_cm, seed + 1)
                    decoded, _ = decode_symbols(observed, fps, requested_rate)
                    ber = bit_error_rate(bits, decoded)
                    effective_rate = parameters["effective_bit_rate"]
                    trial_bers.append(ber)
                    trial_goodputs.append(effective_rate * (1.0 - ber))
                rows.append(
                    {
                        "delta_alpha": delta_alpha,
                        "distance_cm": distance_cm,
                        "requested_raw_bit_rate_bps": requested_rate,
                        "effective_raw_bit_rate_bps": effective_rate,
                        "mean_ber": float(np.mean(trial_bers)),
                        "std_ber": float(np.std(trial_bers)),
                        "mean_correct_bit_rate_bps": float(np.mean(trial_goodputs)),
                        "trials": trials,
                        "result_type": "simulation_not_hardware_measurement",
                    }
                )
    return rows


def save_results(rows, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "part3_simulated_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for delta_alpha in DELTA_ALPHAS:
        for distance_cm in DISTANCES_CM:
            subset = [r for r in rows if r["delta_alpha"] == delta_alpha and r["distance_cm"] == distance_cm]
            reliable = [r for r in subset if r["mean_ber"] <= 0.01]
            best = max(reliable, key=lambda r: r["effective_raw_bit_rate_bps"]) if reliable else None
            summary_rows.append(
                {
                    "delta_alpha": delta_alpha,
                    "distance_cm": distance_cm,
                    "max_rate_at_ber_le_1pct_bps": "none" if best is None else f"{best['effective_raw_bit_rate_bps']:.3f}",
                    "ber_at_max_reliable_rate": "n/a" if best is None else f"{best['mean_ber']:.4f}",
                }
            )
    with (output_dir / "part3_simulated_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    colors = {20: "#1B6CA8", 40: "#3A9D6F", 80: "#E09F3E", 120: "#C43D3D"}
    for axis, delta_alpha in zip(axes, DELTA_ALPHAS):
        for distance_cm in DISTANCES_CM:
            subset = sorted(
                [r for r in rows if r["delta_alpha"] == delta_alpha and r["distance_cm"] == distance_cm],
                key=lambda r: r["effective_raw_bit_rate_bps"],
            )
            axis.errorbar(
                [r["effective_raw_bit_rate_bps"] for r in subset],
                [r["mean_ber"] for r in subset],
                yerr=[r["std_ber"] for r in subset],
                marker="o",
                capsize=2,
                linewidth=1.5,
                color=colors[distance_cm],
                label=f"{distance_cm} cm",
            )
        axis.axhline(0.01, color="0.35", linestyle="--", linewidth=1, label="1% BER threshold" if delta_alpha == DELTA_ALPHAS[0] else None)
        axis.set(title=f"delta alpha = {delta_alpha}", xlabel="Effective raw bit rate (bit/s)", ylim=(-0.02, 0.58))
    axes[0].set_ylabel("Bit error rate")
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle("Simulated alpha-BFSK camera channel (not hardware measurements)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "part3_ber_simulation.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--payload-bits", type=int, default=160)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=Path("part3/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_grid(args.trials, args.payload_bits, args.fps)
    save_results(rows, args.output_dir)
    print(f"Wrote {len(rows)} conditions to {args.output_dir}")


if __name__ == "__main__":
    main()


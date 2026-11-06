#!/usr/bin/env python3
"""Decode an alpha-channel BFSK message from a video file or live camera."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from bfsk import decode_best_alignment, decode_known_frame, encode_frame, modulation_parameters


def parse_roi(value: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if value is None:
        return None
    fields = [int(item) for item in value.split(",")]
    if len(fields) != 4 or fields[2] <= 0 or fields[3] <= 0:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height with positive size.")
    return tuple(fields)  # type: ignore[return-value]


def extract_trace(
    video: Optional[Path], camera_index: Optional[int], seconds: float, roi
):
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("receiver.py requires opencv-python.") from exc

    source = str(video) if video is not None else int(camera_index or 0)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open input: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1:
        fps = 30.0
    maximum_frames = int(seconds * fps) if video is None else None
    values = []
    resolved_roi = roi
    while True:
        ok, frame = capture.read()
        if not ok or (maximum_frames is not None and len(values) >= maximum_frames):
            break
        height, width = frame.shape[:2]
        if resolved_roi is None:
            resolved_roi = (width // 4, height // 4, width // 2, height // 2)
        x, y, w, h = resolved_roi
        crop = frame[max(0, y) : min(height, y + h), max(0, x) : min(width, x + w)]
        if crop.size == 0:
            capture.release()
            raise ValueError("ROI lies outside the video frame.")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        values.append(float(np.mean(gray)))
    capture.release()
    return np.asarray(values, dtype=float), fps, resolved_roi


def remove_slow_drift(signal: np.ndarray, fps: float, bit_rate: float) -> np.ndarray:
    cutoff = max(0.25, 0.35 * bit_rate)
    sos = butter(3, cutoff, btype="highpass", fs=fps, output="sos")
    return sosfiltfilt(sos, signal - np.mean(signal))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--camera-index", type=int)
    parser.add_argument("--seconds", type=float, default=30.0, help="live-camera capture duration")
    parser.add_argument("--bit-rate", type=float, required=True)
    parser.add_argument("--delta-alpha", type=float, help="metadata for the experimental condition")
    parser.add_argument("--distance-cm", type=float, help="metadata for the experimental condition")
    parser.add_argument("--roi", type=parse_roi, help="x,y,width,height; default is center 50%%")
    parser.add_argument("--expected-message", help="calculate BER against the known transmitted frame")
    parser.add_argument("--output", type=Path, default=Path("part3/results/decoded.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace, fps, roi = extract_trace(args.video, args.camera_index, args.seconds, args.roi)
    filtered = remove_slow_drift(trace, fps, args.bit_rate)
    samples_per_symbol, effective_rate, f0, f1 = modulation_parameters(fps, args.bit_rate)
    if args.expected_message is not None:
        # The fractional-timing decoder retains the transmitter's requested
        # rate instead of rounding fps/rate to one integer symbol length.
        samples_per_symbol_value = fps / args.bit_rate
        effective_rate = args.bit_rate
        f0, f1 = args.bit_rate, 2.0 * args.bit_rate
    else:
        samples_per_symbol_value = samples_per_symbol
    common = {
        "source": str(args.video) if args.video else f"camera:{args.camera_index}",
        "fps": fps,
        "frames": int(trace.size),
        "roi_xywh": roi,
        "requested_bit_rate_bps": args.bit_rate,
        "effective_bit_rate_bps": effective_rate,
        "frequency_zero_hz": f0,
        "frequency_one_hz": f1,
        "samples_per_symbol": samples_per_symbol_value,
        "delta_alpha": args.delta_alpha,
        "distance_cm": args.distance_cm,
    }
    if args.expected_message is not None:
        expected = encode_frame(args.expected_message)
        aligned = decode_known_frame(filtered, fps, args.bit_rate, expected)
        frame = aligned["frame"]
        result = {
            **common,
            "message": None if frame is None else frame["message"],
            "crc_ok": False if frame is None else bool(frame["crc_ok"]),
            "payload_matches_expected": False if frame is None else frame["message"] == args.expected_message,
            "raw_frame_ber": aligned["ber"],
            "bit_errors": aligned["bit_errors"],
            "missing_bits": aligned["missing_bits"],
            "total_bits": aligned["total_bits"],
            "frame_start_sample": aligned["start_sample"],
            "frame_start_seconds": aligned["start_seconds"],
            "mean_symbol_confidence": aligned["mean_symbol_confidence"],
        }
    else:
        bits, confidence, frame, sample_offset = decode_best_alignment(filtered, fps, args.bit_rate)
        frame.pop("payload_bytes")
        result = {
            **frame,
            **common,
            "sample_alignment_offset": sample_offset,
            "mean_symbol_confidence": float(np.mean(confidence)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


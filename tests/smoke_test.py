#!/usr/bin/env python3
"""Dependency-light correctness checks for Part 1 and Part 3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part1"))
sys.path.insert(0, str(ROOT / "part3"))

from heart_rate import analyze_trace, synthetic_ppg  # noqa: E402
from bfsk import (  # noqa: E402
    bit_error_rate,
    decode_best_alignment,
    decode_known_frame,
    decode_frame,
    decode_symbols,
    encode_frame,
    modulate_bits,
)


def test_heart_rate() -> None:
    trace, fps = synthetic_ppg(heart_rate_bpm=78.0)
    result = analyze_trace(trace, fps)
    assert abs(float(result["fft_hr_bpm"]) - 78.0) <= 2.1
    assert abs(float(result["peak_hr_bpm"]) - 78.0) <= 4.0
    assert 20 <= int(result["peaks"].size) <= 50


def test_codec() -> None:
    bits = encode_frame("VITAL")
    frame = decode_frame(bits)
    assert frame["message"] == "VITAL"
    assert frame["crc_ok"] is True


def test_modem_clean_channel() -> None:
    rng = np.random.default_rng(2)
    bits = rng.integers(0, 2, 80, dtype=np.uint8)
    waveform, _ = modulate_bits(bits, fps=30.0, bit_rate=4.0, delta_alpha=0.1)
    decoded, _ = decode_symbols(waveform, fps=30.0, bit_rate=4.0)
    assert bit_error_rate(bits, decoded) == 0.0


def test_frame_alignment() -> None:
    bits = encode_frame("SENSE")
    waveform, _ = modulate_bits(bits, fps=30.0, bit_rate=2.0, delta_alpha=0.1)
    padded = np.concatenate([np.zeros(7), waveform, np.zeros(4)])
    _, _, frame, _ = decode_best_alignment(padded, fps=30.0, bit_rate=2.0)
    assert frame["message"] == "SENSE"
    assert frame["crc_ok"] is True


def test_fractional_symbol_clock() -> None:
    bits = encode_frame("VITAL")
    fps, rate = 30.0, 4.0
    sample_count = int(round(len(bits) * fps / rate))
    waveform = np.zeros(sample_count, dtype=float)
    for index, bit in enumerate(bits):
        start = int(round(index * fps / rate))
        end = int(round((index + 1) * fps / rate))
        time = np.arange(end - start) / fps
        frequency = rate if bit == 0 else 2.0 * rate
        waveform[start:end] = 0.5 + 0.2 * np.sin(2 * np.pi * frequency * time)
    padded = np.concatenate([np.zeros(11), waveform, np.zeros(7)])
    result = decode_known_frame(padded, fps, rate, bits)
    assert result["ber"] == 0.0
    assert result["frame"]["crc_ok"] is True


if __name__ == "__main__":
    tests = [
        test_heart_rate,
        test_codec,
        test_modem_clean_channel,
        test_frame_alignment,
        test_fractional_symbol_clock,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} smoke tests passed.")


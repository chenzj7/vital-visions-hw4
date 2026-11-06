#!/usr/bin/env python3
"""Smartphone-camera photoplethysmography (PPG) heart-rate analysis.

The module can analyze a real video through OpenCV or generate a deterministic
synthetic trace for an end-to-end smoke test.  The physiological result from a
real recording should not be interpreted as medical advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


NORMAL_HR_BPM = (60.0, 100.0)


def extract_mean_grayscale(
    video_path: Path, discard_seconds: float = 0.0
) -> Tuple[np.ndarray, float]:
    """Return frame-wise mean grayscale intensity and the video's frame rate."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for video input. Install opencv-python or use "
            "--synthetic for the smoke test."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("The video does not report a valid frame rate.")

    discard_frames = int(round(discard_seconds * fps))
    values = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index >= discard_frames:
            # OpenCV decodes as BGR; COLOR_BGR2GRAY is the correct equivalent
            # of rgb2gray for this input representation.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            values.append(float(np.mean(gray)))
        frame_index += 1
    capture.release()

    if len(values) < int(8 * fps):
        raise ValueError("At least 8 seconds of usable video are required.")
    return np.asarray(values, dtype=float), fps


def synthetic_ppg(
    duration_seconds: float = 30.0,
    fps: float = 30.0,
    heart_rate_bpm: float = 78.0,
    seed: int = 7,
) -> Tuple[np.ndarray, float]:
    """Generate a camera-like PPG trace with drift, harmonics, and noise."""
    rng = np.random.default_rng(seed)
    time = np.arange(int(duration_seconds * fps), dtype=float) / fps
    frequency_hz = heart_rate_bpm / 60.0
    pulse = (
        5.0 * np.sin(2 * np.pi * frequency_hz * time + 0.25)
        + 1.4 * np.sin(4 * np.pi * frequency_hz * time + 0.7)
    )
    drift = 3.0 * np.sin(2 * np.pi * 0.12 * time) + 0.035 * time
    sensor_noise = rng.normal(0.0, 0.9, size=time.size)
    return 142.0 + drift + pulse + sensor_noise, fps


def highpass_filter(
    trace: np.ndarray, fps: float, cutoff_hz: float = 0.5, order: int = 3
) -> np.ndarray:
    """Remove the DC component and slow illumination/exposure drift."""
    centered = np.asarray(trace, dtype=float) - float(np.mean(trace))
    if centered.size < max(30, order * 6):
        raise ValueError("Trace is too short for zero-phase filtering.")
    sos = butter(order, cutoff_hz, btype="highpass", fs=fps, output="sos")
    filtered = sosfiltfilt(sos, centered)
    scale = float(np.std(filtered))
    if scale <= np.finfo(float).eps:
        raise ValueError("The intensity trace is constant after filtering.")
    return filtered / scale


def estimate_hr_fft(
    signal: np.ndarray,
    fps: float,
    bpm_range: Tuple[float, float] = NORMAL_HR_BPM,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Estimate heart rate from the maximum FFT amplitude in the BPM band."""
    n = signal.size
    windowed = (signal - np.mean(signal)) * np.hanning(n)
    frequencies = np.fft.rfftfreq(n, d=1.0 / fps)
    amplitude = 2.0 * np.abs(np.fft.rfft(windowed)) / max(np.sum(np.hanning(n)), 1.0)

    low_hz, high_hz = np.asarray(bpm_range) / 60.0
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band):
        raise ValueError("Recording is too short to resolve the requested BPM band.")
    band_indices = np.flatnonzero(band)
    peak_index = int(band_indices[np.argmax(amplitude[band])])
    frequency_hz = float(frequencies[peak_index])
    return 60.0 * frequency_hz, frequency_hz, frequencies, amplitude


def detect_heartbeats(
    signal: np.ndarray, fps: float, fft_hr_bpm: float
) -> Tuple[np.ndarray, float, int]:
    """Find heartbeat peaks using the FFT estimate to set MinPeakDistance."""
    expected_samples = fps * 60.0 / fft_hr_bpm
    # A 75% margin allows normal beat-to-beat variability while suppressing
    # the pulse waveform's harmonic peak.
    min_peak_distance = max(1, int(round(0.75 * expected_samples)))
    prominence = max(0.25, 0.25 * float(np.std(signal)))

    candidates = []
    for polarity in (1.0, -1.0):
        peaks, properties = find_peaks(
            polarity * signal,
            distance=min_peak_distance,
            prominence=prominence,
        )
        if peaks.size >= 3:
            intervals = np.diff(peaks) / fps
            interval_bpm = 60.0 / float(np.median(intervals))
            regularity = float(np.std(intervals) / max(np.mean(intervals), 1e-9))
            hr_error = abs(interval_bpm - fft_hr_bpm) / fft_hr_bpm
            prominence_score = float(np.median(properties["prominences"]))
            candidates.append((hr_error + 0.25 * regularity, -prominence_score, peaks, interval_bpm))

    if not candidates:
        raise ValueError("No stable heartbeat peak sequence was found.")
    _, _, peaks, peak_hr_bpm = min(candidates, key=lambda item: (item[0], item[1]))
    return peaks, float(peak_hr_bpm), min_peak_distance


def analyze_trace(trace: np.ndarray, fps: float) -> Dict[str, object]:
    filtered = highpass_filter(trace, fps)
    fft_hr_bpm, peak_frequency_hz, frequencies, amplitude = estimate_hr_fft(filtered, fps)
    peaks, peak_hr_bpm, min_peak_distance = detect_heartbeats(filtered, fps, fft_hr_bpm)
    return {
        "filtered": filtered,
        "frequencies": frequencies,
        "amplitude": amplitude,
        "peaks": peaks,
        "fft_hr_bpm": fft_hr_bpm,
        "peak_frequency_hz": peak_frequency_hz,
        "peak_hr_bpm": peak_hr_bpm,
        "min_peak_distance_samples": min_peak_distance,
        "samples_per_beat_from_fft": fps / peak_frequency_hz,
    }


def save_results(
    trace: np.ndarray,
    fps: float,
    analysis: Dict[str, object],
    output_dir: Path,
    source_label: str,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frequencies = np.asarray(analysis["frequencies"])
    amplitude = np.asarray(analysis["amplitude"])
    filtered = np.asarray(analysis["filtered"])
    peaks = np.asarray(analysis["peaks"], dtype=int)
    fft_hr_bpm = float(analysis["fft_hr_bpm"])
    peak_hr_bpm = float(analysis["peak_hr_bpm"])
    peak_frequency_hz = float(analysis["peak_frequency_hz"])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    band = frequencies <= 3.0
    ax.plot(frequencies[band], amplitude[band], color="#165D8C", linewidth=1.8)
    ax.axvspan(1.0, 100.0 / 60.0, color="#F4A261", alpha=0.18, label="60-100 bpm search band")
    ax.axvline(peak_frequency_hz, color="#C43D3D", linestyle="--", linewidth=1.5)
    ax.scatter([peak_frequency_hz], [np.interp(peak_frequency_hz, frequencies, amplitude)], color="#C43D3D", zorder=3)
    ax.set(xlabel="Frequency (Hz)", ylabel="Amplitude (a.u.)", title=f"FFT heart-rate estimate: {peak_frequency_hz:.3f} Hz ({fft_hr_bpm:.1f} bpm)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "part1_fft.png", dpi=180)
    plt.close(fig)

    time = np.arange(filtered.size) / fps
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.plot(time, filtered, color="#165D8C", linewidth=1.0, label="Filtered normalized PPG")
    ax.scatter(time[peaks], filtered[peaks], color="#D1495B", s=20, label=f"Selected beats (n={peaks.size})", zorder=3)
    ax.set(xlabel="Time (s)", ylabel="Normalized intensity", title=f"Heartbeat peaks - interval estimate {peak_hr_bpm:.1f} bpm")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "part1_peaks.png", dpi=180)
    plt.close(fig)

    metrics: Dict[str, object] = {
        "source": source_label,
        "fps": float(fps),
        "duration_seconds": float(trace.size / fps),
        "frames_analyzed": int(trace.size),
        "fft_peak_hz": peak_frequency_hz,
        "fft_heart_rate_bpm": fft_hr_bpm,
        "peak_interval_heart_rate_bpm": peak_hr_bpm,
        "selected_peak_count": int(peaks.size),
        "min_peak_distance_samples": int(analysis["min_peak_distance_samples"]),
        "samples_per_beat_from_fft": float(analysis["samples_per_beat_from_fft"]),
        "note": "Synthetic validation only" if source_label == "synthetic" else "Camera measurement; not medical advice",
    }
    (output_dir / "part1_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="30-second finger-over-camera video")
    source.add_argument("--synthetic", action="store_true", help="run deterministic synthetic smoke test")
    parser.add_argument("--discard-seconds", type=float, default=0.0, help="discard auto-exposure transient")
    parser.add_argument("--output-dir", type=Path, default=Path("part1/results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic:
        trace, fps = synthetic_ppg()
        label = "synthetic"
    else:
        trace, fps = extract_mean_grayscale(args.video, args.discard_seconds)
        label = str(args.video)

    analysis = analyze_trace(trace, fps)
    metrics = save_results(trace, fps, analysis, args.output_dir, label)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


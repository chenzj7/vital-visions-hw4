"""Core codec and signal-processing functions for alpha-channel BFSK."""

from __future__ import annotations

import binascii
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.signal import lfilter


PREAMBLE = np.asarray([1, 0] * 8, dtype=np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def bits_to_bytes(bits: Iterable[int]) -> bytes:
    array = np.asarray(list(bits), dtype=np.uint8)
    if array.size % 8:
        array = np.pad(array, (0, 8 - array.size % 8))
    return np.packbits(array, bitorder="big").tobytes()


def encode_frame(message: str) -> np.ndarray:
    payload = message.encode("utf-8")
    if len(payload) > 65535:
        raise ValueError("Message is too long for the 16-bit length field.")
    length_bits = bytes_to_bits(len(payload).to_bytes(2, "big"))
    crc = binascii.crc_hqx(payload, 0xFFFF)
    return np.concatenate(
        [PREAMBLE, length_bits, bytes_to_bits(payload), bytes_to_bits(crc.to_bytes(2, "big"))]
    )


def decode_frame(bits: np.ndarray, max_preamble_errors: int = 3) -> Dict[str, object]:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size < PREAMBLE.size + 32:
        raise ValueError("Decoded stream is too short for a frame.")
    distances = np.asarray(
        [np.count_nonzero(bits[i : i + PREAMBLE.size] != PREAMBLE) for i in range(bits.size - PREAMBLE.size + 1)]
    )
    preamble_index = int(np.argmin(distances))
    preamble_errors = int(distances[preamble_index])
    if preamble_errors > max_preamble_errors:
        raise ValueError(f"Preamble not found (best Hamming distance {preamble_errors}).")
    cursor = preamble_index + PREAMBLE.size
    if cursor + 16 > bits.size:
        raise ValueError("Incomplete length field.")
    payload_length = int.from_bytes(bits_to_bytes(bits[cursor : cursor + 16]), "big")
    cursor += 16
    payload_bit_count = payload_length * 8
    end_payload = cursor + payload_bit_count
    end_crc = end_payload + 16
    if end_crc > bits.size:
        raise ValueError(
            f"Incomplete frame: length field requests {payload_length} bytes, "
            f"but only {(bits.size - cursor) // 8} bytes remain."
        )
    payload = bits_to_bytes(bits[cursor:end_payload])[:payload_length]
    received_crc = int.from_bytes(bits_to_bytes(bits[end_payload:end_crc]), "big")
    computed_crc = binascii.crc_hqx(payload, 0xFFFF)
    return {
        "preamble_index": preamble_index,
        "preamble_errors": preamble_errors,
        "payload_length_bytes": payload_length,
        "payload_bytes": payload,
        "message": payload.decode("utf-8", errors="replace"),
        "received_crc": received_crc,
        "computed_crc": computed_crc,
        "crc_ok": received_crc == computed_crc,
        "frame_end_bit": end_crc,
    }


def modulation_parameters(fps: float, bit_rate: float) -> Tuple[int, float, float, float]:
    if fps <= 0 or bit_rate <= 0:
        raise ValueError("fps and bit_rate must be positive.")
    samples_per_symbol = max(3, int(round(fps / bit_rate)))
    effective_bit_rate = fps / samples_per_symbol
    frequency_zero = effective_bit_rate
    frequency_one = 2.0 * effective_bit_rate
    return samples_per_symbol, effective_bit_rate, frequency_zero, frequency_one


def modulate_bits(
    bits: np.ndarray,
    fps: float,
    bit_rate: float,
    delta_alpha: float = 0.1,
    baseline_alpha: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Create an alpha waveform; 0 uses one cycle/symbol, 1 uses two."""
    if not 0 < delta_alpha <= 1:
        raise ValueError("delta_alpha must be in (0, 1].")
    samples_per_symbol, effective_rate, f0, f1 = modulation_parameters(fps, bit_rate)
    local_time = np.arange(samples_per_symbol, dtype=float) / fps
    symbols = []
    for bit in np.asarray(bits, dtype=np.uint8):
        frequency = f1 if bit else f0
        symbols.append(baseline_alpha + 0.5 * delta_alpha * np.sin(2 * np.pi * frequency * local_time))
    waveform = np.concatenate(symbols) if symbols else np.empty(0, dtype=float)
    return np.clip(waveform, 0.0, 1.0), {
        "samples_per_symbol": float(samples_per_symbol),
        "effective_bit_rate": effective_rate,
        "frequency_zero_hz": f0,
        "frequency_one_hz": f1,
    }


def decode_symbols(signal: np.ndarray, fps: float, bit_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Non-coherent two-frequency detector applied independently per symbol."""
    samples_per_symbol, _, f0, f1 = modulation_parameters(fps, bit_rate)
    symbol_count = len(signal) // samples_per_symbol
    if symbol_count == 0:
        return np.empty(0, dtype=np.uint8), np.empty(0, dtype=float)
    local_time = np.arange(samples_per_symbol, dtype=float) / fps
    window = np.hanning(samples_per_symbol) if samples_per_symbol >= 4 else np.ones(samples_per_symbol)
    references = [
        np.exp(-2j * np.pi * f0 * local_time),
        np.exp(-2j * np.pi * f1 * local_time),
    ]
    decoded = np.zeros(symbol_count, dtype=np.uint8)
    confidence = np.zeros(symbol_count, dtype=float)
    for index in range(symbol_count):
        segment = np.asarray(signal[index * samples_per_symbol : (index + 1) * samples_per_symbol], dtype=float)
        segment = (segment - np.mean(segment)) * window
        energies = np.asarray([abs(np.vdot(reference, segment)) ** 2 for reference in references])
        decoded[index] = int(energies[1] > energies[0])
        confidence[index] = abs(energies[1] - energies[0]) / max(float(np.sum(energies)), 1e-12)
    return decoded, confidence


def decode_best_alignment(
    signal: np.ndarray, fps: float, bit_rate: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], int]:
    """Try each sample offset and return the alignment with the best preamble."""
    samples_per_symbol, _, _, _ = modulation_parameters(fps, bit_rate)
    candidates = []
    for offset in range(samples_per_symbol):
        bits, confidence = decode_symbols(signal[offset:], fps, bit_rate)
        if bits.size < PREAMBLE.size + 32:
            continue
        distances = [
            np.count_nonzero(bits[i : i + PREAMBLE.size] != PREAMBLE)
            for i in range(bits.size - PREAMBLE.size + 1)
        ]
        candidates.append((min(distances), -float(np.mean(confidence)), offset, bits, confidence))
    if not candidates:
        raise ValueError("Signal is too short to search for a complete frame.")
    _, _, offset, bits, confidence = min(candidates, key=lambda item: (item[0], item[1]))
    frame = decode_frame(bits)
    return bits, confidence, frame, int(offset)


def decode_timed_symbols(
    signal: np.ndarray,
    fps: float,
    bit_rate: float,
    start_sample: int,
    symbol_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode symbols using fractional samples/symbol and variable boundaries.

    A 30 fps camera has 7.5 samples per symbol at 4 bit/s and 3.75 at
    8 bit/s. Rounding that value once and reusing the rounded integer causes
    the receiver clock to drift across a frame. This routine rounds each
    cumulative boundary instead, so the long-term symbol clock stays aligned.
    A value of -1 marks a symbol that lies outside the recorded clip.
    """
    if symbol_count <= 0:
        return np.empty(0, dtype=np.int8), np.empty(0, dtype=float)
    signal = np.asarray(signal, dtype=float)
    samples_per_symbol = fps / bit_rate
    decoded = np.full(symbol_count, -1, dtype=np.int8)
    confidence = np.zeros(symbol_count, dtype=float)
    for index in range(symbol_count):
        nominal_start = int(round(start_sample + index * samples_per_symbol))
        nominal_end = int(round(start_sample + (index + 1) * samples_per_symbol))
        segment_start = max(0, nominal_start)
        segment_end = min(signal.size, nominal_end)
        if segment_end - segment_start < 3:
            continue
        segment = signal[segment_start:segment_end]
        segment = segment - np.mean(segment)
        local_time = np.arange(segment.size, dtype=float) / fps
        window = np.hanning(segment.size) if segment.size >= 4 else np.ones(segment.size)
        energies = []
        for frequency in (bit_rate, 2.0 * bit_rate):
            reference = np.exp(-2j * np.pi * frequency * local_time)
            energies.append(abs(np.vdot(reference, segment * window)) ** 2)
        decoded[index] = int(energies[1] > energies[0])
        confidence[index] = abs(energies[1] - energies[0]) / max(float(sum(energies)), 1e-12)
    return decoded, confidence


def decode_known_frame(
    signal: np.ndarray,
    fps: float,
    bit_rate: float,
    expected_bits: np.ndarray,
    search_margin_seconds: float = 1.0,
) -> Dict[str, object]:
    """Align a known experimental frame and calculate BER.

    The test message is fixed and known at the receiver. We search only the
    frame start time while retaining the requested transmitter bit rate. This
    is a known-sequence timing alignment, not a search over payload values.
    Missing edge symbols in a clipped recording count as bit errors.
    """
    expected = np.asarray(expected_bits, dtype=np.int8)
    expected_samples = expected.size * fps / bit_rate
    padding_samples = signal.size - expected_samples
    margin = max(1, int(round(search_margin_seconds * fps)))
    start_low = int(np.floor(min(-margin, padding_samples - margin)))
    start_high = int(np.ceil(max(margin, padding_samples + margin)))

    best = None
    for start_sample in range(start_low, start_high + 1):
        decoded, confidence = decode_timed_symbols(
            signal, fps, bit_rate, start_sample, expected.size
        )
        valid = decoded >= 0
        error_count = int(np.count_nonzero(decoded[valid] != expected[valid]))
        missing_count = int(np.count_nonzero(~valid))
        error_count += missing_count
        ber = error_count / max(expected.size, 1)
        mean_confidence = float(np.mean(confidence[valid])) if np.any(valid) else 0.0
        score = (ber, -mean_confidence, abs(start_sample))
        if best is None or score < best[0]:
            best = (
                score,
                start_sample,
                decoded.copy(),
                confidence.copy(),
                error_count,
                missing_count,
                mean_confidence,
            )

    if best is None:
        raise ValueError("No candidate frame alignment could be evaluated.")
    _, start_sample, decoded, confidence, error_count, missing_count, mean_confidence = best
    frame = None
    if missing_count == 0:
        try:
            frame = decode_frame(decoded.astype(np.uint8))
        except ValueError:
            frame = None
    return {
        "decoded_bits": decoded,
        "confidence": confidence,
        "start_sample": int(start_sample),
        "start_seconds": float(start_sample / fps),
        "bit_errors": int(error_count),
        "missing_bits": int(missing_count),
        "total_bits": int(expected.size),
        "ber": float(error_count / max(expected.size, 1)),
        "mean_symbol_confidence": mean_confidence,
        "frame": frame,
    }


def simulate_channel(
    bits: np.ndarray,
    fps: float,
    bit_rate: float,
    delta_alpha: float,
    distance_cm: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Simple, explicit camera-channel model for algorithm verification only."""
    rng = np.random.default_rng(seed)
    alpha, parameters = modulate_bits(bits, fps, bit_rate, delta_alpha)
    attenuation = (20.0 / max(distance_cm, 20.0)) ** 0.80
    # Luminance contrast depends on overlay/background content and distance.
    luminance = 0.8 * attenuation * (alpha - 0.5)
    # Automatic exposure behaves approximately as a low-pass response and
    # suppresses the higher BFSK tone more strongly at long distances.
    memory = min(0.78, 0.42 + 0.0025 * distance_cm)
    blurred = lfilter([1.0 - memory], [1.0, -memory], luminance)
    time = np.arange(blurred.size) / fps
    drift = 0.006 * np.sin(2 * np.pi * 0.23 * time + rng.uniform(0, 2 * np.pi))
    motion = 0.0035 * np.sin(2 * np.pi * 3.1 * time + rng.uniform(0, 2 * np.pi))
    noise_sigma = 0.0045 + 0.000045 * distance_cm
    observed = 0.5 + blurred + drift + motion + rng.normal(0.0, noise_sigma, size=blurred.size)
    parameters.update(
        {
            "distance_cm": float(distance_cm),
            "delta_alpha": float(delta_alpha),
            "attenuation": float(attenuation),
            "noise_sigma": float(noise_sigma),
        }
    )
    return observed, parameters


def bit_error_rate(reference: np.ndarray, estimate: np.ndarray) -> float:
    count = min(len(reference), len(estimate))
    if count == 0:
        return 1.0
    errors = int(np.count_nonzero(np.asarray(reference[:count]) != np.asarray(estimate[:count])))
    errors += abs(len(reference) - len(estimate))
    return errors / max(len(reference), len(estimate))


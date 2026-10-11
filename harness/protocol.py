"""Turn one Ollama generate response into prefill, decode, and a marked trace."""

from __future__ import annotations

from dataclasses import dataclass

from harness.footprint import Sample


@dataclass
class MarkedSample:
    t_s: float
    phys_footprint: int
    resident_size: int
    phase: str


def tokens_per_second(count: int, duration_ns: int) -> float:
    if duration_ns <= 0:
        raise ValueError("duration must be positive")
    return count / (duration_ns / 1e9)


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of an empty series")
    return sum(values) / len(values)


def median(values: list[int]) -> float:
    if not values:
        raise ValueError("median of an empty series")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def mark_phases(
    samples: list[Sample],
    t_end_s: float,
    prompt_eval_ns: int,
    eval_ns: int,
) -> list[MarkedSample]:
    """Label samples from Ollama's own phase durations, ending at t_end_s.

    Decode is the eval window. Prefill is the prompt-eval window just before it.
    Samples before those windows are load or idle, and stay unmarked as other.
    """
    decode_start = t_end_s - eval_ns / 1e9
    prefill_start = decode_start - prompt_eval_ns / 1e9
    marked = []
    for sample in samples:
        if sample.t_s >= decode_start:
            phase = "decode"
        elif sample.t_s >= prefill_start:
            phase = "prefill"
        else:
            phase = "other"
        marked.append(
            MarkedSample(sample.t_s, sample.phys_footprint, sample.resident_size, phase)
        )
    return marked


def decode_ceiling_tokens_per_second(weight_bytes: int, bandwidth_bytes_per_s: float = 400e9) -> float:
    """One token per full read of the weights, at the machine's memory bandwidth."""
    return bandwidth_bytes_per_s / weight_bytes


def relative_gap(value: float, baseline: float) -> float:
    """(value - baseline) / baseline. Negative means slower than the baseline."""
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    return (value - baseline) / baseline


def spread(values: list[float]) -> float:
    """(max - min) / mean. The pass rule widens the 5% floor to this when it is wider."""
    if len(values) < 2:
        raise ValueError("spread needs at least two samples")
    return (max(values) - min(values)) / mean(values)


def parity_band(baseline_samples: list[float], floor: float = 0.05) -> float:
    """The close-to-parity band: 5%, or the baseline's own spread when that is wider."""
    if len(baseline_samples) < 2:
        return floor
    return max(floor, spread(baseline_samples))


def inside_band(value: float, baseline: float, band: float) -> bool:
    return abs(relative_gap(value, baseline)) <= band

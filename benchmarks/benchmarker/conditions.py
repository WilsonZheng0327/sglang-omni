"""Shared recording for benchmark runs that compare speed across revisions."""

from __future__ import annotations

import argparse
import logging
import statistics
from typing import TypedDict

from benchmarks.eval.asr_profiling import (
    BenchmarkFingerprint,
    collect_environment_fingerprint,
    collect_server_identity,
)

logger = logging.getLogger(__name__)

# note (wilsonzheng0327): below this count p99 interpolates the two slowest requests.
TAIL_PERCENTILE_MIN_SAMPLES = 100


class MetricAggregate(TypedDict):
    mean: float | None
    min: float | None
    max: float | None
    n: int


def warn_if_tail_percentile_is_thin(sample_count: int) -> None:
    if sample_count >= TAIL_PERCENTILE_MIN_SAMPLES:
        return
    logger.warning(
        f"latency_p99_s interpolates the two slowest of {sample_count} requests; "
        f"use at least {TAIL_PERCENTILE_MIN_SAMPLES} samples before citing tails"
    )


def collect_run_fingerprint(base_url: str) -> BenchmarkFingerprint:
    return {
        "client": collect_environment_fingerprint(),
        "server": collect_server_identity(base_url),
    }


def fingerprint_fields(
    enabled: bool,
    base_url: str,
) -> dict[str, BenchmarkFingerprint]:
    if not enabled:
        return {}
    return {"environment_fingerprint": collect_run_fingerprint(base_url)}


def add_fingerprint_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fingerprint",
        action="store_true",
        help="Record the client environment and the server /v1/models identity.",
    )


def add_talker_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--talker-temperature",
        type=float,
        default=None,
        help="Talker sampling temperature. Unset keeps the server default.",
    )
    parser.add_argument("--talker-top-p", type=float, default=None)
    parser.add_argument("--talker-top-k", type=int, default=None)
    parser.add_argument("--talker-repetition-penalty", type=float, default=None)


def optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def present_floats(values: list[object]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        number = optional_float(value)
        if number is None:
            continue
        numbers.append(number)
    return numbers


def aggregate_numbers(values: list[float]) -> MetricAggregate:
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def sampling_seed_field(seed: int | None) -> dict[str, int]:
    if seed is None:
        return {}
    return {"seed": seed}

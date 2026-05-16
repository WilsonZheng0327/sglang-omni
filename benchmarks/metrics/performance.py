# SPDX-License-Identifier: Apache-2.0
"""System performance metrics: latency, RTF, throughput, token throughput.

Aggregate throughput is reported under two distinct keys:

- ``tok_per_s_engine_agg``: sum of completion tokens divided by sum of
  server-reported engine time. Only valid for backends that surface real
  engine timing (currently TTS via the ``X-Engine-Time`` header).
- ``tok_per_s_clientwall_agg``: sum of completion tokens divided by sum of
  client-side wall-clock elapsed. Used by MMMU, audio-understanding, and
  video-understanding paths where the server does not report engine time.

A request contributes to one bucket based on its ``timing_source``
field. Mixed-source runs emit both keys.

Streaming-mode requests additionally emit ``ttft_*`` and
``inter_content_chunk_*`` keys computed from per-content-chunk arrival
offsets captured by the send_fn.
"""

from __future__ import annotations

import numpy as np

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics._format import SPEED_LABEL_WIDTH, SPEED_LINE_WIDTH


def _compute_token_metrics(successes: list[RequestResult]) -> dict:
    tokens_per_sec = [o.tok_per_s for o in successes if o.tok_per_s > 0]
    gen_token_counts = [
        o.completion_tokens for o in successes if o.completion_tokens > 0
    ]
    total_tokens = sum(gen_token_counts)
    prompt_token_counts = [o.prompt_tokens for o in successes if o.prompt_tokens > 0]

    # Bucket by authoritative timing field; unset timing_source goes into the
    # legacy bucket for backward compat, but *_engine_agg / *_clientwall_agg
    # are the truthful keys for migrated paths.
    engine_timed = [
        o for o in successes if o.timing_source == "engine_time_s" and o.engine_time_s > 0
    ]
    clientwall_timed = [
        o
        for o in successes
        if o.timing_source == "client_wall_time_s"
        and o.client_wall_time_s is not None
        and o.client_wall_time_s > 0
    ]
    legacy_timed = [
        o for o in successes if o.timing_source == "" and o.engine_time_s > 0
    ]

    token_metrics: dict = {}
    if tokens_per_sec:
        token_metrics["tok_per_s_mean"] = round(float(np.mean(tokens_per_sec)), 1)
        token_metrics["tok_per_s_median"] = round(float(np.median(tokens_per_sec)), 1)

    if engine_timed:
        engine_tokens = sum(o.completion_tokens for o in engine_timed)
        engine_time = sum(o.engine_time_s for o in engine_timed)
        if engine_time > 0 and engine_tokens > 0:
            token_metrics["tok_per_s_engine_agg"] = round(
                engine_tokens / engine_time, 1
            )
    if clientwall_timed:
        clientwall_tokens = sum(o.completion_tokens for o in clientwall_timed)
        clientwall_time = sum(
            o.client_wall_time_s or 0.0 for o in clientwall_timed
        )
        if clientwall_time > 0 and clientwall_tokens > 0:
            token_metrics["tok_per_s_clientwall_agg"] = round(
                clientwall_tokens / clientwall_time, 1
            )
    # tok_per_s_agg only emitted when the data genuinely predates the
    # timing_source migration; migrated paths use the explicit aggregate keys.
    if legacy_timed and not engine_timed and not clientwall_timed:
        legacy_tokens = sum(o.completion_tokens for o in legacy_timed)
        legacy_time = sum(o.engine_time_s for o in legacy_timed)
        if legacy_time > 0 and legacy_tokens > 0:
            token_metrics["tok_per_s_agg"] = round(legacy_tokens / legacy_time, 1)

    if gen_token_counts:
        token_metrics["gen_tokens_mean"] = round(float(np.mean(gen_token_counts)), 0)
        token_metrics["gen_tokens_total"] = total_tokens
    if prompt_token_counts:
        token_metrics["prompt_tokens_mean"] = round(
            float(np.mean(prompt_token_counts)), 0
        )
        token_metrics["prompt_tokens_total"] = sum(prompt_token_counts)
    return token_metrics


def _compute_streaming_metrics(successes: list[RequestResult]) -> dict:
    """TTFT + inter-content-chunk stats from streaming runs.

    Inter-chunk intervals pooled across all requests. Single-chunk responses
    contribute to TTFT but not to interval stats. Returns {} when no
    streaming data present.
    """
    streaming_results = [o for o in successes if o.ttft_s is not None]
    if not streaming_results:
        return {}

    metrics: dict = {}
    ttft_values_s = [o.ttft_s for o in streaming_results]
    metrics["ttft_mean_s"] = round(float(np.mean(ttft_values_s)), 4)
    metrics["ttft_p50_s"] = round(float(np.percentile(ttft_values_s, 50)), 4)
    metrics["ttft_p95_s"] = round(float(np.percentile(ttft_values_s, 95)), 4)

    # Pool intervals across requests. The send→first-chunk interval equals
    # TTFT and is excluded here (covered separately above).
    pooled_intervals_ms: list[float] = []
    requests_with_intervals = 0
    for o in streaming_results:
        offsets = o.content_chunk_offsets_ms
        if len(offsets) < 2:
            continue
        requests_with_intervals += 1
        for prev, curr in zip(offsets, offsets[1:]):
            pooled_intervals_ms.append(curr - prev)

    metrics["inter_content_chunk_inclusion"] = {
        "requests_total": len(streaming_results),
        "requests_with_intervals": requests_with_intervals,
    }

    if pooled_intervals_ms:
        intervals_s = np.array(pooled_intervals_ms) / 1000.0
        metrics["inter_content_chunk_mean_s"] = round(float(np.mean(intervals_s)), 4)
        metrics["inter_content_chunk_p50_s"] = round(
            float(np.percentile(intervals_s, 50)), 4
        )
        metrics["inter_content_chunk_p99_s"] = round(
            float(np.percentile(intervals_s, 99)), 4
        )
    else:
        # Single-chunk-only streaming run. Emit None (vs key-absent for
        # non-streaming) so consumers can distinguish the two cases.
        metrics["inter_content_chunk_mean_s"] = None
        metrics["inter_content_chunk_p50_s"] = None
        metrics["inter_content_chunk_p99_s"] = None

    return metrics


def compute_speed_metrics(
    outputs: list[RequestResult], wall_clock_s: float | None = None
) -> dict:
    """Compute system performance summary from a list of request results."""
    successes = [o for o in outputs if o.is_success]
    if not successes:
        return {"completed_requests": 0, "failed_requests": len(outputs)}

    latencies = [o.latency_s for o in successes]
    rtfs = [o.rtf for o in successes if 0 < o.rtf < float("inf")]
    audio_durations = [o.audio_duration_s for o in successes if o.audio_duration_s > 0]

    if wall_clock_s is not None and wall_clock_s > 0:
        throughput = round(len(successes) / wall_clock_s, 3)
    else:
        total_latency = sum(latencies)
        throughput = (
            round(len(successes) / total_latency, 3) if total_latency > 0 else 0
        )

    metrics_summary: dict = {
        "completed_requests": len(successes),
        "failed_requests": len(outputs) - len(successes),
        "latency_mean_s": round(float(np.mean(latencies)), 3),
        "latency_median_s": round(float(np.median(latencies)), 3),
        "latency_p95_s": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_s": round(float(np.percentile(latencies, 99)), 3),
        "audio_duration_mean_s": (
            round(float(np.mean(audio_durations)), 3) if audio_durations else 0
        ),
        "rtf_mean": round(float(np.mean(rtfs)), 4) if rtfs else None,
        "rtf_median": round(float(np.median(rtfs)), 4) if rtfs else None,
        "throughput_qps": throughput,
        **_compute_token_metrics(successes),
        **_compute_streaming_metrics(successes),
    }
    return metrics_summary


def print_speed_summary(
    metrics: dict,
    model_name: str,
    concurrency: int | None = None,
    title: str = "Speed Benchmark Result",
) -> None:
    lw = SPEED_LABEL_WIDTH
    w = SPEED_LINE_WIDTH
    print(f"\n{'=' * w}")
    print(f"{title:^{w}}")
    print(f"{'=' * w}")
    print(f"  {'Model:':<{lw}} {model_name}")
    if concurrency is not None:
        print(f"  {'Concurrency:':<{lw}} {concurrency}")
    print(f"  {'Completed requests:':<{lw}} {metrics['completed_requests']}")
    print(f"  {'Failed requests:':<{lw}} {metrics['failed_requests']}")
    print(f"{'-' * w}")
    print(f"  {'Latency mean (s):':<{lw}} {metrics.get('latency_mean_s', 'N/A')}")
    print(f"  {'Latency median (s):':<{lw}} {metrics.get('latency_median_s', 'N/A')}")
    print(f"  {'Latency p95 (s):':<{lw}} {metrics.get('latency_p95_s', 'N/A')}")
    print(f"  {'Latency p99 (s):':<{lw}} {metrics.get('latency_p99_s', 'N/A')}")
    if metrics.get("rtf_mean") is not None:
        print(f"  {'RTF mean:':<{lw}} {metrics['rtf_mean']}")
        print(f"  {'RTF median:':<{lw}} {metrics['rtf_median']}")
    if metrics.get("audio_duration_mean_s"):
        print(
            f"  {'Audio duration mean (s):':<{lw}} {metrics['audio_duration_mean_s']}"
        )
    if metrics.get("ttft_mean_s") is not None:
        print(f"  {'TTFT mean (s):':<{lw}} {metrics['ttft_mean_s']}")
        print(f"  {'TTFT p50 (s):':<{lw}} {metrics.get('ttft_p50_s', 'N/A')}")
        print(f"  {'TTFT p95 (s):':<{lw}} {metrics.get('ttft_p95_s', 'N/A')}")
    if metrics.get("inter_content_chunk_mean_s") is not None:
        print(
            f"  {'Inter-chunk mean (s):':<{lw}} "
            f"{metrics['inter_content_chunk_mean_s']}"
        )
        print(
            f"  {'Inter-chunk p50 (s):':<{lw}} "
            f"{metrics.get('inter_content_chunk_p50_s', 'N/A')}"
        )
        print(
            f"  {'Inter-chunk p99 (s):':<{lw}} "
            f"{metrics.get('inter_content_chunk_p99_s', 'N/A')}"
        )
    if metrics.get("tok_per_s_mean") is not None:
        print(f"  {'Tok/s (per-req mean):':<{lw}} {metrics['tok_per_s_mean']}")
        print(f"  {'Tok/s (per-req median):':<{lw}} {metrics['tok_per_s_median']}")
    if metrics.get("tok_per_s_engine_agg") is not None:
        print(f"  {'Tok/s (engine agg):':<{lw}} {metrics['tok_per_s_engine_agg']}")
    if metrics.get("tok_per_s_clientwall_agg") is not None:
        print(
            f"  {'Tok/s (clientwall agg):':<{lw}} "
            f"{metrics['tok_per_s_clientwall_agg']}"
        )
    if metrics.get("tok_per_s_agg") is not None:
        print(f"  {'Tok/s (legacy agg):':<{lw}} {metrics['tok_per_s_agg']}")
    if metrics.get("gen_tokens_mean") is not None:
        print(f"  {'Gen tokens (mean):':<{lw}} {metrics['gen_tokens_mean']:.0f}")
        print(f"  {'Gen tokens (total):':<{lw}} {metrics['gen_tokens_total']}")
    if metrics.get("prompt_tokens_mean") is not None:
        print(f"  {'Prompt tokens (mean):':<{lw}} {metrics['prompt_tokens_mean']:.0f}")
        print(f"  {'Prompt tokens (total):':<{lw}} {metrics['prompt_tokens_total']}")
    print(f"  {'Throughput (req/s):':<{lw}} {metrics.get('throughput_qps', 'N/A')}")
    print(f"{'=' * w}")


def build_speed_results(
    outputs: list[RequestResult],
    metrics: dict,
    config: dict,
) -> dict:
    return {
        "summary": metrics,
        "config": config,
        "per_request": [_request_result_to_dict(output) for output in outputs],
    }


def _request_result_to_dict(output: RequestResult) -> dict:
    return {
        "id": output.request_id,
        "text": output.text,
        "is_success": output.is_success,
        "latency_s": round(output.latency_s, 4),
        "audio_duration_s": round(output.audio_duration_s, 4),
        "rtf": round(output.rtf, 4) if output.rtf < float("inf") else None,
        "prompt_tokens": output.prompt_tokens or None,
        "completion_tokens": output.completion_tokens or None,
        "timing_source": output.timing_source or None,
        "engine_time_s": round(output.engine_time_s, 4) if output.engine_time_s else None,
        "client_wall_time_s": (
            round(output.client_wall_time_s, 4)
            if output.client_wall_time_s is not None
            else None
        ),
        "tok_per_s": round(output.tok_per_s, 1) if output.tok_per_s > 0 else None,
        # ttft_s=None signals "not streamed" (vs 0 = streamed but no content).
        "ttft_s": round(output.ttft_s, 4) if output.ttft_s is not None else None,
        "content_chunk_offsets_ms": list(output.content_chunk_offsets_ms),
        "content_chunk_count": output.content_chunk_count,
        "wav_path": output.wav_path or None,
        "error": output.error or None,
    }

# SPDX-License-Identifier: Apache-2.0
"""Shared data structures for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestResult:
    request_id: str = ""
    text: str = ""
    is_success: bool = False
    latency_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Server-reported engine time (TTS reads this from X-Engine-Time / usage_data);
    # MMMU/audio/video send_fns leave it 0.0 and populate client_wall_time_s instead.
    engine_time_s: float = 0.0

    # perf_counter elapsed around session.post(); None means "not measured" so
    # JSON consumers can distinguish from a 0-second request.
    client_wall_time_s: float | None = None

    # Which of the two fields above tok_per_s aggregation should use:
    # "engine_time_s" | "client_wall_time_s" | "" (legacy).
    timing_source: str = ""

    tok_per_s: float = 0.0
    wav_path: str = ""
    error: str = ""

    # Streaming-mode fields. ttft_s = wall-clock from send to first non-empty
    # delta.content; None when not streaming or no content frame arrived.
    ttft_s: float | None = None

    # Per-chunk arrival offsets stored as ms-from-send (not absolute perf_counter)
    # so the data is portable across hosts. Length == content_chunk_count.
    content_chunk_offsets_ms: list[float] = field(default_factory=list)

    # Non-empty delta.content frames only — role-only frames, the finish chunk,
    # and the [DONE] sentinel are excluded.
    content_chunk_count: int = 0

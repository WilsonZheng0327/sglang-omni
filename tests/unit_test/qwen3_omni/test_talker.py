# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.components.talker_input import build_assistant_part
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import (
    MIN_PARTIAL_START_CHUNKS,
    QwenTalkerScheduler,
)
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def _sched_req(**data_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(**data_kwargs))


def _take_decode_input(sched_req: SimpleNamespace) -> torch.Tensor | None:
    return QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_qwen_talker_decode_input_consumes_feedback_and_text_or_pad() -> None:
    """Preserves FIFO consumption for ordinary text and final pad fallback."""
    text_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    assert torch.equal(
        _take_decode_input(text_req),
        torch.tensor([21.0, 22.0]),
    )
    assert len(text_req.data.pending_feedback_queue) == 0
    assert len(text_req.data.pending_text_queue) == 0

    pad_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=True,
    )
    assert torch.equal(_take_decode_input(pad_req), torch.tensor([8.0, 10.0]))
    assert len(pad_req.data.pending_feedback_queue) == 0
    assert len(pad_req.data.pending_text_queue) == 0


def test_qwen_talker_decode_input_preserves_feedback_until_text_arrives() -> None:
    """Preserves queued feedback when neither text nor final pad is ready."""
    sched_req = _sched_req(
        pending_feedback_queue=deque(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        ),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    assert _take_decode_input(sched_req) is None
    assert len(sched_req.data.pending_feedback_queue) == 2

    sched_req.data.pending_text_queue.append(torch.tensor([20.0, 20.0]))
    assert torch.equal(_take_decode_input(sched_req), torch.tensor([21.0, 22.0]))
    assert len(sched_req.data.pending_feedback_queue) == 1
    assert torch.equal(
        sched_req.data.pending_feedback_queue[0],
        torch.tensor([3.0, 4.0]),
    )


def test_qwen_talker_decode_readiness_requires_feedback_and_text_or_pad() -> None:
    """Preserves decode gating across no-text, text-ready, and pad-ready states."""
    no_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    with_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    with_pad = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=True,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )

    assert not QwenTalkerModelRunner._data_has_next_decode_input(no_text)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_text)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_pad)


def test_qwen_talker_scheduler_waits_for_stream_done_without_replay() -> None:
    """Preserves build gating and avoids replaying prefetched text chunks."""
    scheduler = object.__new__(QwenTalkerScheduler)
    payload = SimpleNamespace(prefetched_chunks=[], prefetched_stream_done=False)

    assert not scheduler._is_request_build_ready(
        payload,
        pending_stream_done=False,
    )
    assert scheduler._is_request_build_ready(
        payload,
        pending_stream_done=True,
    )

    req_data = SimpleNamespace(
        pending_text_queue=deque([torch.tensor([11.0, 12.0])]),
        thinker_chunks_done=True,
    )
    payload = SimpleNamespace(
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([20.0, 20.0]))],
        prefetched_stream_done=True,
    )
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)
    scheduler._initialize_request_stream_state(req_data, payload)
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))


def test_qwen_talker_assistant_part_handles_short_prefix() -> None:
    """Preserves the 9-row assistant layout before a fourth text token exists."""
    assistant_embed = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    def zero_codec_embed(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((token_ids.shape[0], 2), dtype=torch.float32)

    result = build_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=lambda tensor: tensor,
        codec_embed_fn=zero_codec_embed,
        tts_bos_embed=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
        tts_eos_embed=torch.tensor([[12.0, 13.0]], dtype=torch.float32),
        tts_pad_embed=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        speaker_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        codec_bos_id=6,
        tts_pad_token_id=99,
    )

    assert result["input_embeds"].shape == (9, 2)
    assert result["input_ids"].tolist() == [99] * 9
    assert torch.equal(result["input_embeds"][:3], assistant_embed)
    assert torch.equal(
        result["input_embeds"][3:7],
        torch.tensor(
            [[7.0, 8.0], [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(result["input_embeds"][7], torch.tensor([10.0, 11.0]))
    assert torch.equal(result["input_embeds"][8], torch.zeros(2, dtype=torch.float32))
    assert torch.equal(
        result["future_text_rows"],
        torch.tensor([[12.0, 13.0]], dtype=torch.float32),
    )


def test_qwen_talker_prefill_ignores_late_text_after_thinker_done() -> None:
    """Preserves completed thinker streams against late text chunk appends."""
    builder = object.__new__(TalkerPrefillBuilder)
    req_data = SimpleNamespace(
        thinker_chunks_done=True,
        pending_text_queue=deque(),
    )
    chunk = SimpleNamespace(
        data=torch.tensor([1.0], dtype=torch.float32),
        metadata={},
    )

    builder.append_text_chunk(req_data, chunk)

    assert list(req_data.pending_text_queue) == []


def test_qwen_code_predictor_keeps_4d_logits_token_shape() -> None:
    """Preserves 4D code-predictor logits as a two-dimensional token tensor."""
    logits = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)

    assert sampled.shape == (1, 2)
    assert sampled.tolist() == [[2, 0]]


def _build_assistant_part_for_n_chunks(n: int) -> dict[str, torch.Tensor]:
    """Build an assistant segment with n thinker chunks under the test layout."""
    hidden_dim = 4
    assistant_embed = torch.arange(n * hidden_dim, dtype=torch.float32).reshape(
        n, hidden_dim
    )

    def codec_embed_fn(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((token_ids.shape[0], hidden_dim), dtype=torch.float32)

    return build_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=lambda tensor: tensor,
        codec_embed_fn=codec_embed_fn,
        tts_bos_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        tts_eos_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        tts_pad_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        speaker_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        codec_bos_id=6,
        tts_pad_token_id=99,
    )


def test_partial_prompt_prefill_layout_invariants() -> None:
    """Locks the assistant-segment row contract used by partial-start.

    Source of truth for ``MIN_PARTIAL_START_CHUNKS`` and the documented
    decode-ready operating point: below 3 chunks ``build_assistant_part``
    fails to assemble the layout (``text_hidden`` is < 9 rows while
    ``codec_hidden`` is fixed at 9 rows, so the subsequent tensor add raises);
    at 3 or 4 chunks the layout is stable but ``future_text_rows`` collapses
    to zero after the trailing EOS row is stripped on the partial path; from
    5 chunks onward at least one consumable future text row remains.
    """
    for n in (1, 2):
        try:
            _build_assistant_part_for_n_chunks(n)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "layout invariant: build_assistant_part must fail "
                f"below MIN_PARTIAL_START_CHUNKS (n={n})"
            )

    for n in (3, 4, 5, 10):
        assert (
            _build_assistant_part_for_n_chunks(n)["input_embeds"].shape[0] == 9
        ), f"layout invariant: at n={n} the assistant tail must be 9 rows"

    # future_text_rows count before include_assistant_eos stripping:
    #   n <= 4 -> 1 row (just the EOS row);
    #   n  > 4 -> (n - 4) projected rows + 1 EOS row.
    assert _build_assistant_part_for_n_chunks(3)["future_text_rows"].shape[0] == 1
    assert _build_assistant_part_for_n_chunks(4)["future_text_rows"].shape[0] == 1
    assert _build_assistant_part_for_n_chunks(5)["future_text_rows"].shape[0] == 2
    assert _build_assistant_part_for_n_chunks(6)["future_text_rows"].shape[0] == 3

    def stripped(n: int) -> int:
        rows = _build_assistant_part_for_n_chunks(n)["future_text_rows"]
        return max(rows.shape[0] - 1, 0)

    # With include_assistant_eos=False on the partial path:
    #   n in {3, 4} -> 0 future rows (decode stalls immediately after prefill);
    #   n == 5 -> 1 future row (documented decode-ready operating point);
    #   n >= 6 -> (n - 4) future rows.
    assert stripped(3) == 0
    assert stripped(4) == 0
    assert stripped(5) == 1
    assert stripped(6) == 2


def _fresh_partial_scheduler(
    partial_start_min_chunks: int | None,
) -> QwenTalkerScheduler:
    """Build a bare scheduler instance with only the partial-start state needed."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._partial_start_min_chunks = partial_start_min_chunks
    return scheduler


def _make_payload(
    *,
    request_id: str = "r0",
    prefetched_chunks: list[Any] | None = None,
    prefetched_stream_done: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prefetched_chunks=list(prefetched_chunks or []),
        prefetched_stream_done=prefetched_stream_done,
    )


def test_partial_disabled_preserves_legacy_path() -> None:
    """Knob None preserves legacy behavior under both stream-done states."""
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=None)
    payload = _make_payload(prefetched_chunks=[object()] * 50)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)


def test_partial_enabled_below_threshold_stays_deferred() -> None:
    """Below the effective threshold the payload is not yet build-ready."""
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=10)
    payload = _make_payload(prefetched_chunks=[object()] * 4)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_enabled_at_threshold_returns_true_with_done_false() -> None:
    """At or above the effective threshold the payload is build-ready early."""
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=5)
    payload = _make_payload(prefetched_chunks=[object()] * 5)

    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_floors_to_min_partial_start_chunks() -> None:
    """User threshold below the layout floor is clamped up to MIN_PARTIAL_START_CHUNKS."""
    assert MIN_PARTIAL_START_CHUNKS >= 1
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=1)

    below_floor = _make_payload(
        prefetched_chunks=[object()] * (MIN_PARTIAL_START_CHUNKS - 1)
    )
    at_floor = _make_payload(prefetched_chunks=[object()] * MIN_PARTIAL_START_CHUNKS)

    assert not scheduler._is_request_build_ready(below_floor, pending_stream_done=False)
    assert scheduler._is_request_build_ready(at_floor, pending_stream_done=False)


def test_partial_count_excludes_im_end_chunks() -> None:
    """im_end chunks are stripped by build_prefill_input, so they do not
    contribute to the usable-prefix count that gates partial-start.
    """
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=3)
    scheduler._im_end_token_id = 13

    def _chunk(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=torch.tensor([0.0]),
            metadata={"token_id": token_id},
        )

    # Three raw chunks but one is im_end -> usable count is 2 -> not ready.
    near_floor = _make_payload(prefetched_chunks=[_chunk(100), _chunk(101), _chunk(13)])
    assert not scheduler._is_request_build_ready(near_floor, pending_stream_done=False)

    # Four raw chunks with one im_end -> usable count is 3 -> ready.
    enough = _make_payload(
        prefetched_chunks=[_chunk(100), _chunk(101), _chunk(102), _chunk(13)]
    )
    assert scheduler._is_request_build_ready(enough, pending_stream_done=False)


def test_partial_non_positive_threshold_disables_partial_start() -> None:
    """Zero or negative thresholds are treated as disabled rather than
    silently promoted to MIN_PARTIAL_START_CHUNKS, so callers cannot
    accidentally enable partial-start with a sentinel-looking value.
    """
    scheduler_zero = _fresh_partial_scheduler(partial_start_min_chunks=0)
    # _fresh_partial_scheduler bypasses __init__, so we exercise the helper
    # the way production constructs the field via __init__:
    OmniScheduler_init = OmniScheduler.__init__
    try:
        OmniScheduler.__init__ = lambda self, *a, **k: None  # type: ignore[method-assign]
        live = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
        QwenTalkerScheduler.__init__(live, partial_start_min_chunks=0)
        assert live._partial_start_min_chunks is None
        live_negative = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
        QwenTalkerScheduler.__init__(live_negative, partial_start_min_chunks=-3)
        assert live_negative._partial_start_min_chunks is None
    finally:
        OmniScheduler.__init__ = OmniScheduler_init  # type: ignore[method-assign]
    del scheduler_zero


def test_partial_enabled_zero_chunks_stays_deferred() -> None:
    """Enabled knob with empty prefetched_chunks never satisfies the threshold."""
    scheduler = _fresh_partial_scheduler(partial_start_min_chunks=1)
    payload = _make_payload(prefetched_chunks=[])

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_no_op_initialize_request_stream_state_prevents_replay() -> None:
    """Talker's _initialize_request_stream_state must not replay prefetched chunks.

    Comparison test: the base OmniScheduler initializer would walk prefetched
    chunks and append them; the talker override is deliberately a no-op.
    """
    scheduler = object.__new__(QwenTalkerScheduler)

    captured_appends: list[Any] = []
    chunks_consumed_at_build = [
        SimpleNamespace(data=torch.tensor([1.0]), metadata={}),
        SimpleNamespace(data=torch.tensor([2.0]), metadata={}),
    ]
    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )
    payload = SimpleNamespace(
        prefetched_chunks=chunks_consumed_at_build,
        prefetched_stream_done=False,
    )

    scheduler._initialize_request_stream_state(req_data, payload)

    # No-op override: no rows appended for chunks consumed at build time.
    assert list(req_data.pending_text_queue) == []

    # Comparison: the base initializer would have appended them, which would be wrong.
    base_req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )

    def _record_append(_self: Any, _req: Any, chunk: Any) -> None:
        captured_appends.append(chunk)

    base_scheduler = object.__new__(QwenTalkerScheduler)
    base_scheduler._append_stream_chunk = _record_append.__get__(
        base_scheduler, QwenTalkerScheduler
    )
    base_scheduler._mark_stream_done = lambda req: None
    # Call the upstream base implementation directly.
    OmniScheduler._initialize_request_stream_state(
        base_scheduler, base_req_data, payload
    )
    assert len(captured_appends) == len(chunks_consumed_at_build)


def test_stream_done_after_partial_build_marks_thinker_done() -> None:
    """_on_stream_done after an early build flips the flag and appends EOS once."""
    builder = object.__new__(TalkerPrefillBuilder)
    eos_embed = torch.tensor([99.0, 98.0], dtype=torch.float32)
    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_eos_embed=eos_embed,
    )

    builder.mark_thinker_done(req_data)
    assert req_data.thinker_chunks_done is True
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], eos_embed.cpu())

    # Second invocation is a no-op (no duplicate EOS row).
    builder.mark_thinker_done(req_data)
    assert len(req_data.pending_text_queue) == 1


def test_chunk_after_partial_build_appends_once() -> None:
    """_on_stream_chunk after an early build appends exactly one row; im_end filtered."""
    builder = object.__new__(TalkerPrefillBuilder)
    builder._im_end_token_id = 13
    builder.project_assistant_chunk = lambda chunk: torch.tensor(
        [7.0, 8.0], dtype=torch.float32
    )

    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )

    ordinary_chunk = SimpleNamespace(
        data=torch.tensor([0.0]),
        metadata={"token_id": 100},
    )
    im_end_chunk = SimpleNamespace(
        data=torch.tensor([0.0]),
        metadata={"token_id": 13},
    )

    builder.append_text_chunk(req_data, ordinary_chunk)
    assert len(req_data.pending_text_queue) == 1

    # im_end is filtered: no second row appended.
    builder.append_text_chunk(req_data, im_end_chunk)
    assert len(req_data.pending_text_queue) == 1

    # After thinker_chunks_done flips, later chunks no longer append.
    req_data.thinker_chunks_done = True
    builder.append_text_chunk(req_data, ordinary_chunk)
    assert len(req_data.pending_text_queue) == 1


def _make_request_builder_stub_env(
    *,
    fallback_chunks_from_state: list[Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Inline the talker request builder around stubbed prefill / build helpers.

    Mirrors the gate / fallback / thread-through contract enforced by the real
    ``request_builders.make_talker_scheduler_adapters.request_builder``.
    """
    captured: dict[str, Any] = {}

    def stub_build_prompt_prefill(
        _payload: Any,
        thinker_chunks: list[Any],
        *,
        thinker_done: bool,
    ) -> dict[str, Any]:
        captured["build_prompt_prefill_thinker_done"] = thinker_done
        captured["build_prompt_prefill_chunk_count"] = len(thinker_chunks)
        return {
            "input_embeds": torch.zeros((9, 2), dtype=torch.float32),
            "pending_text_queue": deque(),
            "tts_eos_embed": torch.zeros((2,), dtype=torch.float32),
        }

    def stub_build_sglang_talker_request(
        *, thinker_chunks_done: bool
    ) -> SimpleNamespace:
        captured["talker_request_thinker_chunks_done"] = thinker_chunks_done
        return SimpleNamespace(req=SimpleNamespace(rid="r0"))

    def request_builder(payload: Any) -> Any:
        thinker_chunks = list(payload.prefetched_chunks)
        thinker_done = bool(payload.prefetched_stream_done)

        if not thinker_chunks:
            if thinker_done:
                thinker_chunks = list(fallback_chunks_from_state or [])
                if not thinker_chunks:
                    raise ValueError(
                        "talker request_builder requires thinker output tokens"
                    )
            else:
                raise RuntimeError(
                    "talker partial-start path entered with zero usable thinker "
                    "chunks; check the partial-start readiness policy"
                )

        prompt_prefill = stub_build_prompt_prefill(
            payload, thinker_chunks, thinker_done=thinker_done
        )
        req_data = stub_build_sglang_talker_request(thinker_chunks_done=thinker_done)
        req_data.tts_eos_embed = prompt_prefill["tts_eos_embed"]
        req_data.stage_payload = payload
        return req_data

    return request_builder, captured


def test_request_builder_threads_thinker_done_false_on_partial_path() -> None:
    """Builder passes thinker_done=False through prefill and request construction."""
    request_builder, captured = _make_request_builder_stub_env()
    payload = SimpleNamespace(
        request_id="r0",
        request=SimpleNamespace(params={}),
        prefetched_chunks=[object(), object(), object(), object(), object()],
        prefetched_stream_done=False,
    )

    request_builder(payload)

    assert captured["build_prompt_prefill_thinker_done"] is False
    assert captured["talker_request_thinker_chunks_done"] is False


def test_request_builder_threads_thinker_done_true_on_completed_stream() -> None:
    """Builder passes thinker_done=True through prefill and request construction."""
    request_builder, captured = _make_request_builder_stub_env()
    payload = SimpleNamespace(
        request_id="r0",
        request=SimpleNamespace(params={}),
        prefetched_chunks=[object(), object(), object()],
        prefetched_stream_done=True,
    )

    request_builder(payload)

    assert captured["build_prompt_prefill_thinker_done"] is True
    assert captured["talker_request_thinker_chunks_done"] is True


def test_partial_path_rejects_zero_chunks_without_done() -> None:
    """Partial path with empty prefetched_chunks raises a clear RuntimeError."""
    request_builder, _ = _make_request_builder_stub_env()
    payload = SimpleNamespace(
        request_id="r0",
        request=SimpleNamespace(params={}),
        prefetched_chunks=[],
        prefetched_stream_done=False,
    )

    try:
        request_builder(payload)
    except RuntimeError as exc:
        assert "partial-start path" in str(exc)
    else:
        raise AssertionError(
            "request_builder must raise RuntimeError on zero-chunk partial path"
        )


def test_fallback_chunks_only_on_done_path() -> None:
    """Completed-stream path with empty chunks consults the fallback helper."""
    request_builder, captured = _make_request_builder_stub_env(
        fallback_chunks_from_state=[object(), object(), object()],
    )
    payload = SimpleNamespace(
        request_id="r0",
        request=SimpleNamespace(params={}),
        prefetched_chunks=[],
        prefetched_stream_done=True,
    )

    request_builder(payload)

    assert captured["build_prompt_prefill_thinker_done"] is True
    assert captured["build_prompt_prefill_chunk_count"] == 3


def test_done_path_with_no_fallback_raises_value_error() -> None:
    """Completed-stream path with no fallback chunks still raises ValueError."""
    request_builder, _ = _make_request_builder_stub_env(fallback_chunks_from_state=[])
    payload = SimpleNamespace(
        request_id="r0",
        request=SimpleNamespace(params={}),
        prefetched_chunks=[],
        prefetched_stream_done=True,
    )

    try:
        request_builder(payload)
    except ValueError as exc:
        assert "thinker output tokens" in str(exc)
    else:
        raise AssertionError(
            "request_builder must raise ValueError when no chunks and no fallback"
        )


def _build_state_machine_scheduler(
    *,
    partial_start_min_chunks: int | None,
    request_builder_stub: Any,
) -> QwenTalkerScheduler:
    """Construct a scheduler with just enough state for process_input_requests."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._partial_start_min_chunks = partial_start_min_chunks
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._aborted_request_ids = set()
    scheduler.waiting_queue = []
    scheduler._request_builder = request_builder_stub
    scheduler.max_req_len = 8192
    return scheduler


def test_process_input_requests_partial_build_state_machine() -> None:
    """Drive process_input_requests through the partial-build path end-to-end."""
    appended: list[Any] = []
    marked_done = [False]

    def stub_request_builder(payload: Any) -> Any:
        captured_done = bool(payload.prefetched_stream_done)
        return SimpleNamespace(
            req=SimpleNamespace(
                rid=payload.request_id,
                _omni_data=None,
                origin_input_ids=[],
                sampling_params=SimpleNamespace(max_new_tokens=0),
            ),
            thinker_chunks_done=captured_done,
            pending_text_queue=deque(),
            _captured_thinker_done=captured_done,
        )

    scheduler = _build_state_machine_scheduler(
        partial_start_min_chunks=5,
        request_builder_stub=stub_request_builder,
    )
    # Stream-state handlers used after build:
    scheduler._append_stream_chunk = lambda req_data, chunk: appended.append(chunk)
    scheduler._mark_stream_done = lambda req_data: marked_done.__setitem__(0, True)

    chunks = [SimpleNamespace(data=torch.tensor([float(i)])) for i in range(5)]
    payload = SimpleNamespace(
        request_id="rid-partial-1",
        prefetched_chunks=list(chunks),
        prefetched_stream_done=False,
    )

    # 1) Drive the upstream-base process_input_requests with a partial payload.
    OmniScheduler.process_input_requests(scheduler, [payload])

    assert scheduler.waiting_queue, "request must have been built and enqueued"
    built = scheduler.waiting_queue[0]._omni_data
    assert built._captured_thinker_done is False
    assert "rid-partial-1" not in scheduler._deferred_request_payloads
    assert "rid-partial-1" not in scheduler._pending_stream_done
    assert appended == []
    assert marked_done == [False]

    # 2) A later stream chunk arrives. _find_request_data is provided by upstream;
    #    short-circuit it for the test so that the live request is found.
    scheduler._find_request_data = lambda rid: built if rid == "rid-partial-1" else None
    OmniScheduler._on_stream_chunk(
        scheduler, "rid-partial-1", SimpleNamespace(data=torch.tensor([42.0]))
    )
    assert len(appended) == 1, "exactly one row must be appended after build"

    # 3) The eventual stream_done arrives.
    OmniScheduler._on_stream_done(scheduler, "rid-partial-1")
    assert marked_done == [True]


def test_process_input_requests_keeps_deferred_when_below_threshold() -> None:
    """Below the partial threshold the payload stays in _deferred_request_payloads."""

    def fail_if_called(_payload: Any) -> Any:
        raise AssertionError(
            "request_builder must not be called when threshold is not met"
        )

    scheduler = _build_state_machine_scheduler(
        partial_start_min_chunks=10,
        request_builder_stub=fail_if_called,
    )
    payload = SimpleNamespace(
        request_id="rid-stay",
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([0.0]))] * 2,
        prefetched_stream_done=False,
    )

    OmniScheduler.process_input_requests(scheduler, [payload])

    assert "rid-stay" in scheduler._deferred_request_payloads
    assert scheduler.waiting_queue == []


def test_abort_filters_subsequent_stream_messages_via_recv_requests() -> None:
    """After abort, subsequent stream messages are filtered at recv_requests.

    The public abort surface includes batch-removal bookkeeping that touches
    scheduler state we do not own in this unit test (running_batch /
    waiting_queue inner state). The filter contract being tested here is the
    line in recv_requests that short-circuits on _aborted_request_ids
    membership; we exercise that line directly by marking the request aborted
    and driving the dispatch loop.
    """
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler.waiting_queue = []

    stream_chunk_calls: list[Any] = []
    stream_done_calls: list[str] = []
    scheduler._on_stream_chunk = lambda rid, data: stream_chunk_calls.append(
        (rid, data)
    )
    scheduler._on_stream_done = lambda rid: stream_done_calls.append(rid)

    messages = [
        IncomingMessage(
            request_id="rid-abort",
            type="stream_chunk",
            data=SimpleNamespace(data=torch.tensor([1.0])),
        ),
        IncomingMessage(request_id="rid-abort", type="stream_done"),
    ]
    scheduler._recv_scheduler_messages = lambda: list(messages)

    # Mark as aborted via the same set the public abort() ultimately writes to.
    scheduler._aborted_request_ids.add("rid-abort")

    OmniScheduler.recv_requests(scheduler)

    assert stream_chunk_calls == []
    assert stream_done_calls == []


def test_wiring_propagation_factory_args_to_scheduler() -> None:
    """factory_args partial_start_min_chunks flows to the live scheduler attribute."""
    from sglang_omni.models.qwen3_omni.config import _talker_stage

    talker_stage = _talker_stage(gpu=0, process="talker_ar")
    assert "partial_start_min_chunks" in talker_stage.factory_args
    assert talker_stage.factory_args["partial_start_min_chunks"] is None

    # Verify the scheduler constructor accepts and stores the kwarg without
    # triggering the heavy OmniScheduler bring-up: stub the parent __init__
    # to a no-op for the duration of the propagation check.
    scheduler = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
    original_parent_init = OmniScheduler.__init__
    try:
        OmniScheduler.__init__ = lambda self, *args, **kwargs: None  # type: ignore[method-assign]
        QwenTalkerScheduler.__init__(scheduler, partial_start_min_chunks=7)
    finally:
        OmniScheduler.__init__ = original_parent_init  # type: ignore[method-assign]

    assert scheduler._partial_start_min_chunks == 7


def test_qwen_model_runner_and_code_predictor_tensor_contracts() -> None:
    """Preserves multimodal embed injection and code-predictor token shape."""

    class RecordingEmbed:
        num_embeddings = 10

        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen = input_ids.clone()
            return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)

    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._embed_tokens = RecordingEmbed()
    runner._image_token_id = 5
    runner._video_token_id = 6
    runner._audio_token_id = 7
    req = SimpleNamespace(
        omni_model_inputs={
            "audio_embeds": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "pad_values": {"audio": 999},
        },
        _omni_consumed=None,
        is_chunked=0,
    )
    input_embeds, _, _ = runner._inject_multimodal_embeds(
        SimpleNamespace(input_ids=torch.tensor([1, 999, 2]), extend_seq_lens_cpu=[3]),
        SimpleNamespace(reqs=[req]),
    )

    assert (
        int(runner._embed_tokens.seen.max().item())
        < runner._embed_tokens.num_embeddings
    )
    assert torch.equal(input_embeds[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    logits = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]])
    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)
    assert sampled.shape == (2, 1)
    assert sampled[:, 0].tolist() == [2, 0]

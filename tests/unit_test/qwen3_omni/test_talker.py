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
    *,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
) -> QwenTalkerScheduler:
    """Build a bare scheduler instance with only the partial-start state needed."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._enable_partial_start = enable_partial_start
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
    """enable_partial_start=False preserves legacy stream_done-only gating."""
    scheduler = _fresh_partial_scheduler(enable_partial_start=False)
    payload = _make_payload(prefetched_chunks=[object()] * 50)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)


def test_partial_enabled_below_threshold_stays_deferred() -> None:
    """Below the threshold the payload is not yet build-ready."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=10
    )
    payload = _make_payload(prefetched_chunks=[object()] * 4)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_enabled_at_threshold_returns_true_with_done_false() -> None:
    """At or above the threshold the payload is build-ready early."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=5
    )
    payload = _make_payload(prefetched_chunks=[object()] * 5)

    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_rejects_min_chunks_below_layout_floor() -> None:
    """partial_start_min_chunks below MIN_PARTIAL_START_CHUNKS raises ValueError."""
    OmniScheduler_init = OmniScheduler.__init__
    try:
        OmniScheduler.__init__ = lambda self, *a, **k: None  # type: ignore[method-assign]
        live = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
        import pytest

        with pytest.raises(ValueError):
            QwenTalkerScheduler.__init__(
                live,
                enable_partial_start=True,
                partial_start_min_chunks=MIN_PARTIAL_START_CHUNKS - 1,
            )
    finally:
        OmniScheduler.__init__ = OmniScheduler_init  # type: ignore[method-assign]


def test_partial_count_excludes_im_end_chunks() -> None:
    """im_end chunks are stripped by build_prefill_input, so they do not
    contribute to the usable-prefix count that gates partial-start.
    """
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=3
    )
    scheduler._im_end_token_id = 13

    def _chunk(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=torch.tensor([0.0]),
            metadata={"token_id": token_id},
        )

    near_floor = _make_payload(prefetched_chunks=[_chunk(100), _chunk(101), _chunk(13)])
    assert not scheduler._is_request_build_ready(near_floor, pending_stream_done=False)

    enough = _make_payload(
        prefetched_chunks=[_chunk(100), _chunk(101), _chunk(102), _chunk(13)]
    )
    assert scheduler._is_request_build_ready(enough, pending_stream_done=False)


def test_partial_enabled_zero_chunks_stays_deferred() -> None:
    """Enabled knob with empty prefetched_chunks never satisfies the threshold."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=MIN_PARTIAL_START_CHUNKS
    )
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


def _drive_real_builder(
    *,
    prefetched_chunks: list[Any] | None,
    prefetched_stream_done: bool,
    fallback_chunks_from_state: list[Any] | None = None,
    request_id: str = "r0",
    request_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Drive the REAL `_build_talker_request_data` helper with stub deps.

    Reviewer flagged that hand-writing a copy of the production closure
    produces false-confidence tests. This helper instead invokes the
    production module-level function directly so the assertions below
    cover the real propagation contract.
    """
    from sglang_omni.models.qwen3_omni.request_builders import (
        _build_talker_request_data,
    )

    captured: dict[str, Any] = {}

    class StubPrefillBuilder:
        def build_prompt_prefill(
            self,
            _payload: Any,
            thinker_chunks: list[Any],
            *,
            thinker_done: bool,
        ) -> dict[str, Any]:
            captured["build_prompt_prefill_thinker_done"] = thinker_done
            captured["build_prompt_prefill_chunk_count"] = len(thinker_chunks)
            return {
                "input_embeds": torch.zeros((9, 2), dtype=torch.float32),
                "input_ids": torch.zeros((9,), dtype=torch.long),
                "pending_text_queue": deque([torch.zeros((2,), dtype=torch.float32)]),
                "tts_eos_embed": torch.full((2,), 0.5, dtype=torch.float32),
                "tts_pad_embed": torch.full((2,), 0.25, dtype=torch.float32),
                "prompt_model_inputs": {"audio_embeds": None},
            }

    def fake_build_sglang_talker_request(**kwargs: Any) -> SimpleNamespace:
        captured["talker_request_kwargs"] = kwargs
        return SimpleNamespace(req=SimpleNamespace(rid=request_id))

    def resolve_sampling_config(_params: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_new_tokens": 4096,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "codec_eos_id": 7,
            "suppress_tokens": [],
            "seed": (_params or {}).get("seed"),
        }

    def fallback(_payload: Any) -> list[Any]:
        return list(fallback_chunks_from_state or [])

    # Patch the module-level build_sglang_talker_request used by the helper.
    from sglang_omni.models.qwen3_omni import request_builders as rb_mod

    original = rb_mod.build_sglang_talker_request
    rb_mod.build_sglang_talker_request = fake_build_sglang_talker_request
    try:
        payload = SimpleNamespace(
            request_id=request_id,
            request=SimpleNamespace(params=request_params or {}),
            prefetched_chunks=list(prefetched_chunks or []),
            prefetched_stream_done=prefetched_stream_done,
        )
        req_data = _build_talker_request_data(
            payload,
            prefill_builder=StubPrefillBuilder(),
            tokenizer=SimpleNamespace(),
            codec_vocab_size=4096,
            codec_bos_id=2149,
            audio_token_id=151646,
            image_token_id=151647,
            video_token_id=151648,
            thinker_config=SimpleNamespace(),
            resolve_sampling_config=resolve_sampling_config,
            fallback_chunks_from_state=fallback,
        )
        return req_data, captured
    finally:
        rb_mod.build_sglang_talker_request = original


def test_real_builder_threads_thinker_done_false_on_partial_path() -> None:
    """Real `_build_talker_request_data` passes thinker_done=False through prefill + request."""
    req_data, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    assert captured["build_prompt_prefill_thinker_done"] is False
    assert captured["talker_request_kwargs"]["thinker_chunks_done"] is False


def test_real_builder_threads_thinker_done_true_on_completed_stream() -> None:
    """Real builder passes thinker_done=True through prefill + request."""
    req_data, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 3, prefetched_stream_done=True
    )
    assert captured["build_prompt_prefill_thinker_done"] is True
    assert captured["talker_request_kwargs"]["thinker_chunks_done"] is True


def test_real_builder_propagates_prefill_outputs_into_talker_request() -> None:
    """input_ids, tts_pad_embed, pending_text_queue, talker_model_inputs all flow through."""
    req_data, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    kw = captured["talker_request_kwargs"]
    assert torch.equal(kw["talker_input_ids"], torch.zeros((9,), dtype=torch.long))
    assert torch.equal(kw["tts_pad_embed"], torch.full((2,), 0.25, dtype=torch.float32))
    assert kw["talker_model_inputs"] == {"audio_embeds": None}
    pending = kw["pending_text_queue"]
    assert pending is not None and len(pending) == 1


def test_real_builder_derives_per_request_seed_when_missing() -> None:
    """When request params carry no seed the builder must derive a stable per-request seed."""
    req_data1, captured1 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-A",
    )
    req_data2, captured2 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-A",
    )
    req_data3, captured3 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-B",
    )
    seed_a1 = captured1["talker_request_kwargs"]["sampling_seed"]
    seed_a2 = captured2["talker_request_kwargs"]["sampling_seed"]
    seed_b = captured3["talker_request_kwargs"]["sampling_seed"]
    assert seed_a1 is not None and seed_a1 > 0
    assert seed_a1 == seed_a2, "same request_id must derive the same seed"
    assert seed_a1 != seed_b, "different request_id must derive different seeds"


def test_real_builder_preserves_explicit_seed_from_request_params() -> None:
    """If the request explicitly carries a seed, builder must not override it."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_params={"seed": 42},
    )
    assert captured["talker_request_kwargs"]["sampling_seed"] == 42


def test_real_builder_attaches_tts_eos_and_stage_payload() -> None:
    """req_data.tts_eos_embed must come from prefill output; stage_payload must round-trip."""
    req_data, _ = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    assert torch.equal(
        req_data.tts_eos_embed, torch.full((2,), 0.5, dtype=torch.float32)
    )
    assert req_data.stage_payload.request_id == "r0"


def test_real_builder_rejects_zero_chunks_without_done() -> None:
    """Partial path with empty prefetched_chunks raises RuntimeError naming the path."""
    import pytest

    with pytest.raises(RuntimeError, match="partial-start path"):
        _drive_real_builder(prefetched_chunks=[], prefetched_stream_done=False)


def test_real_builder_uses_fallback_chunks_on_done_path_only() -> None:
    """Completed-stream path with empty chunks consults the fallback helper."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[],
        prefetched_stream_done=True,
        fallback_chunks_from_state=[object(), object(), object()],
    )
    assert captured["build_prompt_prefill_thinker_done"] is True
    assert captured["build_prompt_prefill_chunk_count"] == 3


def test_real_builder_raises_when_done_and_fallback_empty() -> None:
    """Completed-stream path with no fallback chunks raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="thinker output tokens"):
        _drive_real_builder(
            prefetched_chunks=[],
            prefetched_stream_done=True,
            fallback_chunks_from_state=[],
        )


def _build_state_machine_scheduler(
    *,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
    request_builder_stub: Any,
) -> QwenTalkerScheduler:
    """Construct a scheduler with just enough state for process_input_requests."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._enable_partial_start = enable_partial_start
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
        enable_partial_start=True,
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
        enable_partial_start=True,
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
    """factory_args enable_partial_start + partial_start_min_chunks flow to scheduler."""
    from sglang_omni.models.qwen3_omni.config import _talker_stage

    talker_stage = _talker_stage(gpu=0, process="talker_ar")
    assert talker_stage.factory_args["enable_partial_start"] is False
    assert talker_stage.factory_args["partial_start_min_chunks"] == 5

    scheduler = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
    original_parent_init = OmniScheduler.__init__
    try:
        OmniScheduler.__init__ = lambda self, *args, **kwargs: None  # type: ignore[method-assign]
        QwenTalkerScheduler.__init__(
            scheduler, enable_partial_start=True, partial_start_min_chunks=7
        )
    finally:
        OmniScheduler.__init__ = original_parent_init  # type: ignore[method-assign]

    assert scheduler._enable_partial_start is True
    assert scheduler._partial_start_min_chunks == 7


def test_rollback_decode_prep_after_skip_is_idempotent_across_repeated_stalls() -> None:
    """Repeated stalls must leave talker scheduler state identical to pre-prepare_for_decode.

    Reviewer flagged that an incomplete rollback in shared OmniScheduler could
    leak state across the side-effect set that ``prepare_for_decode`` writes
    (out_cache_loc, seq_lens, decode_batch_idx, kv_committed_len,
    kv_allocated_len). Under talker server_args (overlap disabled, no Mamba)
    those are the only writes; assert that rolling back twice leaves counters
    where they started.
    """
    freed: list[Any] = []

    class FakeAllocator:
        def free(self, slot: Any) -> None:
            freed.append(slot)

    class FakeForwardMode:
        @staticmethod
        def is_decode() -> bool:
            return True

    pre_seq_lens = torch.tensor([12, 12])
    pre_seq_lens_cpu = torch.tensor([12, 12])
    pre_orig_seq_lens = torch.tensor([10, 11])
    pre_seq_lens_sum = 24

    reqs = [
        SimpleNamespace(decode_batch_idx=5, kv_committed_len=12, kv_allocated_len=13),
        SimpleNamespace(decode_batch_idx=7, kv_committed_len=12, kv_allocated_len=13),
    ]
    batch = SimpleNamespace(
        forward_mode=FakeForwardMode(),
        out_cache_loc=object(),
        output_ids=None,
        input_ids=torch.tensor([99, 100]),
        reqs=reqs,
        seq_lens=pre_seq_lens.clone(),
        seq_lens_cpu=pre_seq_lens_cpu.clone(),
        orig_seq_lens=pre_orig_seq_lens.clone(),
        seq_lens_sum=pre_seq_lens_sum,
    )

    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = FakeAllocator()

    # Simulate one prepare_for_decode round: counters already incremented +
    # an out_cache_loc allocation handed in. One stall -> one rollback.
    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is None
    assert batch.output_ids is batch.input_ids
    for req in reqs:
        assert req.decode_batch_idx == [5, 7][reqs.index(req)] - 1
        assert req.kv_committed_len == 11
        assert req.kv_allocated_len == 12
    assert torch.equal(batch.seq_lens, pre_seq_lens - 1)
    assert torch.equal(batch.seq_lens_cpu, pre_seq_lens_cpu - 1)
    assert torch.equal(batch.orig_seq_lens, pre_orig_seq_lens - 1)
    assert batch.seq_lens_sum == pre_seq_lens_sum - len(reqs)
    assert len(freed) == 1

    # A second stall on the next iteration must roll back again without
    # accumulating state — the contract is "leaves no residue per stall".
    # Re-simulate prepare_for_decode having run: counters re-incremented +
    # a fresh out_cache_loc was allocated.
    batch.out_cache_loc = object()
    for req in reqs:
        req.decode_batch_idx += 1
        req.kv_committed_len += 1
        req.kv_allocated_len += 1
    batch.seq_lens.add_(1)
    batch.seq_lens_cpu.add_(1)
    batch.orig_seq_lens.add_(1)
    batch.seq_lens_sum += len(reqs)

    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is None
    for req in reqs:
        assert req.decode_batch_idx == [5, 7][reqs.index(req)] - 1
        assert req.kv_committed_len == 11
        assert req.kv_allocated_len == 12
    assert batch.seq_lens_sum == pre_seq_lens_sum - len(reqs)
    assert len(freed) == 2


def test_rollback_decode_prep_after_skip_is_noop_for_prefill_batches() -> None:
    """Rollback must not fire for prefill batches — those have no decode prep to undo."""

    class FakeForwardMode:
        @staticmethod
        def is_decode() -> bool:
            return False

    freed: list[Any] = []
    batch = SimpleNamespace(
        forward_mode=FakeForwardMode(),
        out_cache_loc=object(),
        seq_lens_sum=99,
    )
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=freed.append)

    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is not None
    assert batch.seq_lens_sum == 99
    assert freed == []


def test_prepare_for_decode_side_effect_contract_with_upstream() -> None:
    """Lock in the upstream side-effect set our rollback knows about.

    Fires when upstream ``ScheduleBatch.prepare_for_decode`` adds writes
    that our ``_rollback_decode_prep_after_skip`` does not undo. If this
    test fails after a sglang bump, either extend the rollback or extend
    the documented invariant (overlap disabled / no Mamba / no hisparse).
    """
    import inspect

    from sglang.srt.managers.schedule_batch import ScheduleBatch

    src = inspect.getsource(ScheduleBatch.prepare_for_decode)
    expected_writes = {
        # Rolled back by _rollback_decode_prep_after_skip
        "out_cache_loc",
        "output_ids",
        "input_ids",
        "decode_batch_idx",
        "kv_committed_len",
        "kv_allocated_len",
        "seq_lens",
        "seq_lens_cpu",
        "orig_seq_lens",
        "seq_lens_sum",
        # Documented as safe under talker server_args (not rolled back)
        "forward_mode",
        "input_embeds",
        "attn_cp_metadata",
        "penalizer_orchestrator",
        "hisparse_coordinator",
        # sampling_info: container accessed for penalizer_orchestrator;
        # idempotent under repeated stalls within one step.
        "sampling_info",
        # enable_overlap / is_spec_v2: read-only flags; talker has overlap
        # disabled and is not in spec-decode mode.
        "enable_overlap",
        "is_spec_v2",
        # Mamba trackers: talker is non-Mamba.
        "mamba_track_indices",
        "mamba_track_mask",
        # Method reference, not an attribute write.
        "prepare_encoder_info_decode",
    }
    # Heuristic scan: tokens that appear as ``self.X =`` / ``self.X.<call>(``
    # in prepare_for_decode. Each must either be rolled back or live in the
    # documented invariant set.
    import re

    self_assignments = set(re.findall(r"self\.([A-Za-z_][A-Za-z0-9_]*)", src))
    novel = (
        self_assignments
        - expected_writes
        - {
            # Read-only attributes referenced via self.X in prepare_for_decode
            "model_runner",
            "spec_algorithm",
            "tree_cache",
            "req_to_token_pool",
            "token_to_kv_pool_allocator",
            "server_args",
            "reqs",
            "is_v1",
            "v1_spec_info",
            "v1_spec_info_filtered",
            "padded_static_len",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_logprob_start_lens",
            "encoder_lens",
            "encoder_cached",
            "encoder_out_cache_loc",
            "encoder_lens_cpu",
            "model_config",
            "device",
            "global_num_tokens",
            "global_num_tokens_for_logprob",
            "global_num_tokens_for_logprob_cpu",
            "global_num_tokens_cpu",
            "spec_info",
            "speculative_num_draft_tokens",
            "is_extend_in_batch",
            "global_dp_buffer_len",
            "global_num_tokens_dispatch",
            "global_num_tokens_dispatch_cpu",
            "can_run_dp_cuda_graph",
            "dp_padding_mode",
            "is_prefill_only",
            "global_forward_mode",
            "is_prefill_only_real",
            "batch_size_real",
            "batch_size_method",
            "batch_size_pd",
            "speculative_algorithm",
            "speculative_num_steps",
            "is_speculative_extend",
            "tp_size",
            "real_bs",
            "chunked_req",
            "padded_extend_len",
            "padded_static_len_min",
            "padded_static_len_max",
            "is_speculative",
            "draft_model_runner",
            "draft_model_config",
            "padded_extend_lens",
            "moe_runner_input",
            "real_decode_bs",
            "real_extend_bs",
            "split_index",
            "lora_paths",
            "kv_cache",
            "token_ids_logprobs",
            "batch_is_full",
            "is_extend",
            "extend_num_tokens",
            "padded_extend_num_tokens",
            "padded_decode_bs",
            "padded_padded_extend_num_tokens",
            "padded_real_bs",
            "use_v1_spec",
            "spec_chunk_size_cap",
            "padded_spec_chunk_size",
            "next_batch_sampling_info",
        }
    )
    assert novel == set(), (
        "ScheduleBatch.prepare_for_decode now writes attribute(s) "
        f"{sorted(novel)} that QwenTalkerScheduler._rollback_decode_prep_after_skip "
        "does not undo. Either extend the rollback or expand the docstring "
        "invariant set in talker_scheduler.py."
    )


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

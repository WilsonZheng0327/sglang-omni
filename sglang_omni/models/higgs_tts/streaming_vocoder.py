# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Higgs TTS."""

from __future__ import annotations

import logging
import queue as _queue_mod
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.utils import reverse_delay_pattern
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)


@dataclass
class _HiggsStreamState:
    delayed_rows: list[torch.Tensor] = field(default_factory=list)
    emitted_raw_frames: int = 0
    next_decode_rows: int = 0
    has_emitted: bool = False


class HiggsStreamingVocoderScheduler:
    """Decode Higgs codec rows incrementally, while preserving non-streaming."""

    def __init__(
        self,
        codec: HiggsAudioCodec,
        *,
        stream_stride: int = 75,
        stream_followup_stride: int = 75,
        stream_overlap_tokens: int = 8,
        stream_holdback_tokens: int = 4,
    ) -> None:
        if stream_stride <= 0 or stream_followup_stride <= 0:
            raise ValueError("stream_stride and stream_followup_stride must be > 0")
        if stream_overlap_tokens < 0:
            raise ValueError("stream_overlap_tokens must be >= 0")
        if stream_holdback_tokens < 0:
            raise ValueError("stream_holdback_tokens must be >= 0")

        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self._codec = codec
        self._stream_stride = int(stream_stride)
        self._stream_followup_stride = int(stream_followup_stride)
        self._stream_overlap_tokens = int(stream_overlap_tokens)
        self._stream_holdback_tokens = int(stream_holdback_tokens)
        self._sample_rate = HiggsAudioCodec.SAMPLE_RATE
        self._running = False

        self._payloads: dict[str, StagePayload] = {}
        self._stream_states: dict[str, _HiggsStreamState] = {}
        self._pending_done: set[str] = set()

    def start(self) -> None:
        self._running = True
        while self._running:
            try:
                msg = self.inbox.get(timeout=0.1)
            except _queue_mod.Empty:
                continue
            try:
                if msg.type == "new_request":
                    self._on_new_request(msg.request_id, msg.data)
                elif msg.type == "stream_chunk":
                    self._on_chunk(msg.request_id, msg.data)
                elif msg.type == "stream_done":
                    self._on_done(msg.request_id)
                else:
                    raise ValueError(f"Unsupported vocoder message type: {msg.type}")
            except Exception as exc:
                logger.exception("Higgs vocoder failed for %s", msg.request_id)
                self.outbox.put(
                    OutgoingMessage(request_id=msg.request_id, type="error", data=exc)
                )
                self.abort(msg.request_id)

    def stop(self) -> None:
        self._running = False

    def abort(self, request_id: str) -> None:
        self._payloads.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._pending_done.discard(request_id)

    def _on_new_request(self, request_id: str, payload: StagePayload) -> None:
        streaming = bool(payload.request.params.get("stream"))
        if not streaming:
            result = self._vocode_payload(payload)
            self.outbox.put(
                OutgoingMessage(request_id=request_id, type="result", data=result)
            )
            return

        self._payloads[request_id] = payload
        self._stream_states.setdefault(request_id, _HiggsStreamState())
        if request_id in self._pending_done:
            self._pending_done.discard(request_id)
            self._finalize_streaming_request(request_id)

    def _on_chunk(self, request_id: str, chunk: Any) -> None:
        # Engine only emits chunks for streaming requests; accept unconditionally.
        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        row = getattr(chunk, "data", chunk)
        if not isinstance(row, torch.Tensor):
            row = torch.tensor(row, dtype=torch.long)
        elif row.dtype != torch.long:
            row = row.to(dtype=torch.long)
        if row.ndim != 1:
            raise ValueError(
                f"Higgs stream chunk must be 1-D [N], got {tuple(row.shape)}"
            )
        state.delayed_rows.append(row)

        output = self._decode_delta(state, is_final=False)
        if output is not None:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

    def _on_done(self, request_id: str) -> None:
        if request_id not in self._payloads:
            self._pending_done.add(request_id)
            return
        self._finalize_streaming_request(request_id)

    def _finalize_streaming_request(self, request_id: str) -> None:
        payload = self._payloads[request_id]
        state = self._stream_states.setdefault(request_id, _HiggsStreamState())
        output = self._decode_delta(state, is_final=True)
        if output is None and not state.has_emitted:
            output = self._audio_payload_from_stage_payload(payload)
        if output is not None:
            self.outbox.put(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data=output,
                    metadata={"modality": "audio"},
                )
            )

        final_data = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        usage = self._build_usage(HiggsTtsState.from_dict(payload.data))
        if usage is not None:
            final_data["usage"] = usage
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data=final_data,
                ),
            )
        )
        self.abort(request_id)

    def _decode_delta(
        self, state: _HiggsStreamState, *, is_final: bool
    ) -> dict[str, Any] | None:
        delayed_count = len(state.delayed_rows)
        if delayed_count == 0:
            return None
        num_codebooks = int(state.delayed_rows[0].shape[0])
        if delayed_count < num_codebooks:
            return None
        raw_total = delayed_count - num_codebooks + 1

        next_decode_rows = state.next_decode_rows or max(
            num_codebooks, self._stream_stride
        )
        if not is_final and delayed_count < next_decode_rows:
            state.next_decode_rows = next_decode_rows
            return None

        emit_until_raw = raw_total
        if not is_final and self._stream_holdback_tokens:
            emit_until_raw = max(0, raw_total - self._stream_holdback_tokens)
        if emit_until_raw <= state.emitted_raw_frames:
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        window_start_raw = max(
            0, state.emitted_raw_frames - self._stream_overlap_tokens
        )
        rows_end = emit_until_raw + num_codebooks - 1
        rows = state.delayed_rows[window_start_raw:rows_end]
        audio = self._decode_delayed_rows(rows)

        decoded_raw_frames = emit_until_raw - window_start_raw
        samples_per_frame = max(int(audio.shape[-1]) // max(decoded_raw_frames, 1), 1)
        trim_frames = state.emitted_raw_frames - window_start_raw
        trim_samples = min(int(trim_frames * samples_per_frame), int(audio.shape[-1]))
        delta = audio[trim_samples:].contiguous()
        if delta.numel() == 0:
            state.next_decode_rows = delayed_count + self._stream_followup_stride
            return None

        state.emitted_raw_frames = emit_until_raw
        state.next_decode_rows = delayed_count + self._stream_followup_stride
        state.has_emitted = True
        return self._build_audio_payload(delta)

    def _audio_payload_from_stage_payload(
        self, payload: StagePayload
    ) -> dict[str, Any] | None:
        state = HiggsTtsState.from_dict(payload.data)
        audio = self._decode_state_to_audio(state)
        if audio is None:
            return None
        return self._build_audio_payload(audio)

    def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        audio = self._decode_state_to_audio(state)
        if audio is None:
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = self._sample_rate
            payload.data["modality"] = "audio"
            return payload
        payload.data["audio_data"] = audio.cpu().numpy().tolist()
        payload.data["sample_rate"] = self._sample_rate
        payload.data["modality"] = "audio"
        usage = self._build_usage(state)
        if usage is not None:
            payload.data["usage"] = usage
        return payload

    def _decode_state_to_audio(self, state: HiggsTtsState) -> torch.Tensor | None:
        delayed_rows = state.output_codes_delayed
        if not delayed_rows:
            return None
        rows = [torch.tensor(row, dtype=torch.long) for row in delayed_rows]
        if len(rows) < int(state.num_codebooks):
            return None
        return self._decode_delayed_rows(rows, codebook_size=int(state.codebook_size))

    def _decode_delayed_rows(
        self,
        rows: list[torch.Tensor],
        *,
        codebook_size: int = 1026,
    ) -> torch.Tensor:
        # Returns float32 audio on the codec's native device. Callers do the
        # single host transfer at serialization time.
        delayed_LN = torch.stack(rows, dim=0).to(torch.long)
        codes_TN = reverse_delay_pattern(delayed_LN)
        codec_vocab = int(codebook_size) - 2
        codes_TN = torch.where(
            codes_TN >= codec_vocab, torch.zeros_like(codes_TN), codes_TN
        )
        return self._codec.decode(codes_TN).detach().to(torch.float32)

    def _build_audio_payload(self, audio: torch.Tensor) -> dict[str, Any]:
        return {
            "audio_data": audio.cpu().numpy().tolist(),
            "sample_rate": self._sample_rate,
            "modality": "audio",
        }

    @staticmethod
    def _build_usage(state: HiggsTtsState) -> dict[str, Any] | None:
        if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
            return None
        usage: dict[str, Any] = {
            "prompt_tokens": state.prompt_tokens,
            "completion_tokens": state.completion_tokens,
            "total_tokens": state.prompt_tokens + state.completion_tokens,
        }
        if state.engine_time_s:
            usage["engine_time_s"] = round(state.engine_time_s, 6)
        return usage


__all__ = ["HiggsStreamingVocoderScheduler"]

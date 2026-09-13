# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_K,
    TEXT_CARD,
    TEXT_PAD_ID,
)
from sglang_omni.models.personaplex.config import CODE2WAV_STAGE
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.sampling import AudioSampling
from sglang_omni.models.personaplex.timeline import (
    Timeline,
    build_prompt_frames,
    build_timeline,
)
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

SEED_NAMESPACE = "personaplex"


@dataclass(frozen=True)
class RequestSampling:
    text_temperature: float
    text_top_k: int
    audio: AudioSampling
    seed: int | None

    @property
    def text_seed(self) -> int | None:
        return (
            None
            if self.seed is None
            else derive_sampling_seed(SEED_NAMESPACE, self.seed, "text")
        )

    @property
    def audio_seed(self) -> int | None:
        return (
            None
            if self.seed is None
            else derive_sampling_seed(SEED_NAMESPACE, self.seed, "audio")
        )


def _float_param(params: dict, key: str, default: float) -> float:
    value = params.get(key)
    return default if value is None else float(value)


def _int_param(params: dict, key: str, default: int) -> int:
    value = params.get(key)
    return default if value is None else int(value)


def resolve_sampling(params: dict) -> RequestSampling:
    """``temperature``/``top_k`` steer the text, ``audio_temperature`` /
    ``audio_top_k`` the codes; ``seed`` makes both draws reproducible."""
    stage = (params.get("stage_sampling") or {}).get("lm") or {}
    seed = params.get("seed")
    if isinstance(seed, bool):
        raise ValueError("PersonaPlex seed must be an integer")
    return RequestSampling(
        text_temperature=_float_param(
            stage,
            "temperature",
            _float_param(params, "temperature", DEFAULT_TEXT_TEMPERATURE),
        ),
        text_top_k=_int_param(
            stage, "top_k", _int_param(params, "top_k", DEFAULT_TEXT_TOP_K)
        ),
        audio=AudioSampling(
            temperature=_float_param(
                params, "audio_temperature", DEFAULT_AUDIO_TEMPERATURE
            ),
            top_k=_int_param(params, "audio_top_k", DEFAULT_AUDIO_TOP_K),
        ),
        seed=None if seed is None else int(seed),
    )


def timeline_from_state(state: PersonaPlexState) -> Timeline:
    if state.user_codes is None:
        raise ValueError("PersonaPlex LM request has no encoded caller audio")
    voice_codes = state.voice_codes
    prompt = build_prompt_frames(
        voice_frames=int(state.voice_frames),
        text_prompt_ids=[int(i) for i in state.text_prompt_ids],
        voice_codes=None if voice_codes is None else voice_codes.to(torch.long),
    )
    return build_timeline(
        prompt,
        state.user_codes.to(torch.long),
        voice_embeddings=state.voice_embeddings,
        voice_tail_codes=(
            None
            if state.voice_tail_codes is None
            else state.voice_tail_codes.to(torch.long)
        ),
    )


def build_lm_request(payload: StagePayload, *, vocab_size: int) -> SGLangARRequestData:
    """One request per recording: the whole prompt as prefill, then one
    decode step per 80 ms frame of the caller's audio."""
    state = PersonaPlexState.from_dict(payload.data)
    timeline = timeline_from_state(state)
    sampling = resolve_sampling(payload.request.params)
    if timeline.num_frames < 1:
        raise ValueError("PersonaPlex needs at least one 80 ms frame of caller audio")

    sampling_params = SamplingParams(
        max_new_tokens=timeline.num_frames,
        temperature=sampling.text_temperature,
        top_k=sampling.text_top_k,
        ignore_eos=True,
    )
    sampling_params.normalize(tokenizer=None)
    if sampling.text_seed is not None:
        sampling_params.sampling_seed = sampling.text_seed

    # Note (wilsonzheng0327): Placeholder ids for SGLang's bookkeeping; the model runner
    # embeds the real rows. The text stream's initial token is outside the vocabulary,
    # so it is masked.
    text_ids = timeline.prefill_tokens[:, 0].clone()
    text_ids[text_ids >= TEXT_CARD] = TEXT_PAD_ID
    input_ids = [int(i) for i in text_ids.tolist()]
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    data = SGLangARRequestData(
        req=req,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        stage_payload=payload,
        max_new_tokens=timeline.num_frames,
        temperature=sampling.text_temperature,
    )
    data.talker_model_inputs = {
        "timeline": timeline,
        "sampling": sampling,
        "frames": [],
        "pending_frames": [],
    }
    return data


def apply_lm_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = PersonaPlexState.from_dict(payload.data)
    frames = data.talker_model_inputs["frames"]
    state.text_ids = [int(i) for i in data.output_ids]
    state.codes = (
        torch.stack(frames).cpu() if frames else torch.zeros(0, 8, dtype=torch.long)
    )
    for name in (
        "waveform",
        "voice_waveform",
        "voice_embeddings",
        "voice_tail_codes",
        "user_codes",
        "voice_codes",
    ):
        setattr(state, name, None)
    state.text_prompt_ids = []
    payload.data = state.to_dict()
    return payload


def lm_stream_output_builder(
    request_id: str, data: SGLangARRequestData, req_output
) -> list[OutgoingMessage]:
    del req_output
    pending = data.talker_model_inputs.get("pending_frames")
    if not pending:
        return []
    frames = torch.stack(pending).cpu()
    pending.clear()
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=frames,
            target=CODE2WAV_STAGE,
            metadata={"modality": "audio_codes"},
        )
    ]


__all__ = [
    "RequestSampling",
    "apply_lm_result",
    "build_lm_request",
    "lm_stream_output_builder",
    "resolve_sampling",
    "timeline_from_state",
]

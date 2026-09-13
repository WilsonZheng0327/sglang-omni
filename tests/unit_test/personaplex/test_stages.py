# SPDX-License-Identifier: Apache-2.0
"""Preprocessing resolves the caller channel, the role prompt and the voice per request."""

import numpy as np
import pytest
import torch

from sglang_omni.models.personaplex import stages
from sglang_omni.models.personaplex.architecture import SAMPLES_PER_FRAME
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.prompts import (
    DEFAULT_TEXT_PROMPT,
    DEFAULT_VOICE,
    VoicePrompt,
    tokenize_text_prompt,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

CALLER_SAMPLES = 2000


class _Tokenizer:
    def encode(self, text):
        return [len(word) for word in text.split()]


@pytest.fixture
def preprocess(monkeypatch, tmp_path):
    loads = []

    def load_voice_prompt(path, *, load_audio):
        loads.append(path)
        return VoicePrompt(
            frames=3,
            embeddings=torch.zeros(2, 4),
            tail_codes=torch.zeros(2, 8, dtype=torch.long),
        )

    caller = np.stack(
        [np.full(CALLER_SAMPLES, 0.5), np.full(CALLER_SAMPLES, -1.0)]
    ).astype(np.float32)
    monkeypatch.setattr(stages, "load_text_tokenizer", lambda _: _Tokenizer())
    monkeypatch.setattr(stages, "load_audio", lambda source, **_: caller)
    monkeypatch.setattr(stages, "resolve_voice_path", lambda _, voice: f"{voice}.pt")
    monkeypatch.setattr(stages, "load_voice_prompt", load_voice_prompt)
    scheduler = stages.create_preprocessing_executor(str(tmp_path))

    def run(**params):
        payload = StagePayload(
            "r",
            request=OmniRequest(inputs={"audio_path": "caller.wav"}, params=params),
            data={},
        )
        return PersonaPlexState.from_dict(scheduler._fn(payload).data)

    run.loads = loads
    return run


def test_caller_is_channel_zero_padded_to_whole_frames(preprocess):
    state = preprocess()
    waveform = state.waveform
    assert waveform.shape[-1] % SAMPLES_PER_FRAME == 0
    assert waveform.shape[-1] == SAMPLES_PER_FRAME * 2
    assert torch.all(waveform[:CALLER_SAMPLES] == 0.5)
    assert torch.all(waveform[CALLER_SAMPLES:] == 0.0)


def test_role_prompt_default_alias_and_empty(preprocess):
    tokenizer = _Tokenizer()
    assert preprocess().text_prompt_ids == tokenize_text_prompt(
        tokenizer, DEFAULT_TEXT_PROMPT
    )
    assert preprocess(text_prompt="Be brief").text_prompt_ids == [8, 2, 5, 8]
    assert preprocess(instructions="Be brief").text_prompt_ids == [8, 2, 5, 8]
    assert preprocess(text_prompt="").text_prompt_ids == []


def test_voice_default_empty_and_cached(preprocess):
    state = preprocess()
    assert preprocess.loads == [f"{DEFAULT_VOICE}.pt"]
    assert state.voice_frames == 3
    assert state.voice_embeddings.shape == (2, 4)

    preprocess()
    assert preprocess.loads == [f"{DEFAULT_VOICE}.pt"]

    state = preprocess(voice="")
    assert state.voice_frames == 0
    assert state.voice_embeddings is None and state.voice_tail_codes is None

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from transformers import WhisperConfig

import sglang_omni.models.arkasr.stages as arkasr_stages
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig


def _tiny_config() -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=256,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=4,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


class _StubModel(torch.nn.Module):
    def __init__(self, config: ArkasrConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = ArkAudioMLPAdapter(config)


def _stub_torch_compile_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # The helper calls sglang's set_torch_compile_config, which mutates global
    # dynamo/inductor config; keep unit tests side-effect free.
    import sglang.srt.compilation.torch_compile_decoration as torch_compile_decoration

    monkeypatch.setattr(
        torch_compile_decoration, "set_torch_compile_config", lambda: None
    )


def test_compile_arkasr_audio_encoder_compiles_forwards_with_dynamic_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_torch_compile_config(monkeypatch)
    model = _StubModel(_tiny_config())
    original_tower_forward = model.audio_encoder.whisper.forward
    original_adapting_forward = model.audio_encoder.adapting.forward
    param_names = set(dict(model.audio_encoder.named_parameters()))

    compile_calls: list[dict] = []
    warmup_batches: list[int] = []

    def _fake_compile(fn, dynamic=None):
        compile_calls.append({"fn": fn, "dynamic": dynamic})

        def _wrapped(*args, **kwargs):
            if args and isinstance(args[0], torch.Tensor):
                warmup_batches.append(int(args[0].shape[0]))
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    arkasr_stages._compile_arkasr_audio_encoder(model, warmup_mel_frames=32)

    assert [call["dynamic"] for call in compile_calls] == [True, True]
    assert compile_calls[0]["fn"] == original_tower_forward
    assert compile_calls[1]["fn"] == original_adapting_forward
    # Bound-method compile must leave the module tree and parameter names
    # intact: load_weights matches checkpoint names against named_parameters.
    assert set(dict(model.audio_encoder.named_parameters())) == param_names
    assert isinstance(model.audio_encoder.whisper, type(model.audio_encoder.whisper))
    # B=1 is today's per-item encode path; B=2 pre-builds the batched graph.
    # Both compiled forwards (tower, then adapting MLP) run at both sizes.
    assert warmup_batches == [1, 1, 2, 2]


def test_compile_warmup_runs_in_inference_mode_when_pre_lm_encoder_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dynamo guards on grad mode, so a warmup in the wrong mode compiles a graph
    the serving caller misses."""
    _stub_torch_compile_config(monkeypatch)
    model = _StubModel(_tiny_config())
    observed: list[bool] = []

    def _fake_compile(fn, dynamic=None):
        def _wrapped(*args, **kwargs):
            observed.append(torch.is_inference_mode_enabled())
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    arkasr_stages._compile_arkasr_audio_encoder(
        model, warmup_mel_frames=32, warmup_inference_mode=True
    )
    assert observed and all(observed)

    observed.clear()
    model = _StubModel(_tiny_config())
    arkasr_stages._compile_arkasr_audio_encoder(
        model, warmup_mel_frames=32, warmup_inference_mode=False
    )
    assert observed and not any(observed)


@pytest.mark.parametrize("frames", [0, 1, 3])
def test_compile_rejects_warmup_too_short_to_build_a_symbolic_graph(
    monkeypatch: pytest.MonkeyPatch, frames: int
) -> None:
    _stub_torch_compile_config(monkeypatch)
    model = _StubModel(_tiny_config())

    with pytest.raises(ValueError, match="warmup_mel_frames"):
        arkasr_stages._compile_arkasr_audio_encoder(model, warmup_mel_frames=frames)


def test_encoder_compile_is_off_by_default() -> None:
    import inspect

    signature = inspect.signature(arkasr_stages.create_sglang_arkasr_executor)
    assert signature.parameters["enable_encoder_torch_compile"].default is False

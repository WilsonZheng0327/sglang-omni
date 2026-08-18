# SPDX-License-Identifier: Apache-2.0
"""A-PR5: dynamic-shape torch.compile of the ARK-ASR audio encoder."""

from __future__ import annotations

import pytest
import torch
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.encoder_compile import compile_arkasr_audio_encoder

_WARMUP_FRAMES = 16


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
    """Only the attributes the compile helper touches."""

    def __init__(self, config: ArkasrConfig) -> None:
        super().__init__()
        self.config = config
        self.audio_encoder = ArkAudioMLPAdapter(config)


def _stub_torch_compile_config(monkeypatch) -> None:
    # The helper calls sglang's set_torch_compile_config, which mutates global
    # dynamo/inductor config; keep unit tests side-effect free.
    import sglang.srt.compilation.torch_compile_decoration as torch_compile_decoration

    monkeypatch.setattr(
        torch_compile_decoration, "set_torch_compile_config", lambda: None
    )


def test_compiles_tower_and_adapter_mlp_with_dynamic_shapes(monkeypatch) -> None:
    _stub_torch_compile_config(monkeypatch)
    model = _StubModel(_tiny_config())
    original_tower_forward = model.audio_encoder.whisper.forward
    original_adapting_forward = model.audio_encoder.adapting.forward
    adapter_forward = type(model.audio_encoder).forward
    param_names = set(dict(model.audio_encoder.named_parameters()))

    compile_calls = []

    def _fake_compile(fn, dynamic=None):
        compile_calls.append({"fn": fn, "dynamic": dynamic})
        return fn

    monkeypatch.setattr(torch, "compile", _fake_compile)

    compile_arkasr_audio_encoder(model, warmup_mel_frames=_WARMUP_FRAMES)

    assert [call["dynamic"] for call in compile_calls] == [True, True]
    assert compile_calls[0]["fn"] == original_tower_forward
    assert compile_calls[1]["fn"] == original_adapting_forward
    # ArkAudioMLPAdapter.forward stays eager: its frame-merge branches on
    # seq_len % merge_factor, which a symbolic length would only specialize,
    # and both arms are a slice and a reshape.
    assert type(model.audio_encoder).forward is adapter_forward
    # Bound-method compile must leave the module tree and parameter names
    # intact -- load_weights matches checkpoint names against named_parameters,
    # so no _orig_mod prefixes may appear.
    assert set(dict(model.audio_encoder.named_parameters())) == param_names


def test_warmup_covers_both_serving_signatures(monkeypatch) -> None:
    """get_audio_feature emits B1/mask=None and B>=2/mask=tensor, and Dynamo
    guards on batch 1 and on None-vs-tensor separately."""
    _stub_torch_compile_config(monkeypatch)
    model = _StubModel(_tiny_config())
    tower_calls = []

    original_tower_forward = model.audio_encoder.whisper.forward

    def _fake_compile(fn, dynamic=None):
        if fn != original_tower_forward:
            return fn

        def _wrapped(input_features, attention_mask=None):
            tower_calls.append((tuple(input_features.shape), attention_mask))
            return fn(input_features, attention_mask=attention_mask)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    compile_arkasr_audio_encoder(model, warmup_mel_frames=_WARMUP_FRAMES)

    assert len(tower_calls) == 2
    (single_shape, single_mask), (batched_shape, batched_mask) = tower_calls
    assert single_shape == (1, 8, _WARMUP_FRAMES)
    assert single_mask is None
    assert batched_shape == (2, 8, _WARMUP_FRAMES)
    assert batched_mask is not None
    assert batched_mask.shape == (2, _WARMUP_FRAMES)
    assert batched_mask.dtype == torch.bool
    # A ragged mask, matching a real mixed-length group rather than a
    # degenerate all-valid one.
    assert batched_mask.sum(dim=1).tolist() == [_WARMUP_FRAMES, _WARMUP_FRAMES // 2]


def test_warmup_tensors_are_allocated_in_the_serving_grad_mode(monkeypatch) -> None:
    # The pre-LM encoder service encodes under torch.inference_mode, and Dynamo
    # guards on the input's dispatch key set. Inference-mode tensors lack
    # ADInplaceOrView, so the warmup tensor must be *allocated* inside the
    # context -- merely calling inside it leaves a normal tensor whose graph the
    # service's tensors fail, paying a full recompile on the first real request.
    _stub_torch_compile_config(monkeypatch)
    modes = []

    def _fake_compile(fn, dynamic=None):
        def _wrapped(*args, **kwargs):
            modes.append(
                (torch.is_inference_mode_enabled(), torch.is_inference(args[0]))
            )
            return fn(*args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    compile_arkasr_audio_encoder(
        _StubModel(_tiny_config()), warmup_mel_frames=_WARMUP_FRAMES
    )
    # Two signatures x (tower + adapter MLP).
    assert modes == [(True, True)] * 4

    modes.clear()
    compile_arkasr_audio_encoder(
        _StubModel(_tiny_config()),
        warmup_mel_frames=_WARMUP_FRAMES,
        warmup_inference_mode=False,
    )
    assert modes == [(False, False)] * 4


# merge_factor=4: conv2 halves the mel length and the adapter merges 4 frames,
# so 16 mel frames is the smallest warmup leaving 2 merged frames. 15 and below
# collapse the merged axis to 0 or 1, which Dynamo shape-specializes.
@pytest.mark.parametrize("mel_frames", [2, 4, 15])
def test_rejects_warmup_lengths_that_collapse_a_dynamic_axis(
    monkeypatch, mel_frames: int
) -> None:
    _stub_torch_compile_config(monkeypatch)

    def _fail_compile(fn, dynamic=None):
        raise AssertionError("torch.compile must not run for invalid warmup")

    monkeypatch.setattr(torch, "compile", _fail_compile)

    with pytest.raises(ValueError, match="warmup_mel_frames"):
        compile_arkasr_audio_encoder(
            _StubModel(_tiny_config()), warmup_mel_frames=mel_frames
        )


def test_minimum_warmup_length_tracks_merge_factor(monkeypatch) -> None:
    """The bound is not a constant: a larger frame merge needs a longer warmup
    to leave the merged axis dynamic."""
    _stub_torch_compile_config(monkeypatch)
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic=None: fn)
    config = _tiny_config()
    config.merge_factor = 8
    model = _StubModel(config)

    with pytest.raises(ValueError, match="merge_factor=8"):
        compile_arkasr_audio_encoder(model, warmup_mel_frames=16)

    # 4 * merge_factor is accepted, and leaves 2 merged frames.
    compile_arkasr_audio_encoder(model, warmup_mel_frames=32)


def test_default_warmup_length_is_accepted(monkeypatch) -> None:
    _stub_torch_compile_config(monkeypatch)
    merged = []

    def _fake_compile(fn, dynamic=None):
        def _wrapped(x, *args, **kwargs):
            merged.append(tuple(x.shape))
            return fn(x, *args, **kwargs)

        return _wrapped

    monkeypatch.setattr(torch, "compile", _fake_compile)

    compile_arkasr_audio_encoder(_StubModel(_tiny_config()))

    # The adapter MLP sees [B, merged_frames, D * merge_factor]; the default
    # 256-frame warmup must leave that middle axis well clear of 0/1.
    adapter_shapes = [shape for shape in merged if shape[-1] == 32 * 4]
    assert adapter_shapes
    assert all(shape[1] >= 2 for shape in adapter_shapes)


def test_compiled_encoder_is_numerically_unchanged(monkeypatch) -> None:
    """torch.compile is a no-op on results; guard the plumbing (bound-method
    reassignment, mask threading) against silently changing the output."""
    _stub_torch_compile_config(monkeypatch)
    torch.manual_seed(0)
    model = _StubModel(_tiny_config()).eval()
    mel = torch.randn(2, 8, 40)
    mask = torch.zeros(2, 40, dtype=torch.bool)
    mask[0, :] = True
    mask[1, :24] = True

    with torch.no_grad():
        expected = model.audio_encoder(mel, attention_mask=mask)

    # Stand in for the real backend: identity wrapping keeps the test on CPU
    # and off inductor, while still exercising the reassigned forwards.
    monkeypatch.setattr(torch, "compile", lambda fn, dynamic=None: fn)
    compile_arkasr_audio_encoder(model, warmup_mel_frames=_WARMUP_FRAMES)

    with torch.no_grad():
        actual = model.audio_encoder(mel, attention_mask=mask)

    assert torch.equal(actual, expected)

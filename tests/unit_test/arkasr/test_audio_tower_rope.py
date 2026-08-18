# SPDX-License-Identifier: Apache-2.0
"""The cached ARK-ASR RoPE table must be numerically identical to the
per-forward recompute it replaces, and must stay out of ``state_dict``."""

from __future__ import annotations

import pytest
import torch
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_tower import ArkAudioTower, ArkRotaryEmbedding
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig

_MAX_POSITION = 64
_DIM = 8


def _reference_emb(dim: int, seq_len: int, rope_ratio: int = 1) -> torch.Tensor:
    """The checkpoint's original per-forward computation, verbatim."""
    base = 10000 * rope_ratio
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    t = torch.arange(seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def _tiny_config() -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=8,
        max_source_positions=_MAX_POSITION,
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


@pytest.mark.parametrize("seq_len", [2, 17, _MAX_POSITION])
def test_cached_table_matches_the_original_recompute(seq_len: int) -> None:
    rope = ArkRotaryEmbedding(_DIM, max_position=_MAX_POSITION)

    emb = rope.get_emb(seq_len, torch.float32, torch.device("cpu"))

    assert emb.shape == (seq_len, _DIM // 2, 2)
    assert torch.equal(emb, _reference_emb(_DIM, seq_len))


def test_repeated_calls_slice_one_table_instead_of_rebuilding() -> None:
    rope = ArkRotaryEmbedding(_DIM, max_position=_MAX_POSITION)

    first = rope.get_emb(16, torch.float32, torch.device("cpu"))
    second = rope.get_emb(16, torch.float32, torch.device("cpu"))

    # Both are views of the single buffer: no arange/outer/cos/sin on the hot
    # path, and nothing new allocated per encoder forward.
    assert first.data_ptr() == rope.rope_cache.data_ptr()
    assert second.data_ptr() == rope.rope_cache.data_ptr()


def test_table_is_not_persisted_in_state_dict() -> None:
    """load_weights matches checkpoint names; a persistent buffer would add a
    key the checkpoint does not ship."""
    rope = ArkRotaryEmbedding(_DIM, max_position=_MAX_POSITION)

    assert "rope_cache" not in rope.state_dict()
    assert "rotary_embedding.rope_cache" not in ArkAudioTower(
        _tiny_config()
    ).state_dict()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_request_casts_like_the_original(dtype: torch.dtype) -> None:
    rope = ArkRotaryEmbedding(_DIM, max_position=_MAX_POSITION)

    emb = rope.get_emb(12, dtype, torch.device("cpu"))

    assert emb.dtype == dtype
    assert torch.equal(emb, _reference_emb(_DIM, 12).to(dtype))


def test_overlong_sequence_recomputes_instead_of_truncating() -> None:
    """The table is an optimization, not a new length limit."""
    rope = ArkRotaryEmbedding(_DIM, max_position=_MAX_POSITION)
    seq_len = _MAX_POSITION + 5

    emb = rope.get_emb(seq_len, torch.float32, torch.device("cpu"))

    assert emb.shape == (seq_len, _DIM // 2, 2)
    assert torch.equal(emb, _reference_emb(_DIM, seq_len))


def test_tower_sizes_its_table_from_max_source_positions() -> None:
    tower = ArkAudioTower(_tiny_config())

    assert tower.rotary_embedding.rope_cache.shape[0] == _MAX_POSITION


def test_tower_forward_is_unchanged_by_the_cache() -> None:
    """End-to-end guard: the tower output must equal what it produces when the
    table is recomputed per call from the reference formula."""
    torch.manual_seed(0)
    cfg = _tiny_config()
    tower = ArkAudioTower(cfg).eval()
    mel = torch.randn(1, cfg.whisper_config.num_mel_bins, 40)

    with torch.no_grad():
        cached = tower(mel)

    head_dim = cfg.whisper_config.d_model // cfg.whisper_config.encoder_attention_heads
    recomputed = _reference_emb(head_dim // 2, 20)
    original_get_emb = tower.rotary_embedding.get_emb
    try:
        tower.rotary_embedding.get_emb = (
            lambda seq_len, dtype, device: recomputed[:seq_len]
        )
        with torch.no_grad():
            reference = tower(mel)
    finally:
        tower.rotary_embedding.get_emb = original_get_emb

    assert torch.equal(cached, reference)

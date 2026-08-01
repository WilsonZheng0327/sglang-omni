# SPDX-License-Identifier: Apache-2.0
"""Batched mel encoding must produce exactly what the per-item loop produced.

The tower now runs once over a pad-to-max batch with the padding masked out, so
the property that matters is that every item's rows are byte-for-byte the rows it
would get if it were the only request in flight.
"""

from __future__ import annotations

import pytest
import torch
from transformers import WhisperConfig

from sglang_omni.models.arkasr.audio_lengths import arkasr_num_audio_tokens
from sglang_omni.models.arkasr.audio_tower import ArkAudioMLPAdapter
from sglang_omni.models.arkasr.configuration_arkasr import ArkasrConfig
from sglang_omni.models.arkasr.sglang_model import ArkasrForConditionalGeneration

_MEL_BINS = 8
_MERGE_FACTOR = 4
# Batched matmuls reassociate differently from single-row ones, so equality is
# to float32 rounding, not bitwise.
_TOL = dict(atol=1e-5, rtol=1e-4)


def _tiny_config() -> ArkasrConfig:
    whisper = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=_MEL_BINS,
        max_source_positions=512,
    )
    return ArkasrConfig(
        whisper_config=whisper,
        merge_factor=_MERGE_FACTOR,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=256,
        audio_token_id=151663,
    )


class _StubModel:
    """Only ``get_audio_feature`` is under test; the LM half is irrelevant."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.audio_encoder = ArkAudioMLPAdapter(_tiny_config()).eval()

    def get_audio_feature(self, items):  # noqa: ANN001
        return ArkasrForConditionalGeneration.get_audio_feature(self, items)


class _Item:
    def __init__(self, mel_frames: int, *, with_mask: bool = True) -> None:
        torch.manual_seed(mel_frames)
        self.feature = torch.randn(1, _MEL_BINS, mel_frames)
        self.feature_attention_mask = (
            torch.ones((1, mel_frames), dtype=torch.long) if with_mask else None
        )
        if not with_mask:
            del self.feature_attention_mask


def _encode_alone(model: _StubModel, item: _Item) -> torch.Tensor:
    return model.get_audio_feature([item])


@pytest.mark.parametrize(
    "mel_frames",
    [
        [40, 40],  # equal lengths: no padding at all
        [40, 24],  # ragged
        [17, 64, 33],  # ragged, odd frame counts, widest last
        [64, 8],  # extreme ratio
    ],
)
def test_batched_encoding_matches_per_item_encoding(mel_frames: list[int]) -> None:
    model = _StubModel()
    items = [_Item(frames) for frames in mel_frames]

    with torch.no_grad():
        batched = model.get_audio_feature(items)
        solo = [_encode_alone(model, item) for item in items]

    assert batched.shape == torch.cat(solo, dim=0).shape
    offset = 0
    for frames, expected in zip(mel_frames, solo):
        rows = arkasr_num_audio_tokens(frames, _MERGE_FACTOR)
        assert expected.shape[0] == rows
        torch.testing.assert_close(batched[offset : offset + rows], expected, **_TOL)
        offset += rows
    assert offset == batched.shape[0]


def test_every_length_parity_matches_solo_encoding() -> None:
    """Sweep the conv boundary: only odd mel widths make conv2's window at the
    last real downsampled frame reach into the padding, so parity has to be
    covered exhaustively rather than sampled."""
    model = _StubModel()
    widest = 48
    for frames in range(5, widest + 1):
        item = _Item(frames)
        with torch.no_grad():
            alone = model.get_audio_feature([item])
            batched = model.get_audio_feature([item, _Item(widest)])
        rows = arkasr_num_audio_tokens(frames, _MERGE_FACTOR)
        assert alone.shape[0] == rows
        torch.testing.assert_close(
            batched[:rows], alone, **_TOL, msg=f"mel_frames={frames}"
        )


def test_row_count_matches_the_scatter_contract() -> None:
    """The flat output must have exactly the rows the <|audio|> placeholders
    reserved, or embeddings land on the wrong positions."""
    model = _StubModel()
    mel_frames = [40, 24, 17]
    items = [_Item(frames) for frames in mel_frames]

    with torch.no_grad():
        batched = model.get_audio_feature(items)

    expected_rows = sum(
        arkasr_num_audio_tokens(frames, _MERGE_FACTOR) for frames in mel_frames
    )
    assert batched.shape[0] == expected_rows
    assert batched.shape[1] == _tiny_config().hidden_size


def test_short_clip_padded_to_one_merge_group_matches_solo_encoding() -> None:
    """A clip shorter than merge_factor downsampled frames is zero-padded up to
    one group; batching must not merge encoder noise from the padded region."""
    model = _StubModel()
    short = _Item(6)  # -> 3 downsampled frames < merge_factor
    long = _Item(64)

    with torch.no_grad():
        batched = model.get_audio_feature([short, long])
        solo_short = _encode_alone(model, short)

    assert solo_short.shape[0] == 1
    torch.testing.assert_close(batched[:1], solo_short, **_TOL)


def test_padding_is_masked_out_of_attention() -> None:
    """Extending the batch width with padding must not change a shorter item's
    embedding — the proof that the additive mask reaches SDPA."""
    model = _StubModel()
    item = _Item(24)

    with torch.no_grad():
        alone = model.get_audio_feature([item])
        with_wide_neighbour = model.get_audio_feature([item, _Item(256)])

    torch.testing.assert_close(
        with_wide_neighbour[: alone.shape[0]], alone, **_TOL
    )


def test_missing_mask_falls_back_to_full_length() -> None:
    model = _StubModel()
    masked = _Item(40)
    unmasked = _Item(40, with_mask=False)

    with torch.no_grad():
        from_mask = model.get_audio_feature([masked])
        from_shape = model.get_audio_feature([unmasked])

    torch.testing.assert_close(from_mask, from_shape, **_TOL)


def test_mask_width_mismatch_raises() -> None:
    model = _StubModel()
    item = _Item(40)
    item.feature_attention_mask = torch.ones((1, 39), dtype=torch.long)

    with pytest.raises(ValueError, match="does not match"):
        model.get_audio_feature([item])


def test_two_dimensional_feature_is_accepted() -> None:
    """The historical (num_mel_bins, T) shape must still work."""
    model = _StubModel()
    item = _Item(40)
    flat = _Item(40)
    flat.feature = flat.feature.squeeze(0)

    with torch.no_grad():
        torch.testing.assert_close(
            model.get_audio_feature([flat]), model.get_audio_feature([item]), **_TOL
        )


def test_tower_mask_leaves_unpadded_batches_untouched() -> None:
    """With no padding the mask is all-zeros additive, so results must equal the
    no-mask path the single-item loop used."""
    config = _tiny_config()
    torch.manual_seed(0)
    adapter = ArkAudioMLPAdapter(config).eval()
    mel = torch.randn(2, _MEL_BINS, 40)

    with torch.no_grad():
        without = adapter(mel)
        with_lengths = adapter(mel, mel_lengths=torch.tensor([40, 40]))

    torch.testing.assert_close(with_lengths, without, **_TOL)

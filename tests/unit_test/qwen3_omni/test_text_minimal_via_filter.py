# SPDX-License-Identifier: Apache-2.0
"""Integration: Qwen3-Omni text-minimal pipeline via framework filter.

Verifies that constructing Qwen3OmniPipelineConfig with
enabled_stages=["preprocessing", "decode"] yields the measured 3-stage
minimal pipeline from RFC #462. The framework auto-includes thinker
(required=True) and rewires preprocessing -> thinker via the model's
declared next_fallback / project_payload_fallback.
"""

from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig


@pytest.fixture
def minimal_config() -> Qwen3OmniPipelineConfig:
    return Qwen3OmniPipelineConfig(
        model_path="fake/Qwen3-Omni-30B-A3B-Instruct",
        enabled_stages=["preprocessing", "decode"],
    )


def _stage(config, name):
    return next(s for s in config.stages if s.name == name)


def test_minimal_pipeline_has_exactly_three_stages(minimal_config):
    names = [s.name for s in minimal_config.stages]
    assert set(names) == {"preprocessing", "thinker", "decode"}


def test_image_encoder_pruned(minimal_config):
    names = {s.name for s in minimal_config.stages}
    assert "image_encoder" not in names


def test_audio_encoder_pruned(minimal_config):
    names = {s.name for s in minimal_config.stages}
    assert "audio_encoder" not in names


def test_mm_aggregate_pruned(minimal_config):
    names = {s.name for s in minimal_config.stages}
    assert "mm_aggregate" not in names


def test_thinker_auto_included_via_required(minimal_config):
    """Thinker is required=True, so the framework keeps it even though
    the caller did not list it in enabled_stages."""
    thinker = _stage(minimal_config, "thinker")
    assert thinker.required is True


def test_preprocessing_next_rewired_to_thinker(minimal_config):
    prep = _stage(minimal_config, "preprocessing")
    targets = prep.next if isinstance(prep.next, list) else [prep.next]
    assert targets == ["thinker"]


def test_preprocessing_projection_uses_textonly_fallback(minimal_config):
    prep = _stage(minimal_config, "preprocessing")
    assert "thinker" in prep.project_payload
    assert prep.project_payload["thinker"].endswith(
        "project_preprocessing_to_thinker_textonly"
    )
    # Pre-fallback projections targeting pruned stages are gone.
    assert "image_encoder" not in prep.project_payload
    assert "audio_encoder" not in prep.project_payload
    assert "mm_aggregate" not in prep.project_payload


def test_default_pipeline_still_has_six_stages():
    """No enabled_stages set -> default 6-stage text pipeline unchanged."""
    default_config = Qwen3OmniPipelineConfig(
        model_path="fake/Qwen3-Omni-30B-A3B-Instruct"
    )
    names = [s.name for s in default_config.stages]
    assert names == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]


def test_decode_marked_required():
    default_config = Qwen3OmniPipelineConfig(
        model_path="fake/Qwen3-Omni-30B-A3B-Instruct"
    )
    decode = _stage(default_config, "decode")
    assert decode.required is True


def test_omitting_decode_keeps_decode_via_required():
    """Caller can leave decode out; required=True auto-includes it."""
    config = Qwen3OmniPipelineConfig(
        model_path="fake/Qwen3-Omni-30B-A3B-Instruct",
        enabled_stages=["preprocessing"],
    )
    names = {s.name for s in config.stages}
    # preprocessing (whitelist) + thinker + decode (both required) = 3
    assert names == {"preprocessing", "thinker", "decode"}


def test_partial_multimodal_prune_uses_guarded_mm_aggregate_projection():
    config = Qwen3OmniPipelineConfig(
        model_path="fake/Qwen3-Omni-30B-A3B-Instruct",
        enabled_stages=["preprocessing", "mm_aggregate", "decode"],
    )

    prep = _stage(config, "preprocessing")
    assert prep.project_payload["mm_aggregate"].endswith(
        "project_preprocessing_to_mm_aggregate_textonly"
    )

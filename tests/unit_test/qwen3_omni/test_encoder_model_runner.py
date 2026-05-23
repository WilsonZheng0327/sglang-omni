# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.model_runner.encoder_model_runner import EncoderModelRunner
from sglang_omni.models.qwen3_omni.encoder_model_runner import (
    Qwen3OmniAudioEncoderModelRunner,
    Qwen3OmniImageEncoderModelRunner,
)
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.scheduling.stage_cache import StageOutputCache
from tests.unit_test.fixtures.qwen_fakes import (
    FakeAudioEncoderModel,
    FakeImageEncoderModel,
    make_qwen_payload,
    make_qwen_state,
)


class _GraphDispatchRunner(EncoderModelRunner):
    def __init__(self, *, use_graph: bool) -> None:
        super().__init__(model=object(), stage_name="test_encoder")
        self.use_graph = use_graph

    def load_state(self, payload: Any) -> Any:
        return payload

    def store_state(self, payload: Any, state: Any) -> Any:
        del state
        return payload

    def build_encoder_request(self, payload: Any, state: Any) -> Any:
        del state
        return payload

    def apply_result(self, state: Any, result: Any) -> None:
        del state, result

    def can_run_cuda_graph(self, prepared: Any) -> bool:
        del prepared
        return self.use_graph

    def forward_eager(self, prepared: Any) -> str:
        del prepared
        return "eager"

    def forward_cuda_graph(self, prepared: Any) -> str:
        del prepared
        return "graph"


def test_encoder_model_runner_dispatches_cuda_graph_from_forward() -> None:
    assert _GraphDispatchRunner(use_graph=True).forward({}) == "graph"
    assert _GraphDispatchRunner(use_graph=False).forward({}) == "eager"


def test_encoder_model_runner_reuses_static_buffers_until_capture() -> None:
    runner = _GraphDispatchRunner(use_graph=False)
    first = runner.static_input_buffer(
        "small",
        "input",
        shape=(2, 3),
        dtype=torch.float32,
        device="cpu",
    )
    second = runner.static_input_buffer(
        "small",
        "input",
        shape=(2, 3),
        dtype=torch.float32,
        device="cpu",
    )
    assert second.data_ptr() == first.data_ptr()

    resized = runner.static_input_buffer(
        "small",
        "input",
        shape=(3, 3),
        dtype=torch.float32,
        device="cpu",
    )
    assert resized.shape == (3, 3)

    runner.cuda_graphs["small"] = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Cannot resize CUDA graph buffer"):
        runner.static_input_buffer(
            "small",
            "input",
            shape=(4, 3),
            dtype=torch.float32,
            device="cpu",
        )


def test_qwen_image_encoder_runner_batches_splits_and_uses_cache() -> None:
    model = FakeImageEncoderModel()
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        cache=StageOutputCache(max_size=8, cache_device="cpu"),
    )
    state = make_qwen_state(
        encoder_inputs={
            "image_encoder": {
                "cache_key": "shared-image",
                "pixel_values": torch.ones((2, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            }
        }
    )
    first = make_qwen_payload(state, request_id="image-1")
    duplicate = make_qwen_payload(state, request_id="image-2")

    outputs = runner.execute_batch([first, duplicate])

    assert len(outputs) == 2
    assert len(model.calls) == 1
    for payload in outputs:
        out_state = PipelineState.from_dict(payload.data)
        image_out = out_state.encoder_outs["image_encoder"]
        assert image_out["image_embeds"].shape == (2, 2)
        assert image_out["image_token_counts"].tolist() == [2]
        assert image_out["deepstack_visual_embeds_image"][0].shape == (2, 2)

    cached = runner.execute(make_qwen_payload(state, request_id="image-3"))

    assert len(model.calls) == 1
    cached_state = PipelineState.from_dict(cached.data)
    assert cached_state.encoder_outs["image_encoder"]["image_embeds"].shape == (2, 2)


def test_qwen_image_encoder_graph_body_matches_visual_forward() -> None:
    config_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe"
    )
    modeling_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
    )

    config = config_module.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    visual = modeling_module.Qwen3OmniMoeVisionEncoder(config).eval()
    model = SimpleNamespace(
        visual=visual,
        spatial_merge_size=1,
        out_hidden_size=8,
        deepstack_layers=1,
        visual_dtype_bytes=4,
    )
    runner = Qwen3OmniImageEncoderModelRunner(
        model=model,
        enable_cuda_graph=False,
    )
    pixel_values = torch.randn(4, 12)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    with torch.no_grad():
        eager_embeds, eager_deepstack = visual(pixel_values, grid_thw)
        prepared = runner._prepare_visual_graph_inputs(pixel_values, grid_thw)
        graph_body = runner._forward_visual_graph_body(
            hidden_states=prepared["hidden_states"],
            cu_seqlens=prepared["cu_seqlens"],
            position_embeddings=prepared["position_embeddings"],
        )

    torch.testing.assert_close(graph_body["embeds"], eager_embeds)
    assert len(graph_body["deepstack"]) == len(eager_deepstack)
    for graph_tensor, eager_tensor in zip(
        graph_body["deepstack"],
        eager_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_qwen_image_encoder_runner_replays_cuda_graph() -> None:
    config_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe"
    )
    modeling_module = pytest.importorskip(
        "transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe"
    )

    config = config_module.Qwen3OmniMoeVisionEncoderConfig(
        depth=2,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        out_hidden_size=8,
        num_position_embeddings=16,
        spatial_merge_size=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        deepstack_visual_indexes=[0],
    )
    visual = modeling_module.Qwen3OmniMoeVisionEncoder(config).cuda().eval()
    model = SimpleNamespace(
        visual=visual,
        spatial_merge_size=1,
        out_hidden_size=8,
        deepstack_layers=1,
        visual_dtype_bytes=4,
    )
    runner = Qwen3OmniImageEncoderModelRunner(model=model)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)

    def run_graph(pixel_values: torch.Tensor) -> dict[str, Any]:
        return runner.forward(
            {
                "model_inputs": {
                    "pixel_values": pixel_values,
                    "image_grid_thw": grid_thw,
                },
                "metas": [],
            }
        )

    with torch.no_grad():
        first_pixels = torch.randn(4, 12)
        first_eager, first_deepstack = visual(first_pixels.cuda(), grid_thw.cuda())
        first_graph = run_graph(first_pixels)

        second_pixels = torch.randn(4, 12)
        second_eager, second_deepstack = visual(second_pixels.cuda(), grid_thw.cuda())
        second_graph = run_graph(second_pixels)

    assert len(runner.cuda_graphs) == 1
    torch.testing.assert_close(first_graph["image_embeds"], first_eager)
    torch.testing.assert_close(second_graph["image_embeds"], second_eager)
    for graph_tensor, eager_tensor in zip(
        first_graph["deepstack_visual_embeds_image"],
        first_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)
    for graph_tensor, eager_tensor in zip(
        second_graph["deepstack_visual_embeds_image"],
        second_deepstack,
    ):
        torch.testing.assert_close(graph_tensor, eager_tensor)


def test_qwen_audio_encoder_runner_pads_and_splits_batch_outputs() -> None:
    model = FakeAudioEncoderModel()
    runner = Qwen3OmniAudioEncoderModelRunner(model=model)
    first = make_qwen_payload(
        make_qwen_state(
            encoder_inputs={
                "audio_encoder": {
                    "input_features": torch.ones((1, 2, 2)),
                    "audio_feature_lengths": torch.tensor([2]),
                }
            }
        ),
        request_id="audio-1",
    )
    second = make_qwen_payload(
        make_qwen_state(
            encoder_inputs={
                "audio_encoder": {
                    "input_features": torch.ones((1, 2, 3)),
                    "audio_feature_lengths": torch.tensor([3]),
                }
            }
        ),
        request_id="audio-2",
    )

    outputs = runner.execute_batch([first, second])

    assert len(outputs) == 2
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["input_features"].shape == (2, 2, 3)
    assert call["audio_feature_lengths"].tolist() == [2, 3]

    first_out = PipelineState.from_dict(outputs[0].data).encoder_outs["audio_encoder"]
    second_out = PipelineState.from_dict(outputs[1].data).encoder_outs["audio_encoder"]
    assert first_out["audio_embeds"].shape == (2, 2)
    assert second_out["audio_embeds"].shape == (3, 2)
    assert first_out["audio_output_lengths"].tolist() == [2]
    assert second_out["audio_output_lengths"].tolist() == [3]

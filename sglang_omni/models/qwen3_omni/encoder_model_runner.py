# SPDX-License-Identifier: Apache-2.0
"""Encoder model runners for Qwen3-Omni."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from sglang_omni.model_runner.encoder_model_runner import (
    EncoderBatchItem,
    EncoderModelRunner,
    tensor_bytes,
)
from sglang_omni.models.qwen3_omni.payload_types import PipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    apply_encoder_result,
    build_encoder_request,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache

QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER = 5


class Qwen3OmniEncoderModelRunner(EncoderModelRunner):
    def load_state(self, payload: StagePayload) -> PipelineState:
        return PipelineState.from_dict(payload.data)

    def store_state(self, payload: StagePayload, state: PipelineState) -> StagePayload:
        payload.data = state.to_dict()
        return payload

    def build_encoder_request(
        self,
        payload: StagePayload,
        state: PipelineState,
    ) -> Any:
        del payload
        return build_encoder_request(state, stage_name=self.stage_name)

    def apply_result(self, state: PipelineState, result: Any) -> None:
        apply_encoder_result(state, stage_name=self.stage_name, result=result)


class Qwen3OmniImageEncoderModelRunner(Qwen3OmniEncoderModelRunner):
    def __init__(
        self,
        *,
        model: Any,
        cache: StageOutputCache | None = None,
        enable_cuda_graph: bool = True,
    ) -> None:
        super().__init__(
            model=model,
            stage_name=IMAGE_STAGE,
            cache=cache,
            enable_cuda_graph=enable_cuda_graph,
        )

    def is_batchable(self, request: Any) -> bool:
        if self.request_skip_result(request) is not None:
            return False
        input_dict = self.request_model_inputs(request)
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            value = input_dict.get(key)
            if value is not None and not isinstance(value, torch.Tensor):
                return False
        return True

    def estimate_request_cost(self, request: Any) -> int:
        merge = int(self.model.spatial_merge_size) ** 2
        hidden = int(self.model.out_hidden_size)
        output_layers = 1 + int(self.model.deepstack_layers)
        dtype_bytes = int(self.model.visual_dtype_bytes)
        model_inputs = self.request_model_inputs(request)

        raw_bytes = tensor_bytes(model_inputs.get("pixel_values"))
        raw_bytes += tensor_bytes(model_inputs.get("pixel_values_videos"))
        visual_tokens = _grid_visual_tokens(model_inputs.get("image_grid_thw"), merge)
        visual_tokens += _grid_visual_tokens(model_inputs.get("video_grid_thw"), merge)
        output_bytes = visual_tokens * hidden * dtype_bytes * output_layers
        return (raw_bytes + output_bytes) * QWEN3_IMAGE_ENCODER_ACTIVATION_MULTIPLIER

    def prepare(self, items: list[EncoderBatchItem]) -> dict[str, Any]:
        image_pixels: list[torch.Tensor] = []
        image_grids: list[torch.Tensor] = []
        video_pixels: list[torch.Tensor] = []
        video_grids: list[torch.Tensor] = []
        metas: list[dict[str, Any]] = []
        merge = self.model.spatial_merge_size**2

        for item in items:
            input_dict = self.request_model_inputs(item.request)
            image_grid = input_dict.get("image_grid_thw")
            video_grid = input_dict.get("video_grid_thw")
            image_rows = (
                int(image_grid.shape[0]) if isinstance(image_grid, torch.Tensor) else 0
            )
            video_rows = (
                int(video_grid.shape[0]) if isinstance(video_grid, torch.Tensor) else 0
            )
            image_token_counts = (
                (image_grid.prod(-1) // merge).to(dtype=torch.long)
                if isinstance(image_grid, torch.Tensor)
                else None
            )
            video_token_counts = (
                (video_grid.prod(-1) // merge).to(dtype=torch.long)
                if isinstance(video_grid, torch.Tensor)
                else None
            )
            image_token_total = (
                int(image_token_counts.sum().item())
                if isinstance(image_token_counts, torch.Tensor)
                else 0
            )
            video_token_total = (
                int(video_token_counts.sum().item())
                if isinstance(video_token_counts, torch.Tensor)
                else 0
            )
            if isinstance(input_dict.get("pixel_values"), torch.Tensor):
                image_pixels.append(input_dict["pixel_values"])
                image_grids.append(image_grid)
            if isinstance(input_dict.get("pixel_values_videos"), torch.Tensor):
                video_pixels.append(input_dict["pixel_values_videos"])
                video_grids.append(video_grid)
            metas.append(
                {
                    "item": item,
                    "image_rows": image_rows,
                    "video_rows": video_rows,
                    "image_token_total": image_token_total,
                    "video_token_total": video_token_total,
                }
            )

        model_inputs: dict[str, Any] = {}
        if image_pixels:
            model_inputs["pixel_values"] = torch.cat(image_pixels, dim=0)
            model_inputs["image_grid_thw"] = torch.cat(image_grids, dim=0)
        if video_pixels:
            model_inputs["pixel_values_videos"] = torch.cat(video_pixels, dim=0)
            model_inputs["video_grid_thw"] = torch.cat(video_grids, dim=0)

        return {"model_inputs": model_inputs, "metas": metas}

    def forward_eager(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return self.model(**prepared["model_inputs"])

    def cuda_graph_key(self, prepared: dict[str, Any]) -> Any | None:
        if not self._visual_cuda_graph_supported():
            return None

        model_inputs = prepared["model_inputs"]
        graph_keys = []
        image_key = self._visual_request_graph_key(
            model_inputs.get("pixel_values"),
            model_inputs.get("image_grid_thw"),
        )
        if image_key is not None:
            graph_keys.append(("image", image_key))

        video_key = self._visual_request_graph_key(
            model_inputs.get("pixel_values_videos"),
            model_inputs.get("video_grid_thw"),
        )
        if video_key is not None:
            graph_keys.append(("video", video_key))

        return tuple(graph_keys) if graph_keys else None

    def forward_cuda_graph(self, prepared: dict[str, Any]) -> dict[str, Any]:
        model_inputs = prepared["model_inputs"]
        outputs: dict[str, Any] = {}
        merge = self.model.spatial_merge_size**2

        if isinstance(model_inputs.get("pixel_values"), torch.Tensor):
            image_grid_thw = model_inputs["image_grid_thw"]
            image_embeds, image_multiscale = self._run_visual_cuda_graph(
                model_inputs["pixel_values"],
                image_grid_thw,
            )
            image_grid_thw = image_grid_thw.to(
                device=image_embeds.device,
                dtype=torch.long,
            )
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_grid_thw.prod(-1) // merge,
                    "deepstack_visual_embeds_image": image_multiscale,
                }
            )

        if isinstance(model_inputs.get("pixel_values_videos"), torch.Tensor):
            video_grid_thw = model_inputs["video_grid_thw"]
            video_embeds, video_multiscale = self._run_visual_cuda_graph(
                model_inputs["pixel_values_videos"],
                video_grid_thw,
            )
            video_grid_thw = video_grid_thw.to(
                device=video_embeds.device,
                dtype=torch.long,
            )
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_grid_thw.prod(-1) // merge,
                    "deepstack_visual_embeds_video": video_multiscale,
                }
            )

        return outputs

    def prepare_cuda_graph_capture(
        self,
        graph_key: Any,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        return self._copy_visual_graph_buffers(graph_key, prepared)

    def prepare_cuda_graph_replay(
        self,
        graph_key: Any,
        prepared: dict[str, Any],
    ) -> None:
        self._copy_visual_graph_buffers(graph_key, prepared)

    def forward_cuda_graph_capture(
        self,
        graph_key: Any,
        static_prepared: dict[str, Any],
    ) -> dict[str, Any]:
        del graph_key
        return self._forward_visual_graph_body(
            hidden_states=static_prepared["hidden_states"],
            cu_seqlens=static_prepared["cu_seqlens"],
            position_embeddings=static_prepared["position_embeddings"],
        )

    def post(self, prepared: dict[str, Any], combined: dict[str, Any]) -> list[Any]:
        image_grid_all = combined.get("image_grid_thw")
        image_counts_all = combined.get("image_token_counts")
        image_embeds_all = combined.get("image_embeds")
        image_multiscale_all = combined.get("deepstack_visual_embeds_image")
        video_grid_all = combined.get("video_grid_thw")
        video_counts_all = combined.get("video_token_counts")
        video_embeds_all = combined.get("video_embeds")
        video_multiscale_all = combined.get("deepstack_visual_embeds_video")

        image_row_cursor = 0
        image_token_cursor = 0
        video_row_cursor = 0
        video_token_cursor = 0
        results: list[Any] = []
        for meta in prepared["metas"]:
            stage_result: dict[str, Any] = {}
            if meta["image_rows"] > 0:
                row_end = image_row_cursor + meta["image_rows"]
                token_end = image_token_cursor + meta["image_token_total"]
                stage_result["image_embeds"] = _split_visual_features(
                    image_embeds_all,
                    start=image_token_cursor,
                    end=token_end,
                )
                stage_result["image_grid_thw"] = image_grid_all[
                    image_row_cursor:row_end
                ]
                stage_result["image_token_counts"] = image_counts_all[
                    image_row_cursor:row_end
                ]
                stage_result["deepstack_visual_embeds_image"] = (
                    _split_visual_multiscale(
                        image_multiscale_all,
                        start=image_token_cursor,
                        end=token_end,
                    )
                )
                image_row_cursor = row_end
                image_token_cursor = token_end
            if meta["video_rows"] > 0:
                row_end = video_row_cursor + meta["video_rows"]
                token_end = video_token_cursor + meta["video_token_total"]
                stage_result["video_embeds"] = _split_visual_features(
                    video_embeds_all,
                    start=video_token_cursor,
                    end=token_end,
                )
                stage_result["video_grid_thw"] = video_grid_all[
                    video_row_cursor:row_end
                ]
                stage_result["video_token_counts"] = video_counts_all[
                    video_row_cursor:row_end
                ]
                stage_result["deepstack_visual_embeds_video"] = (
                    _split_visual_multiscale(
                        video_multiscale_all,
                        start=video_token_cursor,
                        end=token_end,
                    )
                )
                video_row_cursor = row_end
                video_token_cursor = token_end
            results.append(stage_result)

        return results

    def _visual_cuda_graph_supported(self) -> bool:
        visual = getattr(self.model, "visual", None)
        if visual is None or getattr(visual, "training", False):
            return False
        required_attrs = (
            "patch_embed",
            "fast_pos_embed_interpolate",
            "rot_pos_emb",
            "blocks",
            "merger",
            "deepstack_visual_indexes",
            "deepstack_merger_list",
        )
        if any(not hasattr(visual, attr) for attr in required_attrs):
            return False
        try:
            return next(visual.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _visual_request_graph_key(
        self,
        pixel_values: Any,
        grid_thw: Any,
    ) -> Any | None:
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(
            grid_thw,
            torch.Tensor,
        ):
            return None
        if pixel_values.numel() == 0 or grid_thw.numel() == 0:
            return None

        grid_key = tuple(
            tuple(int(value) for value in row)
            for row in grid_thw.detach().to("cpu", dtype=torch.long).tolist()
        )
        return (
            "qwen3_omni_vision",
            str(pixel_values.dtype),
            tuple(pixel_values.shape),
            grid_key,
        )

    def _run_visual_cuda_graph(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        prepared = self._prepare_visual_graph_inputs(pixel_values, grid_thw)
        graph_output = self.run_cuda_graph_piece(prepared["graph_key"], prepared)

        # Graph outputs are stable replay buffers. Clone before storing them in
        # stage state/cache so a later replay cannot mutate an older payload.
        return (
            graph_output["embeds"].clone(),
            [tensor.clone() for tensor in graph_output["deepstack"]],
        )

    def _prepare_visual_graph_inputs(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> dict[str, Any]:
        visual = self.model.visual
        device = next(visual.parameters()).device
        dtype = next(visual.parameters()).dtype

        grid_thw = grid_thw.to(device=device, dtype=torch.long)
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        hidden_states = visual.patch_embed(pixel_values)
        hidden_states = hidden_states + visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb = visual.rot_pos_emb(grid_thw).reshape(
            hidden_states.shape[0], -1
        )
        rotary_pos_emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (
            rotary_pos_emb.cos().contiguous(),
            rotary_pos_emb.sin().contiguous(),
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        if self._visual_attention_impl() != "flash_attention_2":
            cu_seqlens = cu_seqlens.to("cpu")

        graph_key = self._visual_graph_key_from_inputs(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        return {
            "graph_key": graph_key,
            "hidden_states": hidden_states.contiguous(),
            "cu_seqlens": cu_seqlens.contiguous(),
            "position_embeddings": position_embeddings,
        }

    def _visual_graph_key_from_inputs(
        self,
        *,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> Any:
        return (
            "qwen3_omni_vision_blocks",
            self._visual_attention_impl(),
            str(hidden_states.dtype),
            str(hidden_states.device),
            tuple(hidden_states.shape),
            tuple(int(value) for value in cu_seqlens.detach().cpu().tolist()),
        )

    def _copy_visual_graph_buffers(
        self,
        graph_key: Any,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        hidden_states = prepared["hidden_states"]
        cu_seqlens = prepared["cu_seqlens"]
        position_cos, position_sin = prepared["position_embeddings"]

        hidden_buffer = self.static_input_buffer(
            graph_key,
            "hidden_states",
            shape=hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_buffer.copy_(hidden_states)

        cu_seqlens_buffer = self.static_metadata_buffer(
            graph_key,
            "cu_seqlens",
            shape=cu_seqlens.shape,
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        cu_seqlens_buffer.copy_(cu_seqlens)

        cos_buffer = self.static_metadata_buffer(
            graph_key,
            "position_cos",
            shape=position_cos.shape,
            dtype=position_cos.dtype,
            device=position_cos.device,
        )
        sin_buffer = self.static_metadata_buffer(
            graph_key,
            "position_sin",
            shape=position_sin.shape,
            dtype=position_sin.dtype,
            device=position_sin.device,
        )
        cos_buffer.copy_(position_cos)
        sin_buffer.copy_(position_sin)

        return {
            "hidden_states": hidden_buffer,
            "cu_seqlens": cu_seqlens_buffer,
            "position_embeddings": (cos_buffer, sin_buffer),
        }

    def _forward_visual_graph_body(
        self,
        *,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> dict[str, Any]:
        visual = self.model.visual
        deepstack_features: list[torch.Tensor] = []
        deepstack_index_by_layer = {
            int(layer_idx): idx
            for idx, layer_idx in enumerate(visual.deepstack_visual_indexes)
        }

        for layer_num, block in enumerate(visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            deepstack_idx = deepstack_index_by_layer.get(layer_num)
            if deepstack_idx is not None:
                deepstack_features.append(
                    visual.deepstack_merger_list[deepstack_idx](hidden_states)
                )

        return {
            "embeds": visual.merger(hidden_states),
            "deepstack": deepstack_features,
        }

    def _visual_attention_impl(self) -> str:
        visual = self.model.visual
        for block in visual.blocks:
            attn = getattr(block, "attn", None)
            config = getattr(attn, "config", None)
            implementation = getattr(config, "_attn_implementation", None)
            if implementation is not None:
                return str(implementation)
        return "unknown"


class Qwen3OmniAudioEncoderModelRunner(Qwen3OmniEncoderModelRunner):
    def __init__(
        self,
        *,
        model: Any,
        cache: StageOutputCache | None = None,
    ) -> None:
        super().__init__(model=model, stage_name=AUDIO_STAGE, cache=cache)

    def is_batchable(self, request: Any) -> bool:
        if self.request_skip_result(request) is not None:
            return False
        input_dict = self.request_model_inputs(request)
        features = input_dict.get("input_features")
        if not isinstance(features, torch.Tensor):
            return False
        lengths = input_dict.get("audio_feature_lengths")
        mask = input_dict.get("feature_attention_mask")
        return (lengths is None or isinstance(lengths, torch.Tensor)) and (
            mask is None or isinstance(mask, torch.Tensor)
        )

    def prepare(self, items: list[EncoderBatchItem]) -> dict[str, Any]:
        normalized = []
        max_time = 0
        for item in items:
            features, mask, lengths = _normalize_audio_request_tensors(item.request)
            max_time = max(max_time, int(features.shape[-1]))
            normalized.append(
                {
                    "item": item,
                    "features": features,
                    "mask": mask,
                    "lengths": lengths,
                    "count": int(lengths.shape[0]),
                }
            )

        batched_features = torch.cat(
            [_pad_audio_features(item["features"], max_time) for item in normalized],
            dim=0,
        )
        batched_mask = torch.cat(
            [_pad_audio_mask(item["mask"], max_time) for item in normalized],
            dim=0,
        )
        batched_lengths = torch.cat([item["lengths"] for item in normalized], dim=0)

        return {
            "normalized": normalized,
            "model_inputs": {
                "input_features": batched_features,
                "feature_attention_mask": batched_mask,
                "audio_feature_lengths": batched_lengths,
            },
        }

    def forward_eager(self, prepared: dict[str, Any]) -> dict[str, Any]:
        return self.model(**prepared["model_inputs"])

    def post(self, prepared: dict[str, Any], combined: dict[str, Any]) -> list[Any]:
        output_lengths = combined["audio_output_lengths"]
        embeds = combined["audio_embeds"]
        row_cursor = 0
        token_cursor = 0
        results: list[Any] = []

        for item in prepared["normalized"]:
            row_end = row_cursor + item["count"]
            req_output_lengths = output_lengths[row_cursor:row_end]
            token_end = token_cursor + int(req_output_lengths.sum().item())
            results.append(
                {
                    "audio_embeds": embeds[token_cursor:token_end],
                    "audio_feature_lengths": combined["audio_feature_lengths"][
                        row_cursor:row_end
                    ],
                    "audio_output_lengths": req_output_lengths,
                }
            )
            row_cursor = row_end
            token_cursor = token_end

        return results


def _split_visual_features(
    tensor: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _split_visual_multiscale(
    tensors: list[torch.Tensor] | None,
    *,
    start: int,
    end: int,
) -> list[torch.Tensor] | None:
    if tensors is None:
        return None
    return [tensor[start:end] for tensor in tensors]


def _grid_visual_tokens(grid: Any, merge: int) -> int:
    if not isinstance(grid, torch.Tensor) or grid.numel() == 0:
        return 0
    return int((grid.to(dtype=torch.long).prod(dim=-1) // merge).sum().item())


def _normalize_audio_request_tensors(
    request: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_dict = request.model_inputs
    features = input_dict["input_features"]
    if features.ndim == 2:
        features = features.unsqueeze(0)

    lengths = input_dict.get("audio_feature_lengths")
    mask = input_dict.get("feature_attention_mask")
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.to(dtype=torch.long).view(-1)
    elif isinstance(mask, torch.Tensor):
        lengths = mask.to(dtype=torch.long).sum(dim=1).view(-1)
    else:
        raise ValueError("audio_feature_lengths or feature_attention_mask is required")

    time_dim = features.shape[-1]
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
    else:
        steps = torch.arange(time_dim, dtype=torch.long, device=lengths.device)
        mask = steps.unsqueeze(0) < lengths.unsqueeze(1)

    return features, mask, lengths


def _pad_audio_features(features: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(features.shape[-1])
    if pad <= 0:
        return features
    return F.pad(features, (0, pad))


def _pad_audio_mask(mask: torch.Tensor, target_time: int) -> torch.Tensor:
    pad = target_time - int(mask.shape[-1])
    if pad <= 0:
        return mask
    return F.pad(mask, (0, pad), value=False)

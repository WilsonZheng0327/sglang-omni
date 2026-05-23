# SPDX-License-Identifier: Apache-2.0
"""Encoder model runners for Qwen3-Omni."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs

from sglang_omni.model_runner.encoder_model_runner import (
    EncoderBatchItem,
    EncoderModelRunner,
    nested_tensor_bytes,
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
QWEN3_VISION_CUDA_GRAPH_TOKEN_BUDGETS = (
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
)
QWEN3_VISION_CUDA_GRAPH_SEQUENCE_BUDGETS = (2, 4, 8, 16, 32, 64, 128, 256)
QWEN3_VISION_CUDA_GRAPH_MAX_SEQUENCE_TOKEN_BUDGETS = (
    256,
    512,
    1024,
    2048,
    4096,
)
QWEN3_VISION_CUDA_GRAPH_MAX_GRAPHS = 8
QWEN3_VISION_CUDA_GRAPH_MAX_BUFFER_BYTES = 3 * 1024**3


@dataclass(frozen=True, slots=True)
class QwenVisionCudaGraphBudget:
    mode: str
    token_budget: int
    sequence_budget: int
    max_sequence_token_budget: int
    dtype: str
    device: str
    attention_impl: str
    exact_cu_seqlens: tuple[int, ...] = ()


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
        cuda_graph_token_budgets: tuple[int, ...] | None = None,
        cuda_graph_sequence_budgets: tuple[int, ...] | None = None,
        cuda_graph_max_sequence_token_budgets: tuple[int, ...] | None = None,
        cuda_graph_max_graphs: int = QWEN3_VISION_CUDA_GRAPH_MAX_GRAPHS,
        cuda_graph_max_buffer_bytes: int = (QWEN3_VISION_CUDA_GRAPH_MAX_BUFFER_BYTES),
    ) -> None:
        super().__init__(
            model=model,
            stage_name=IMAGE_STAGE,
            cache=cache,
            enable_cuda_graph=enable_cuda_graph,
        )
        merge = int(self.model.spatial_merge_size) ** 2
        self.cuda_graph_token_budgets = _sorted_unique_budgets(
            cuda_graph_token_budgets or QWEN3_VISION_CUDA_GRAPH_TOKEN_BUDGETS,
            multiple=merge,
        )
        self.cuda_graph_sequence_budgets = _sorted_unique_budgets(
            cuda_graph_sequence_budgets or QWEN3_VISION_CUDA_GRAPH_SEQUENCE_BUDGETS
        )
        self.cuda_graph_max_sequence_token_budgets = _sorted_unique_budgets(
            cuda_graph_max_sequence_token_budgets
            or QWEN3_VISION_CUDA_GRAPH_MAX_SEQUENCE_TOKEN_BUDGETS,
            multiple=merge,
        )
        self.cuda_graph_max_graphs = int(cuda_graph_max_graphs)
        self.cuda_graph_max_buffer_bytes = int(cuda_graph_max_buffer_bytes)
        self.cuda_graph_fallback_reasons: dict[str, int] = {}

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
        graph_keys: list[tuple[str, QwenVisionCudaGraphBudget]] = []
        image_budget = self._select_visual_graph_budget(
            model_inputs.get("pixel_values"),
            model_inputs.get("image_grid_thw"),
        )
        if image_budget is not None:
            graph_keys.append(("image", image_budget))
        elif isinstance(model_inputs.get("pixel_values"), torch.Tensor):
            return None

        video_budget = self._select_visual_graph_budget(
            model_inputs.get("pixel_values_videos"),
            model_inputs.get("video_grid_thw"),
        )
        if video_budget is not None:
            graph_keys.append(("video", video_budget))
        elif isinstance(model_inputs.get("pixel_values_videos"), torch.Tensor):
            return None

        return tuple(graph_keys) if graph_keys else None

    def forward_cuda_graph(self, prepared: dict[str, Any]) -> dict[str, Any]:
        if not self._prepared_visual_graph_budgets_fit(prepared):
            self.cuda_graph_stats.fallbacks += 1
            return self.forward_eager(prepared)

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
        static_prepared = self._copy_visual_graph_buffers(graph_key, prepared)
        with torch.no_grad(), envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.override(True):
            self._forward_visual_graph_body(
                hidden_states=static_prepared["hidden_states"],
                cu_seqlens=static_prepared["cu_seqlens"],
                position_embeddings=static_prepared["position_embeddings"],
                graph_key=graph_key,
                cu_seqlens_lengths=static_prepared["cu_seqlens_lengths"],
                output_ws=static_prepared["output_ws"],
            )
        torch.cuda.synchronize()
        return static_prepared

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
        with envs.SGLANG_VIT_ENABLE_CUDA_GRAPH.override(True):
            return self._forward_visual_graph_body(
                hidden_states=static_prepared["hidden_states"],
                cu_seqlens=static_prepared["cu_seqlens"],
                position_embeddings=static_prepared["position_embeddings"],
                graph_key=graph_key,
                cu_seqlens_lengths=static_prepared["cu_seqlens_lengths"],
                output_ws=static_prepared["output_ws"],
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
        if self._visual_attention_impl() not in (
            "VisionFlash3Attention",
            "VisionTritonAttention",
        ):
            return False
        try:
            return next(visual.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _prepared_visual_graph_budgets_fit(self, prepared: dict[str, Any]) -> bool:
        graph_key = self.cuda_graph_key(prepared)
        if graph_key is None:
            return False

        new_budgets = {
            budget for _, budget in graph_key if budget not in self.cuda_graphs
        }
        if (
            new_budgets
            and self.cuda_graph_max_graphs > 0
            and len(self.cuda_graphs) + len(new_budgets) > self.cuda_graph_max_graphs
        ):
            self._record_visual_graph_fallback("max_graphs")
            return False

        new_buffer_bytes = sum(
            self._estimate_visual_graph_buffer_bytes(budget) for budget in new_budgets
        )
        if (
            new_buffer_bytes
            and self.cuda_graph_max_buffer_bytes > 0
            and self._visual_graph_reserved_buffer_bytes() + new_buffer_bytes
            > self.cuda_graph_max_buffer_bytes
        ):
            self._record_visual_graph_fallback("max_buffer_bytes")
            return False

        return True

    def _select_visual_graph_budget(
        self,
        pixel_values: Any,
        grid_thw: Any,
    ) -> QwenVisionCudaGraphBudget | None:
        if not isinstance(pixel_values, torch.Tensor) or not isinstance(
            grid_thw,
            torch.Tensor,
        ):
            return None
        if pixel_values.numel() == 0 or grid_thw.numel() == 0:
            return None
        if grid_thw.device.type != "cpu":
            self._record_visual_graph_fallback("grid_not_cpu")
            return None

        grid_thw = grid_thw.to(dtype=torch.long)
        grid_stats = _visual_grid_stats(grid_thw)
        if grid_stats is None:
            return None

        actual_tokens, actual_sequences, actual_max_sequence_tokens = grid_stats
        if int(pixel_values.shape[0]) != actual_tokens:
            self._record_visual_graph_fallback("pixel_grid_mismatch")
            return None
        merge = int(self.model.spatial_merge_size) ** 2
        if actual_tokens % merge != 0:
            self._record_visual_graph_fallback("unaligned_token_count")
            return None

        device = next(self.model.visual.parameters()).device
        dtype = next(self.model.visual.parameters()).dtype
        attention_impl = self._visual_attention_impl()

        for token_budget in self.cuda_graph_token_budgets:
            if token_budget < actual_tokens:
                continue
            for max_sequence_token_budget in self.cuda_graph_max_sequence_token_budgets:
                if max_sequence_token_budget < actual_max_sequence_tokens:
                    continue
                padding_tokens = token_budget - actual_tokens
                if (
                    actual_max_sequence_tokens < max_sequence_token_budget
                    and padding_tokens < max_sequence_token_budget
                ):
                    continue
                padding_sequences = (
                    math.ceil(padding_tokens / max_sequence_token_budget)
                    if padding_tokens > 0
                    else 0
                )
                required_sequences = actual_sequences + padding_sequences
                for sequence_budget in self.cuda_graph_sequence_budgets:
                    if sequence_budget < required_sequences:
                        continue
                    return QwenVisionCudaGraphBudget(
                        mode="budget",
                        token_budget=token_budget,
                        sequence_budget=sequence_budget,
                        max_sequence_token_budget=max_sequence_token_budget,
                        dtype=str(dtype),
                        device=str(device),
                        attention_impl=attention_impl,
                    )

        self._record_visual_graph_fallback("no_fitting_budget")
        return None

    def _build_budgeted_cu_seqlens(
        self,
        *,
        grid_thw: torch.Tensor,
        graph_key: QwenVisionCudaGraphBudget,
        device: torch.device,
    ) -> torch.Tensor:
        if graph_key.mode == "exact":
            cu_seqlens = torch.tensor(
                graph_key.exact_cu_seqlens,
                dtype=torch.int32,
                device="cpu",
            )
            return cu_seqlens.to(device=device, non_blocking=True)

        lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).to(dtype=torch.int32)
        actual_tokens = int(lengths.sum().item())
        actual_sequences = int(lengths.shape[0])

        if actual_tokens > graph_key.token_budget:
            raise RuntimeError(
                f"actual visual tokens {actual_tokens} exceed graph budget "
                f"{graph_key.token_budget}"
            )
        if actual_sequences > graph_key.sequence_budget:
            raise RuntimeError(
                f"actual visual sequences {actual_sequences} exceed graph budget "
                f"{graph_key.sequence_budget}"
            )

        cu_seqlens = torch.empty(
            graph_key.sequence_budget + 1,
            dtype=torch.int32,
            device="cpu",
        )
        cu_seqlens[0] = 0
        if actual_sequences:
            cu_seqlens[1 : actual_sequences + 1] = lengths.cumsum(
                dim=0,
                dtype=torch.int32,
            )

        cursor = actual_tokens
        write_idx = actual_sequences + 1
        while (
            cursor < graph_key.token_budget and write_idx <= graph_key.sequence_budget
        ):
            cursor += min(
                graph_key.max_sequence_token_budget,
                graph_key.token_budget - cursor,
            )
            cu_seqlens[write_idx] = cursor
            write_idx += 1

        if cursor != graph_key.token_budget:
            raise RuntimeError(
                f"visual graph budget cannot pad {actual_tokens} tokens to "
                f"{graph_key.token_budget} with {graph_key.sequence_budget} "
                "cu_seqlens slots"
            )
        if write_idx <= graph_key.sequence_budget:
            cu_seqlens[write_idx:] = graph_key.token_budget

        return cu_seqlens.to(device=device, non_blocking=True)

    def _estimate_visual_graph_buffer_bytes(
        self, budget: QwenVisionCudaGraphBudget
    ) -> int:
        visual = self.model.visual
        visual_config = getattr(visual, "config", None)
        hidden_size = int(
            getattr(visual, "hidden_size", None)
            or getattr(visual_config, "hidden_size")
        )
        num_heads = int(
            getattr(visual, "num_heads", None) or getattr(visual_config, "num_heads")
        )
        head_dim = hidden_size // num_heads
        merge = int(self.model.spatial_merge_size) ** 2
        out_hidden = int(self.model.out_hidden_size)
        output_layers = 1 + int(self.model.deepstack_layers)
        dtype_bytes = int(self.model.visual_dtype_bytes)

        hidden_bytes = budget.token_budget * hidden_size * dtype_bytes
        position_bytes = 2 * budget.token_budget * head_dim * dtype_bytes
        output_bytes = (
            (budget.token_budget // merge) * out_hidden * dtype_bytes * output_layers
        )
        cu_bytes = (budget.sequence_budget + 1) * torch.tensor(
            [],
            dtype=torch.int32,
        ).element_size()
        return hidden_bytes + position_bytes + output_bytes + cu_bytes

    def _visual_graph_reserved_buffer_bytes(self) -> int:
        total = 0
        for store in (
            self.cuda_graph_input_buffers,
            self.cuda_graph_metadata_buffers,
            self.cuda_graph_output_buffers,
        ):
            total += nested_tensor_bytes(store)
        return total

    def _record_visual_graph_fallback(self, reason: str) -> None:
        self.cuda_graph_fallback_reasons[reason] = (
            self.cuda_graph_fallback_reasons.get(reason, 0) + 1
        )

    def _run_visual_cuda_graph(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        prepared = self._prepare_visual_graph_inputs(pixel_values, grid_thw)
        graph_output = self.run_cuda_graph_piece(prepared["graph_key"], prepared)
        output_tokens = int(prepared["output_tokens"])

        # Graph outputs are stable replay buffers. Clone before storing them in
        # stage state/cache so a later replay cannot mutate an older payload.
        return (
            graph_output["embeds"][:output_tokens].clone(),
            [tensor[:output_tokens].clone() for tensor in graph_output["deepstack"]],
        )

    def _prepare_visual_forward_inputs(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> dict[str, Any]:
        visual = self.model.visual
        device = next(visual.parameters()).device
        dtype = next(visual.parameters()).dtype

        grid_thw = grid_thw.to(dtype=torch.long)
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        hidden_states = visual.patch_embed(pixel_values)
        hidden_states = hidden_states + visual.fast_pos_embed_interpolate(grid_thw)
        rotary_pos_emb_cos, rotary_pos_emb_sin = visual.rot_pos_emb(grid_thw.tolist())
        position_embeddings = (
            rotary_pos_emb_cos.contiguous(),
            rotary_pos_emb_sin.contiguous(),
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu_seqlens = cu_seqlens.to(device=device, non_blocking=True)

        return {
            "hidden_states": hidden_states.contiguous(),
            "cu_seqlens": cu_seqlens.contiguous(),
            "position_embeddings": position_embeddings,
        }

    def _prepare_visual_graph_inputs(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> dict[str, Any]:
        visual = self.model.visual
        device = next(visual.parameters()).device
        graph_key = self._select_visual_graph_budget(pixel_values, grid_thw)
        if graph_key is None:
            raise RuntimeError("Qwen vision CUDA graph was requested without a budget")

        visual_inputs = self._prepare_visual_forward_inputs(pixel_values, grid_thw)
        hidden_states = visual_inputs["hidden_states"]
        position_embeddings = visual_inputs["position_embeddings"]
        actual_tokens = int(hidden_states.shape[0])
        output_tokens = actual_tokens // (self.model.spatial_merge_size**2)
        cu_seqlens = self._build_budgeted_cu_seqlens(
            grid_thw=grid_thw.to(dtype=torch.long),
            graph_key=graph_key,
            device=device,
        )
        return {
            "graph_key": graph_key,
            "hidden_states": hidden_states.contiguous(),
            "cu_seqlens": cu_seqlens.contiguous(),
            "position_embeddings": position_embeddings,
            "actual_tokens": actual_tokens,
            "output_tokens": output_tokens,
        }

    def _copy_visual_graph_buffers(
        self,
        graph_key: Any,
        prepared: dict[str, Any],
    ) -> dict[str, Any]:
        hidden_states = prepared["hidden_states"]
        cu_seqlens = prepared["cu_seqlens"]
        position_cos, position_sin = prepared["position_embeddings"]
        actual_tokens = int(prepared["actual_tokens"])

        hidden_buffer = self.static_input_buffer(
            graph_key,
            "hidden_states",
            shape=(graph_key.token_budget, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if actual_tokens < graph_key.token_budget:
            hidden_buffer.zero_()
        hidden_buffer[:actual_tokens].copy_(hidden_states)

        cu_seqlens_buffer = self.static_metadata_buffer(
            graph_key,
            "cu_seqlens",
            shape=(graph_key.sequence_budget + 1,),
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        cu_seqlens_buffer.copy_(cu_seqlens)

        cu_seqlens_lengths = self.static_metadata_buffer(
            graph_key,
            "cu_seqlens_lengths",
            shape=(graph_key.sequence_budget,),
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        cu_seqlens_lengths.copy_(cu_seqlens[1:] - cu_seqlens[:-1])

        cos_buffer = self.static_metadata_buffer(
            graph_key,
            "position_cos",
            shape=(graph_key.token_budget, position_cos.shape[-1]),
            dtype=position_cos.dtype,
            device=position_cos.device,
        )
        sin_buffer = self.static_metadata_buffer(
            graph_key,
            "position_sin",
            shape=(graph_key.token_budget, position_sin.shape[-1]),
            dtype=position_sin.dtype,
            device=position_sin.device,
        )
        if actual_tokens < graph_key.token_budget:
            cos_buffer.zero_()
            sin_buffer.zero_()
        cos_buffer[:actual_tokens].copy_(position_cos)
        sin_buffer[:actual_tokens].copy_(position_sin)

        output_ws = None
        if self._visual_attention_impl() == "VisionTritonAttention":
            first_attn = self.model.visual.blocks[0].attn
            output_ws = self.static_input_buffer(
                graph_key,
                "attention_output_ws",
                shape=(
                    graph_key.token_budget,
                    int(first_attn.num_attention_heads_per_partition),
                    int(first_attn.head_size),
                ),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        return {
            "hidden_states": hidden_buffer,
            "cu_seqlens": cu_seqlens_buffer,
            "cu_seqlens_lengths": cu_seqlens_lengths,
            "position_embeddings": (cos_buffer, sin_buffer),
            "output_ws": output_ws,
        }

    def _forward_visual_graph_body(
        self,
        *,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        graph_key: QwenVisionCudaGraphBudget | None = None,
        cu_seqlens_lengths: torch.Tensor | None = None,
        output_ws: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        visual = self.model.visual
        deepstack_features: list[torch.Tensor] = []
        deepstack_index_by_layer = {
            int(layer_idx): idx
            for idx, layer_idx in enumerate(visual.deepstack_visual_indexes)
        }
        rotary_pos_emb_cos, rotary_pos_emb_sin = position_embeddings
        hidden_states = hidden_states.unsqueeze(1)
        cu_seqlens_arg: Any = cu_seqlens
        if graph_key is not None:
            if graph_key.attention_impl == "VisionFlash3Attention":
                cu_seqlens_arg = [cu_seqlens, graph_key.max_sequence_token_budget]
            elif graph_key.attention_impl == "VisionTritonAttention":
                if cu_seqlens_lengths is None:
                    raise RuntimeError("Triton vision graph requires seqlens lengths")
                cu_seqlens_arg = [
                    cu_seqlens,
                    cu_seqlens_lengths,
                    graph_key.max_sequence_token_budget,
                ]

        for layer_num, block in enumerate(visual.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens_arg,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                output_ws=output_ws,
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
            qkv_backend = getattr(attn, "qkv_backend", None)
            if qkv_backend is not None:
                return type(qkv_backend).__name__
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


def _sorted_unique_budgets(
    values: tuple[int, ...],
    *,
    multiple: int = 1,
) -> tuple[int, ...]:
    budgets = tuple(sorted({int(value) for value in values if int(value) > 0}))
    if not budgets:
        raise ValueError("CUDA graph budget list cannot be empty")
    if multiple > 1 and any(value % multiple != 0 for value in budgets):
        raise ValueError(
            f"CUDA graph budgets {budgets} must be divisible by {multiple}"
        )
    return budgets


def _visual_grid_stats(grid_thw: torch.Tensor) -> tuple[int, int, int] | None:
    if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3 or grid_thw.numel() == 0:
        return None
    grid_thw = grid_thw.to(dtype=torch.long)
    frame_tokens = grid_thw[:, 1] * grid_thw[:, 2]
    actual_tokens = int((grid_thw[:, 0] * frame_tokens).sum().item())
    actual_sequences = int(grid_thw[:, 0].sum().item())
    actual_max_sequence_tokens = int(frame_tokens.max().item())
    if actual_tokens <= 0 or actual_sequences <= 0 or actual_max_sequence_tokens <= 0:
        return None
    return actual_tokens, actual_sequences, actual_max_sequence_tokens


def _exact_visual_cu_seqlens_tuple(grid_thw: torch.Tensor) -> tuple[int, ...]:
    lengths = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).to(dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
    return tuple(int(value) for value in cu_seqlens.tolist())


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

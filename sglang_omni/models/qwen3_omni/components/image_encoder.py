# SPDX-License-Identifier: Apache-2.0
"""Image encoder component for Qwen3-Omni."""

from __future__ import annotations

import logging
import os
import socket
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeVisionEncoderConfig
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeVisionEncoder

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.weight_loader import (
    default_weight_loader,
    load_weights_by_prefix,
    resolve_dtype,
)

logger = logging.getLogger(__name__)

VISUAL_PREFIX = ("thinker.visual.", "visual.")
VISUAL_CLASS = Qwen3OmniMoeVisionEncoder

_SKIPPED_VISUAL_WEIGHT_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rotary_emb.cos_cached",
    "rotary_emb.sin_cached",
    "rotary_pos_emb.inv_freq",
)


@dataclass(frozen=True)
class _SGLangVisionRuntimeState:
    initialized_model_parallel: bool
    attn_tp_size: int | None
    attn_tp_rank: int | None


def _patch_embed_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Optimized PatchEmbed forward using Linear instead of Conv3d."""
    return self.linear(hidden_states.to(dtype=self.linear.weight.dtype))


def _optimize_patch_embed(visual: nn.Module) -> None:
    """Replace Conv3d with Linear in PatchEmbed for ~7-15× speedup.

    The Conv3d kernel does not slide (kernel_size == stride), so it is
    equivalent to a reshape + linear. We load weights via Conv3d for
    checkpoint compatibility, then copy them to a Linear layer.

    Reference: https://github.com/sgl-project/sglang/pull/19788
    """
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None:
        return
    conv = getattr(patch_embed, "proj", None)
    if conv is None or not isinstance(conv, nn.Conv3d):
        return

    if list(conv.kernel_size) != list(conv.stride):
        logger.debug(
            "PatchEmbed Conv3d kernel_size=%s != stride=%s, skipping optimization",
            conv.kernel_size,
            conv.stride,
        )
        return

    if conv.padding != (0, 0, 0) or conv.dilation != (1, 1, 1) or conv.groups != 1:
        logger.debug(
            "PatchEmbed Conv3d has non-trivial padding/dilation/groups, skipping"
        )
        return

    embed_dim = conv.out_channels
    in_features = (
        conv.in_channels
        * conv.kernel_size[0]
        * conv.kernel_size[1]
        * conv.kernel_size[2]
    )

    linear = nn.Linear(
        in_features,
        embed_dim,
        bias=True,
        dtype=conv.weight.dtype,
        device=conv.weight.device,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.view(embed_dim, -1))
        linear.bias.copy_(conv.bias)

    patch_embed.linear = linear
    patch_embed.forward = types.MethodType(_patch_embed_forward, patch_embed)
    logger.info(
        "PatchEmbed optimized: Conv3d(%d→%d) replaced with Linear(%d→%d)",
        conv.in_channels,
        embed_dim,
        in_features,
        embed_dim,
    )


def _sync_visual_graph_runner(visual: nn.Module) -> None:
    graph_runners = getattr(visual, "graph_runners", None)
    if graph_runners is None:
        try:
            from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner

            graph_runners = ViTCudaGraphRunner(visual)
            visual.graph_runners = graph_runners
        except Exception as exc:
            logger.debug("Skipping Qwen3-Omni visual graph runner setup: %s", exc)
            return
    if graph_runners is None:
        return
    if hasattr(visual, "deepstack_visual_indexes"):
        graph_runners._deepstack_visual_indexes = list(visual.deepstack_visual_indexes)
    if hasattr(visual, "deepstack_merger_list"):
        graph_runners._deepstack_merger_list = visual.deepstack_merger_list


def _vision_config_dict(vision_cfg: object) -> dict[str, object]:
    if hasattr(vision_cfg, "to_dict"):
        values = vision_cfg.to_dict()
    else:
        values = {
            key: value
            for key, value in vars(vision_cfg).items()
            if not key.startswith("_")
        }
    for key in ("model_type", "transformers_version", "torch_dtype"):
        values.pop(key, None)
    return values


def _ensure_sglang_vision_runtime(
    model_path: str, *, device: str
) -> _SGLangVisionRuntimeState:
    """Initialize the minimal SGLang runtime needed by VisionAttention."""
    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers import dp_attention as dp
    from sglang.srt.server_args import (
        ServerArgs,
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except ValueError:
        mm_attention_backend = None if torch.device(device).type == "cuda" else "sdpa"
        set_global_server_args_for_scheduler(
            ServerArgs(
                model_path=model_path,
                mm_attention_backend=mm_attention_backend,
            )
        )

    state = _SGLangVisionRuntimeState(
        initialized_model_parallel=False,
        attn_tp_size=getattr(dp, "_ATTN_TP_SIZE", None),
        attn_tp_rank=getattr(dp, "_ATTN_TP_RANK", None),
    )
    dp_tp_ready = (
        getattr(dp, "_ATTN_TP_SIZE", None) is not None and dp._ATTN_TP_SIZE > 0
    )
    if dp_tp_ready and parallel_state.model_parallel_is_initialized():
        return state

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            os.environ["MASTER_PORT"] = str(sock.getsockname()[1])

    if not parallel_state.model_parallel_is_initialized():
        backend = "nccl" if torch.device(device).type == "cuda" else "gloo"
        parallel_state.init_distributed_environment(
            backend=backend,
            world_size=1,
            rank=0,
            local_rank=0,
        )
        parallel_state.initialize_model_parallel()
        state = _SGLangVisionRuntimeState(
            initialized_model_parallel=True,
            attn_tp_size=state.attn_tp_size,
            attn_tp_rank=state.attn_tp_rank,
        )

    dp._ATTN_TP_SIZE = 1
    dp._ATTN_TP_RANK = 0
    return state


def _restore_sglang_vision_runtime(state: _SGLangVisionRuntimeState) -> None:
    if not state.initialized_model_parallel:
        return

    from sglang.srt.distributed import parallel_state
    from sglang.srt.layers import dp_attention as dp

    parallel_state.destroy_model_parallel()
    dp._ATTN_TP_SIZE = state.attn_tp_size
    dp._ATTN_TP_RANK = state.attn_tp_rank


def _detach_sglang_visual_from_runtime(visual: nn.Module) -> None:
    """Remove TP=1 runtime dependencies that are not needed after construction."""
    pos_embed = getattr(visual, "pos_embed", None)
    weight = getattr(pos_embed, "weight", None)
    if not isinstance(weight, torch.nn.Parameter):
        return

    embedding = nn.Embedding(
        weight.shape[0],
        weight.shape[1],
        dtype=weight.dtype,
        device=weight.device,
    )
    visual.pos_embed = embedding


def _mapped_sglang_visual_weight_name(name: str) -> tuple[str, str | None]:
    for weight_name, shard_id in (
        ("attn.q.", "q"),
        ("attn.k.", "k"),
        ("attn.v.", "v"),
    ):
        if weight_name in name:
            return name.replace(weight_name, "attn.qkv_proj."), shard_id
    return name.replace("attn.qkv.", "attn.qkv_proj."), None


def _load_sglang_visual_weights(
    visual: nn.Module,
    weights: dict[str, torch.Tensor],
) -> set[str]:
    params = dict(visual.named_parameters(remove_duplicate=False))
    loaded: set[str] = set()
    skipped: list[str] = []
    unexpected: list[str] = []

    for name, tensor in weights.items():
        mapped_name, shard_id = _mapped_sglang_visual_weight_name(name)
        if mapped_name not in params:
            if name.endswith(_SKIPPED_VISUAL_WEIGHT_SUFFIXES):
                skipped.append(name)
            else:
                unexpected.append(name)
            continue

        param = params[mapped_name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            weight_loader(param, tensor)
        else:
            weight_loader(param, tensor, shard_id)
        loaded.add(mapped_name)

    missing = sorted(set(params) - loaded)
    if unexpected or missing:
        details = []
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if missing:
            details.append(f"missing={missing[:5]}")
        raise RuntimeError(
            "Qwen3-Omni visual weight loading mismatch: " + ", ".join(details)
        )

    if skipped:
        logger.debug(
            "Skipping %d non-parameter Qwen3-Omni visual weights; first skipped=%s",
            len(skipped),
            skipped[0],
        )

    return loaded


def _build_visual(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    vision_cfg = thinker_cfg.vision_config
    runtime_state = _ensure_sglang_vision_runtime(model_path, device=device)
    visual_config = Qwen3OmniMoeVisionEncoderConfig(**_vision_config_dict(vision_cfg))
    try:
        visual = VISUAL_CLASS(visual_config)
        visual.config = visual_config
        _detach_sglang_visual_from_runtime(visual)
    finally:
        _restore_sglang_vision_runtime(runtime_state)
    state_dict = load_weights_by_prefix(
        model_path,
        prefix=VISUAL_PREFIX,
    )
    loaded = _load_sglang_visual_weights(visual, state_dict)
    if not loaded:
        raise RuntimeError("No Qwen3-Omni visual weights were loaded")
    visual.to(device=device, dtype=torch_dtype)
    visual.eval()
    _optimize_patch_embed(visual)
    _sync_visual_graph_runner(visual)
    logger.info(
        "Loaded Qwen3-Omni visual encoder with SGLang VisionAttention backend "
        "(%d tensors)",
        len(loaded),
    )
    return visual


class Qwen3OmniImageEncoder(nn.Module):
    """Vision tower extracted from the HF thinker."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        vision_cfg = thinker_cfg.vision_config
        self._device = torch.device(device)
        self.visual = _build_visual(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        self.spatial_merge_size = int(vision_cfg.spatial_merge_size)
        self.out_hidden_size = int(vision_cfg.out_hidden_size)
        self.deepstack_layers = len(vision_cfg.deepstack_visual_indexes)
        self.visual_dtype_bytes = torch.empty(
            (), dtype=self.visual.dtype
        ).element_size()
        self.eval()

    def _split_visual_output(
        self,
        visual_output: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if isinstance(visual_output, tuple):
            return visual_output

        hidden = self.out_hidden_size
        expected_hidden = hidden * (1 + self.deepstack_layers)
        if visual_output.shape[-1] != expected_hidden:
            raise RuntimeError(
                "Unexpected Qwen3-Omni visual output hidden size: "
                f"got {visual_output.shape[-1]}, expected {expected_hidden}"
            )

        embeds = visual_output[:, :hidden]
        deepstack = [
            visual_output[:, hidden * (idx + 1) : hidden * (idx + 2)]
            for idx in range(self.deepstack_layers)
        ]
        return embeds, deepstack

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        merge = self.spatial_merge_size**2

        if isinstance(pixel_values, torch.Tensor) and isinstance(
            image_grid_thw, torch.Tensor
        ):
            image_grid_thw = image_grid_thw.to(self._device, dtype=torch.long)
            pixel_values = pixel_values.to(device=self._device, dtype=self.visual.dtype)
            image_embeds, image_embeds_multiscale = self._split_visual_output(
                self.visual(pixel_values, grid_thw=image_grid_thw)
            )
            image_token_counts = image_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                    "image_token_counts": image_token_counts.to(device=self._device),
                    "deepstack_visual_embeds_image": image_embeds_multiscale,
                }
            )

        if isinstance(pixel_values_videos, torch.Tensor) and isinstance(
            video_grid_thw, torch.Tensor
        ):
            video_grid_thw = video_grid_thw.to(self._device, dtype=torch.long)
            pixel_values_videos = pixel_values_videos.to(
                device=self._device, dtype=self.visual.dtype
            )
            video_embeds, video_embeds_multiscale = self._split_visual_output(
                self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            )
            video_token_counts = video_grid_thw.prod(-1) // merge
            outputs.update(
                {
                    "video_embeds": video_embeds,
                    "video_grid_thw": video_grid_thw,
                    "video_token_counts": video_token_counts.to(device=self._device),
                    "deepstack_visual_embeds_video": video_embeds_multiscale,
                }
            )

        return outputs

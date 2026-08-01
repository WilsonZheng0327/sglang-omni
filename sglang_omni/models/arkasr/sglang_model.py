# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR-3B: Whisper(RoPE) audio tower + MLP adapter + dense Qwen2 LM.

Audio embeddings are scattered into ``<|audio|>`` placeholder positions via
``general_mm_embed_routine``, matching the qwen3_asr / higgs_audio_asr pattern.
"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix

from .audio_lengths import arkasr_num_audio_tokens
from .audio_tower import ArkAudioMLPAdapter
from .configuration_arkasr import ArkasrConfig

logger = logging.getLogger(__name__)


class ArkasrForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: ArkasrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_token_id = int(getattr(config, "audio_token_id", 151663))

        # audio_encoder = whisper tower + MLP frame-merge adapter (checkpoint name)
        self.audio_encoder = ArkAudioMLPAdapter(config)
        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode a batch of requests' mel features to LLM-space embeddings.

        Each item.feature is (num_mel_bins, T) or (1, num_mel_bins, T) with a
        per-request T. SGLang hands every cache miss in a forward batch to one
        call, so mixed-length mels are padded to the batch max and encoded in a
        single tower pass, with the padding masked out of attention and zeroed
        before the frame merge. Each item's rows are then trimmed back to its own
        token count and concatenated, so the flat sequence lines up with the
        scattered <|audio|> positions exactly as the per-item loop did.

        Zero padding is what makes the batched result match: conv1/conv2 pad with
        zeros implicitly, so a zero-padded frame at the boundary contributes the
        same value it would have when the clip was encoded alone.
        """
        device = next(self.audio_encoder.parameters()).device
        dtype = self.audio_encoder.dtype
        merge_factor = int(self.audio_encoder.merge_factor)

        features: list[torch.Tensor] = []
        mel_lengths: list[int] = []
        for item in items:
            feat = item.feature
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)  # (1, mel, T)
            features.append(feat.to(device=device, dtype=dtype))

            # The request builder already ships this; it is the same count it
            # derived num_audio_tokens from, so the scatter contract holds.
            mask = getattr(item, "feature_attention_mask", None)
            if mask is None:
                mel_lengths.append(int(feat.shape[-1]))
                continue
            if int(mask.shape[-1]) != int(feat.shape[-1]):
                raise ValueError(
                    f"ARK-ASR feature_attention_mask width {int(mask.shape[-1])} "
                    f"does not match the item's {int(feat.shape[-1])} mel frames"
                )
            mel_lengths.append(int(mask.sum().item()))

        max_frames = max(feature.shape[-1] for feature in features)
        if any(feature.shape[-1] != max_frames for feature in features):
            features = [
                nn.functional.pad(feature, (0, max_frames - feature.shape[-1]))
                for feature in features
            ]
        batch = torch.cat(features, dim=0)
        lengths = torch.tensor(mel_lengths, dtype=torch.long, device=device)

        embeddings = self.audio_encoder(batch, mel_lengths=lengths)  # (B, S, H)
        outs = [
            embeddings[index, : arkasr_num_audio_tokens(length, merge_factor)]
            for index, length in enumerate(mel_lengths)
        ]
        return torch.cat(outs, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        tie = bool(getattr(self.config, "tie_word_embeddings", False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if tie and "lm_head.weight" in name:
                continue

            # checkpoint layout:
            #   audio_encoder.whisper.*   audio_encoder.layer_norm.*  audio_encoder.adapting.*
            #   model.*  (Qwen2 decoder)  lm_head.*
            is_audio = name.startswith("audio_encoder.")
            if not is_audio:
                if name.startswith("model."):
                    name = "language_model." + name
                elif name.startswith("lm_head."):
                    name = "language_model." + name

            if is_audio:
                # audio tower params load directly (no qkv stacking: q/k/v are separate
                # Linear layers in WhisperRoPESdpaAttention, matching the checkpoint)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("arkasr: skip unmatched audio weight %s", name)
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )
                continue

            for param_name, weight_name, shard_id in llm_stacked_params:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("arkasr: skip unmatched llm weight %s", name)
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )


EntryClass = ArkasrForConditionalGeneration

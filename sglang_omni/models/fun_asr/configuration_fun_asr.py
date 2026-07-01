# SPDX-License-Identifier: Apache-2.0
# Author:
# PoTaTo-Mika: https://github.com/PoTaTo-Mika

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.utils import logger
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.audio_utils import mel_filter_bank, window_function
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length


AUDIO_PLACEHOLDER_TOKEN = "<|object_ref_start|>"


class FunAsrNanoFeatureExtractor(SequenceFeatureExtractor):
    """80-mel log-mel fbank + LFR stacking, matching Fun-ASR's WavFrontend.

    Output ``input_features`` shape is ``[batch, lfr_m * n_mels, T_lfr]`` =
    ``[batch, 560, T_lfr]`` where ``T_lfr = ceil(T_mel / lfr_n)``. The encoder's
    ``input_size`` is 560 (= 7 * 80). ``attention_mask`` tracks valid LFR
    frames; its per-row sum is the post-LFR frame count fed to
    :func:`fun_asr_low_frame_rate_length`.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        lfr_m: int = 7,
        lfr_n: int = 6,
        window: str = "hamming",
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.feature_size = feature_size
        self.n_mels = feature_size
        self.sampling_rate = sampling_rate
        # frame_length/shift in ms (for torchaudio.compliance.kaldi.fbank)
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        # transformers expresses window length/shift in samples at sampling_rate
        self.n_fft = int(round(frame_length * sampling_rate / 1000))  # 400 @ 16k
        self.hop_length = int(round(frame_shift * sampling_rate / 1000))  # 160 @ 16k
        self.win_length = self.n_fft
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.window = window
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

        # Precompute the mel filterbank (n_mels x (n_fft//2 + 1)).
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=self.n_mels,
            min_frequency=0.0,
            max_frequency=float(self.sampling_rate / 2),
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="htk",
        ).astype(np.float32)

        if window == "hamming":
            self._window = window_function(self.win_length, name="hamming", periodic=False).astype(
                np.float32
            )
        else:
            raise ValueError(f"Unsupported window: {window!r} (Fun-ASR uses hamming)")

    @property
    def nb_max_frames(self) -> int:
        """Max LFR frames for a 30s clip — used for context_length sizing."""
        max_mel = int(round(30.0 * self.sampling_rate / self.hop_length))
        return (max_mel + self.lfr_n - 1) // self.lfr_n

    def _extract_fbank(self, waveform: np.ndarray) -> tuple[torch.Tensor, int]:
        """Compute 80-mel log-mel fbank via Kaldi compliance (matches funasr WavFrontend).

        Mirrors ``funasr.frontends.wav_frontend.WavFrontend.forward``:
        ``waveform * (1 << 15)`` (int16 scale) → ``torchaudio.compliance.kaldi.fbank``
        (hamming window, energy_floor=0, dither=0, snip_edges=True).
        Returns ``(fbank, T_mel)`` where ``fbank`` is ``[T_mel, n_mels]``.
        """
        import torchaudio.compliance.kaldi as kaldi

        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # WavFrontend.forward: waveform * (1 << 15) — scale to int16 range.
        # Without this, log-mel values are ~21 lower (2*log(32768)) and the
        # encoder/adaptor produce embeddings the LLM cannot decode (→ /sil).
        wav_t = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
        # Cap frame_length for very short audio (funasr WavFrontend does this).
        frame_length = min(
            self.frame_length, wav.shape[0] / self.sampling_rate * 1000
        )
        mat = kaldi.fbank(
            wav_t,
            num_mel_bins=self.n_mels,
            frame_length=frame_length,
            frame_shift=self.frame_shift,
            dither=0.0,
            energy_floor=0.0,
            window_type=self.window,
            sample_frequency=self.sampling_rate,
            snip_edges=True,
        )  # [T_mel, n_mels]
        return mat, mat.shape[0]

    def _lfr(self, fbank: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Low frame rate stacking (matches funasr ``apply_lfr``).

        Stacks ``lfr_m`` frames every ``lfr_n`` stride: left-pad by repeating
        the first frame ``(lfr_m-1)//2`` times, right-pad the last frame to
        fill the final window, then gather via ``as_strided``.
        Returns ``(lfr_out, T_lfr)`` where ``lfr_out`` is ``[T_lfr, lfr_m*n_mels]``.
        """
        t_mel = fbank.shape[0]
        t_lfr = int(np.ceil(t_mel / self.lfr_n))
        pad_left = (self.lfr_m - 1) // 2
        left_padding = fbank[0:1].repeat(pad_left, 1)
        inputs = torch.vstack([left_padding, fbank])
        t_padded = inputs.shape[0]
        feat_dim = inputs.shape[-1]
        strides = (self.lfr_n * feat_dim, 1)
        sizes = (t_lfr, self.lfr_m * feat_dim)
        last_idx = (t_padded - self.lfr_m) // self.lfr_n + 1
        num_padding = self.lfr_m - (t_padded - last_idx * self.lfr_n)
        if num_padding > 0:
            num_padding = (
                (2 * self.lfr_m - 2 * t_padded + (t_lfr - 1 + last_idx) * self.lfr_n)
                / 2
                * (t_lfr - last_idx)
            )
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
        out = inputs.as_strided(sizes, strides)
        return out.clone().to(torch.float32), t_lfr

    def __call__(
        self,
        audio,
        sampling_rate: int | None = None,
        return_tensors=None,
        return_attention_mask: bool | None = None,
        padding: str | bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ):
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            waveforms = [audio]
        elif isinstance(audio, (list, tuple)) and (
            not audio or isinstance(audio[0], (int, float, np.floating))
        ):
            waveforms = [np.asarray(audio, dtype=np.float32)]
        elif isinstance(audio, np.ndarray) and audio.ndim == 2:
            waveforms = [audio[i] for i in range(audio.shape[0])]
        else:
            waveforms = list(audio)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"FunAsrNanoFeatureExtractor: sampling_rate {sampling_rate} != "
                f"{self.sampling_rate}; resampling is the caller's responsibility."
            )

        feats, masks = [], []
        for wav in waveforms:
            fbank, t_mel = self._extract_fbank(wav)
            lfr_feat, t_lfr = self._lfr(fbank)  # [t_lfr, lfr_m*n_mels=560]
            # Transpose to [lfr_m * n_mels, t_lfr] = [560, t_lfr] (encoder expects [B, T, 560])
            lfr_feat = lfr_feat.t().contiguous()
            feats.append(lfr_feat)
            masks.append([1] * t_lfr)

        if padding == "max_length":
            max_t = self.nb_max_frames
        elif padding == "longest" or padding is True:
            max_t = max(f.shape[1] for f in feats)
        else:
            max_t = max(f.shape[1] for f in feats)

        n_feat = self.lfr_m * self.n_mels
        batched = np.full(
            (len(feats), n_feat, max_t), self.padding_value, dtype=np.float32
        )
        attention = np.zeros((len(feats), max_t), dtype=np.int64)
        for i, f in enumerate(feats):
            t = f.shape[1]
            batched[i, :, :t] = f
            attention[i, :t] = masks[i]

        return_attention_mask = (
            self.return_attention_mask if return_attention_mask is None else return_attention_mask
        )
        out = {"input_features": batched}
        if return_attention_mask:
            out["attention_mask"] = attention
        if return_tensors == "pt":
            out["input_features"] = torch.from_numpy(out["input_features"])
            out["attention_mask"] = torch.from_numpy(out["attention_mask"])
        return out


# ---------------------------------------------------------------------------
# Processor — feature extractor + tokenizer + placeholder expansion.
# ---------------------------------------------------------------------------


class FunAsrNanoProcessor:
    """Composite processor: FunAsrNanoFeatureExtractor + Qwen2Tokenizer.

    Mirrors ``Qwen3ASRProcessor``. AutoProcessor.from_pretrained for the HF
    Fun-ASR checkpoint expects remote code (``processing_fun_asr_nano.py``)
    that is not bundled, so sglang-omni provides this processor directly and
    registers it via ``register_customized_processor``.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = FunAsrNanoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def _get_feat_extract_output_lengths(self, input_lengths):
        """LFR frames -> adaptor audio-token count (3x stride-2)."""
        return fun_asr_low_frame_rate_length(input_lengths)

    def __call__(self, text=None, audio=None, audio_kwargs=None, **kwargs):
        inputs: dict[str, Any] = {}
        if audio is not None:
            audio_kwargs = audio_kwargs or {}
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors=kwargs.get("return_tensors"),
                return_attention_mask=True,
                **audio_kwargs,
            )
            inputs["input_features"] = audio_inputs["input_features"]
            if "attention_mask" in audio_inputs:
                inputs["feature_attention_mask"] = audio_inputs["attention_mask"]

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
            )
            input_ids = text_inputs["input_ids"]

            # Expand the single <|object_ref_start|> placeholder in the prompt
            # to N copies, where N is the adaptor's audio-token count for this
            # clip. Without this, the model sees only 1 audio token for hundreds
            # of LFR frames and cannot align audio embeddings with positions.
            if audio is not None and "feature_attention_mask" in inputs:
                audio_pad_id = self.tokenizer.convert_tokens_to_ids(
                    AUDIO_PLACEHOLDER_TOKEN
                )
                feat_lengths = inputs["feature_attention_mask"].sum(dim=-1)
                audio_token_counts = self._get_feat_extract_output_lengths(
                    feat_lengths
                )
                expanded = []
                for seq_idx in range(input_ids.shape[0]):
                    ids = (
                        input_ids[seq_idx].tolist()
                        if hasattr(input_ids[seq_idx], "tolist")
                        else list(input_ids[seq_idx])
                    )
                    audio_idx = 0
                    new_ids = []
                    for tid in ids:
                        if tid == audio_pad_id and audio_idx < len(
                            audio_token_counts
                        ):
                            n = int(audio_token_counts[audio_idx].item())
                            new_ids.extend([audio_pad_id] * n)
                            audio_idx += 1
                        else:
                            new_ids.append(tid)
                    expanded.append(new_ids)
                max_len = max(len(s) for s in expanded)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = [s + [pad_id] * (max_len - len(s)) for s in expanded]
                input_ids = torch.tensor(padded, dtype=torch.long)

            inputs["input_ids"] = input_ids
        return inputs


# ---------------------------------------------------------------------------
# Config classes
# ---------------------------------------------------------------------------


class FunAsrNanoEncoderConfig(PretrainedConfig):
    """SenseVoice-style SANM Conformer encoder hyperparameters.

    Mirrors ``config.json`` ``audio_encoder_config``. The encoder itself
    (``model.audio_encoder.*`` weights: encoders0/encoders/tp_encoders SANM
    blocks + after_norm/tp_norm) is implemented in ``sglang_model.py``.
    """

    model_type = "fun_asr_nano_encoder"

    def __init__(
        self,
        input_size: int = 560,
        output_size: int = 512,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 50,
        tp_blocks: int = 20,
        kernel_size: int = 11,
        sanm_shift: int = 0,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.attention_heads = attention_heads
        self.linear_units = linear_units
        self.num_blocks = num_blocks
        self.tp_blocks = tp_blocks
        self.kernel_size = kernel_size
        self.sanm_shift = sanm_shift
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.positional_dropout_rate = positional_dropout_rate
        self.initializer_range = initializer_range


class FunAsrNanoAdaptorConfig(PretrainedConfig):
    """Low-frame-rate adaptor hyperparameters (``config.json`` top-level ``adaptor_*``)."""

    model_type = "fun_asr_nano_adaptor"

    def __init__(
        self,
        encoder_dim: int = 512,
        llm_dim: int = 1024,
        ffn_dim: int = 2048,
        num_layers: int = 2,
        attention_heads: int = 8,
        downsample_rate: int = 1,
        dropout_rate: float = 0.0,
        use_low_frame_rate: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.downsample_rate = downsample_rate
        self.dropout_rate = dropout_rate
        self.use_low_frame_rate = use_low_frame_rate


@register_customized_processor(FunAsrNanoProcessor)
class FunAsrNanoConfig(PretrainedConfig):
    """Top-level config for ``FunAsrNanoForConditionalGeneration``.

    Matches the HF-adapted ``config.json`` shipped with Fun-ASR-Nano-2512:
    ``audio_encoder_config`` (sub-config) + ``text_config`` (Qwen3) + top-level
    ``adaptor_*`` fields + ``audio_token_index``. Note the HF checkpoint has NO
    ``thinker_config`` wrapper (unlike Qwen3-ASR) — encoder/adaptor/text are
    siblings under the top-level config.
    """

    model_type = "fun_asr_nano"
    sub_configs: ClassVar[dict[str, Any]] = {
        "audio_encoder_config": FunAsrNanoEncoderConfig,
    }

    def __init__(
        self,
        audio_encoder_config=None,
        text_config=None,
        audio_token_index: int = 151646,
        # Adaptor fields (top-level in config.json, prefixed adaptor_*)
        adaptor_encoder_dim: int = 512,
        adaptor_llm_dim: int = 1024,
        adaptor_ffn_dim: int = 2048,
        adaptor_num_layers: int = 2,
        adaptor_attention_heads: int = 8,
        adaptor_downsample_rate: int = 1,
        adaptor_dropout_rate: float = 0.0,
        use_low_frame_rate: bool = True,
        **kwargs,
    ):
        # Resolve sub-configs BEFORE super().__init__: transformers 5.6 runs
        # validate_token_ids inside __init__, which calls get_text_config() —
        # so self.text_config must already exist when super() executes.
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config = FunAsrNanoEncoderConfig(**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = FunAsrNanoEncoderConfig()
        self.audio_encoder_config = audio_encoder_config

        from transformers.models.qwen3.configuration_qwen3 import (
            Qwen3Config as HFQwen3Config,
        )

        if isinstance(text_config, dict):
            text_config = HFQwen3Config(**text_config)
        elif text_config is None:
            text_config = HFQwen3Config()
        self.text_config = text_config

        super().__init__(**kwargs)

        # Adaptor config carried as a nested object for sglang_model.py.
        self.adaptor_config = FunAsrNanoAdaptorConfig(
            encoder_dim=adaptor_encoder_dim or self.audio_encoder_config.output_size,
            llm_dim=adaptor_llm_dim or getattr(text_config, "hidden_size", 1024),
            ffn_dim=adaptor_ffn_dim,
            num_layers=adaptor_num_layers,
            attention_heads=adaptor_attention_heads,
            downsample_rate=adaptor_downsample_rate,
            dropout_rate=adaptor_dropout_rate,
            use_low_frame_rate=use_low_frame_rate,
        )

        # Audio placeholder token id (<|object_ref_start|>).
        self.audio_token_index = audio_token_index

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        # Called by transformers during super().__init__ validation and by
        # sglang for context-length sizing. Guard against the brief window
        # before self.text_config is assigned (should not trigger after the
        # reorder above, but kept defensive).
        text_config = getattr(self, "text_config", None)
        if text_config is None:
            return self
        return text_config


AutoConfig.register("fun_asr_nano", FunAsrNanoConfig)
AutoConfig.register("fun_asr_nano_encoder", FunAsrNanoEncoderConfig)
AutoConfig.register("fun_asr_nano_adaptor", FunAsrNanoAdaptorConfig)

# FunAsrNanoFeatureExtractor is a custom class (not a transformers built-in),
# so AutoFeatureExtractor.from_pretrained — which stages.py uses to load the
# feature extractor from preprocessor_config.json (feature_extractor_type=
# "FunAsrNanoFeatureExtractor") — cannot resolve it without this registration.
# The checkpoint ships no remote-code feature extractor (processor_config.json
# only auto-maps AutoProcessor), so we register it ourselves.
AutoFeatureExtractor.register("FunAsrNanoFeatureExtractor", FunAsrNanoFeatureExtractor)

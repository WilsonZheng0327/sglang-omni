# Fun-ASR-Nano × sglang-omni 集成进度

目标:把 Fun-ASR-Nano-2512(Tongyi Lab,2025 年底,~800M)接入 sglang-omni serving 框架,结构镜像已有的 `qwen3_asr`。权重 checkpoint 在 `checkpoints/Fun-ASR-Nano-2512/`(HF 适配版)。

参考实现来源(均为 ground truth):
- 官方 funasr 1.3.14 wheel(已解压到 `/tmp/funasr_src/funasr_extracted/`,可 import 对比):
  `funasr/models/sense_voice/model.py`(`SenseVoiceEncoderSmall`、`MultiHeadedAttentionSANM`、`EncoderLayerSANM`、`SinusoidalPositionEncoder`、`PositionwiseFeedForward`)
  `funasr/models/llm_asr/adaptor.py`(`Transformer` adaptor)
  `funasr/models/fun_asr_nano/model.py` + `inference_vllm.py`(`use_low_frame_rate` 截断逻辑)
- C++ 端口 `Fun-ASR/runtime/llama.cpp/funasr-cli/funasr-cli.cpp`(`run_encoder`、`sanm_attn`、`sanm_layer`、`adp_layer`)
- HF checkpoint `checkpoints/Fun-ASR-Nano-2512/`(`config.json`、`model.safetensors`、`chat_template.jinja`、`tokenizer_config.json`)

---

## 文件清单与状态

```
sglang_omni/models/fun_asr/
├── __init__.py                  ✅ from . import config  (让 registry 扫到 EntryClass)
├── config.py                    ✅ FunASRPipelineConfig, architecture="FunAsrNanoForConditionalGeneration"
├── configuration_fun_asr.py     ✅ FunAsrNanoConfig + FunAsrNanoProcessor + FunAsrNanoFeatureExtractor
├── tool_funcs/
│   ├── __init__.py              ✅
│   └── audio_lengths.py         ✅ fun_asr_low_frame_rate_length 等,bit-exact 验证
├── request_builders.py          ✅ make_fun_asr_scheduler_adapters + FunASRRequestData
├── sglang_model.py              ✅ FunAsrNanoForConditionalGeneration(本轮完成)
├── stages.py                    ✅ create_sglang_fun_asr_executor(本轮完成,GPU 端到端验证通过)
└── payload_types.py             ⚠️ 空,qwen3_asr 也无,可省
```

---

## 已完成验证(本轮 stages.py)

### stages.py — `create_sglang_fun_asr_executor`
镜像 `qwen3_asr/stages.py`,差异点:
- `feature_extractor = AutoFeatureExtractor.from_pretrained(...)` → 我们的 `FunAsrNanoFeatureExtractor`。需在 `configuration_fun_asr.py` 末尾 `AutoFeatureExtractor.register("FunAsrNanoFeatureExtractor", FunAsrNanoFeatureExtractor)`(该类非 transformers 内置,checkpoint 的 `preprocessor_config.json` 只声明 `feature_extractor_type` 无 auto_map)。
- `encoder_token_count = fun_asr_low_frame_rate_length(feature_extractor.nb_max_frames)` = **63**(30s 上限;`nb_max_frames`=500 是 post-LFR 帧数,只剩 adaptor 三次 stride-2)。`context_length = 63 + max_new_tokens + 8`。
- `model_arch_override="FunAsrNanoForConditionalGeneration"`。
- 其余(`disable_cuda_graph` 临时关闭→init_device_graphs、`mm_attention_backend` sm≥100 用 triton_attn、`mem_fraction_static`、`disable_overlap_schedule`、`sampling_backend=pytorch`、`init_mm_embedding_cache`)与 qwen3_asr 一致。

### 三处配套改动(让 sglang 发现并加载我们的模型)
1. **`sglang_omni/model_runner/sglang_model_runner.py`** `_register_omni_model` 的 `sglang_omni_models` 字典加 `"FunAsrNanoForConditionalGeneration": "sglang_omni.models.fun_asr.sglang_model:FunAsrNanoForConditionalGeneration"`(sglang 0.5.12 内置 registry 无此 arch,需 omni 注册;`_register_omni_model` 在 `SGLModelRunner.__init__` 内调用)。
2. **`sglang_omni/model_runner/model_worker.py`** `_ARCH_CONFIG_MAP` 加 `"FunAsrNanoForConditionalGeneration": ("text_config", None)`。Fun-ASR 的 HF config 无 `thinker_config` 包装(Qwen3-ASR 是 `("thinker_config","text_config")`),text_config 是顶层 sibling,故 `text_config_attr=None` → `hf_text_config = hf_config.text_config`(Qwen3Config,hidden=1024,layers=28)。
3. **`configuration_fun_asr.py`** `AutoFeatureExtractor.register`(见上)。

### GPU 端到端 smoke test(RTX 4080, sm_89)✅
`create_sglang_fun_asr_executor(CKPT, dtype=bfloat16, max_running_requests=4, max_new_tokens=256, mem_fraction_static=0.45)` 成功:
- 权重加载(1 shard)、CUDA graph 捕获(bs=4/2/1)均通过。
- `OmniScheduler` 构建;`context_length = 63 + max_new_tokens + 128`(prompt overhead 从 +8 改为 +128,Fun-ASR 的 ChatML prompt 23-94 tokens,远大于 qwen3_asr 的 +8 余量)。
- `model_config.hf_config.architectures == ['FunAsrNanoForConditionalGeneration']`,`hf_text_config` = Qwen3Config(hidden=1024, layers=28)→ `_ARCH_CONFIG_MAP` override 生效。
- 模型类 `FunAsrNanoForConditionalGeneration`,`audio_encoder`/`audio_adaptor`/`language_model` 子模块齐全。
- **dtype=bfloat16(不是 float16)**:adaptor 输出 std≈29.6(max≈650),fp16 的 ~65504 范围在 Qwen3 attention/MLP 里溢出成 NaN,输出退化成 `!!!!!`。bf16 与 fp32 同指数范围,稳定。官方 `demo_vllm.py` 也默认 `--dtype bf16`。
- 注:`Failed to load generation_config.json` 警告无害(checkpoint 未 shipped generation_config.json)。

### 端到端 ASR 推理 ✅(本轮修复)
`test_data/*.opus` 3 条音频(24s/16s/48s 中文)转写结果与参考 `.txt` 高度一致(标点/个别字差异)。latency 0.9-1.0s。

**关键 bug 修复:`FunAsrNanoFeatureExtractor` 特征提取**
- **原 bug**:用自写 numpy fbank(`np.fft.rfft` + `mel_filter_bank`),且**未做 `waveform * (1 << 15)` int16 缩放**。funasr `WavFrontend.forward`(`funasr/frontends/wav_frontend.py:167`)做 `waveform = waveform * (1 << 15)` 后再 `torchaudio.compliance.kaldi.fbank`。缺这步缩放 → log-mel 值低 ~21(=2*log(32768))→ encoder/adaptor 输出错误 → LLM 输出 `/sil`(静音 token)。
- **修复**:`_extract_fbank` 改用 `torchaudio.compliance.kaldi.fbank`(hamming, energy_floor=0, dither=0, snip_edges=True)+ `waveform * (1 << 15)`。`_lfr` 改用 funasr `apply_lfr` 的 `as_strided` 实现。修复后 fbank 与 kaldi bit-exact(max_diff=0.000000)。
- **验证方法**:对照 transformers `Qwen3ForCausalLM`(从 checkpoint 加载 LLM 权重)喂相同 prompt+audio embeddings,sglang 与 transformers 的 first-token logits **完全一致**(top5: [2687, 40, 1782, 99530, 785])→ sglang Qwen3 forward 正确,问题在特征提取。修复后 top token 变为正确中文字。
- **已排除的非根因**:prompt format(bit-identical 官方 chat_template)、audio splicing(max|LLM_input - get_audio_feature|=0.0)、positions(1D [0..N],model_is_mrope=False 因 rope_type=default 无 mrope_section)、config(rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True, q_norm/k_norm 已加载)、weights(qkv_proj 堆叠正确,0 missing)、`<|object_ref_end|>` end marker(加不加都一样)。

### 单元测试 ✅
`tests/unit_test/fun_asr/test_pipeline.py`(5)+ `test_request_builders.py`(3)共 8 项全过;qwen3_asr 8 项回归不破。

---

## 已完成验证(上一轮 sglang_model.py)

### 1. 权重加载 — 100% 匹配
- encoder: **914/914** 参数,strict load 成功,0 missing 0 extra 0 shape mismatch。
- adaptor: **36/36** 参数,strict load 成功。
- 与官方 funasr 类的参数名集合完全相同(`ref_enc_keys == my_enc_keys`、`ref_adp_keys == my_adp_keys`)。
- LLM(Qwen3ForCausalLM)权重映射 `model.language_model.*` → `language_model.model.*`,q/k/v→`qkv_proj` 堆叠、gate/up→`gate_up_proj` 堆叠,与 qwen3_asr 同构(已被 qwen3_asr 验证)。`lm_head.weight` 因 `tie_word_embeddings=true` 跳过。

### 2. Encoder forward — bit-exact
对照官方 `SenseVoiceEncoderSmall`,相同输入(T_lfr=17/83/167)输出 `max_abs_diff = 0.000e+00`,逐阶段(scale+PE、encoders0、encoders×49、after_norm、tp_encoders×20、tp_norm)全部 0.000e+00。

### 3. Adaptor forward — ~1e-3 差异(待确认是否纯 float32 累积误差)
encoder bit-exact 后接 adaptor,`max_abs_diff ≈ 3e-3 ~ 6e-3`。差异来源**尚未定位**,最可能是:
- funasr `Transformer` adaptor 的 `EncoderLayer` 用标准 `MultiHeadedAttention`(query/key/value 分立),其 `forward_attention` 对 `mask` 做了 `masked_fill(-inf)` 处理(即使全 valid 也会走 mask 分支,与我的无 mask 分支在 softmax 数值路径上略有不同);
- 或 funasr `forward_qkv` 的 reshape/transpose 顺序与我的实现存在细微差异。

**这个差异量级(1e-3)在 bf16 推理下大概率可忽略,但建议定位清楚再上线。** 详见下方「待办:精度对比」。

### 4. 无法在纯 CPU 上测的:完整模型 forward
`FunAsrNanoForConditionalGeneration.__init__` 内部构造 `Qwen3ForCausalLM`,而 sglang 的 Qwen3ForCausalLM 需要完整 runtime(dp_attention、pipeline parallel group、global server args)。这些由 `stages.py` 的 `create_sglang_infrastructure` 初始化。所以**完整模型(含 LLM)的构建+加载只能在 stages.py 完成后、通过 sglang server 启动路径验证**。

测试时需要的 sglang runtime 初始化样板(供 stages.py 或测试脚本参考):
```python
import os
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29512")
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel, model_parallel_is_initialized,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
if not model_parallel_is_initialized():
    init_distributed_environment(world_size=1, rank=0)
    initialize_model_parallel(1, 1)
# 必须先 import configuration_fun_asr 让 AutoConfig 注册 fun_asr_nano,再建 ServerArgs
from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoConfig  # noqa
set_global_server_args_for_scheduler(ServerArgs(model_path=CKPT, trust_remote_code=True))
```

---

## sglang_model.py 架构要点(已严格对齐 funasr)

### Audio encoder = SenseVoiceEncoderSmall(SANM Conformer)
- `SinusoidalPositionEncoder`:depth=input_size=560,positions 1..T,**不加 sqrt 缩放**(sqrt 由 encoder forward 在 PE 前做)。
- encoder forward:`xs *= output_size**0.5`(sqrt(512))→ PE → encoders0[1](560→512,**attn 无残差**因 in_size≠size,ffn 有残差)→ encoders[49](512→512)→ after_norm → tp_encoders[20](512→512)→ tp_norm。
- `EncoderLayerSANM`:pre-norm,`r=x; h=norm1(x); x=(r+attn(h) if in==out else attn(h)); r=x; h=norm2(x); x=r+ffn(h)`。LayerNorm eps=1e-5。
- `MultiHeadedAttentionSANM`:merged `linear_q_k_v`(in_feat→n_feat*3),`fsmn_block`= Conv1d(n_feat,n_feat,k=11,groups=n_feat,bias=False)+ `pad_fn` ConstantPad1d((5,5))(k=11,sanm_shift=0)残差在 v 上,`linear_out`,4 heads dk=128,scaling dk**-0.5。返回 `att_outs + fsmn_memory`。
- `PositionwiseFeedForward`:`w_2(relu(w_1(x)))`,hidden=linear_units=2048。
- 权重直接对应:`encoders0.0`/`encoders.N`/`tp_encoders.N`/`after_norm`/`tp_norm`,每个 block 含 `norm1`/`norm2`/`self_attn.{linear_q_k_v,linear_out,fsmn_block,linear_q_k_v.bias?无}`/`feed_forward.{w_1,w_2}`。**无堆叠**(linear_q_k_v checkpoint 里已是合并态)。

### Audio adaptor = funasr Transformer(downsample_rate=1)
- `linear1`(512→2048)→ relu → `linear2`(2048→1024)→ 2× pre-norm transformer block。
- block:标准 `MultiHeadedAttention`(分立 linear_q/k/v/out,8 heads dk=128)+ `PositionwiseFeedForward`(hidden=llm_dim//4=256)。pre-norm 残差。
- **downsample_rate=1 ⇒ 无内部下采样**,输出全长 T_lfr。
- 权重直接对应:`linear1`/`linear2`/`blocks.N.{norm1,norm2,self_attn.linear_{q,k,v,out},feed_forward.w_{1,2}}`。**无堆叠**。

### Low frame rate = 截断(非池化!)
`use_low_frame_rate`(model.py:480、inference_vllm.py:422、funasr-cli.cpp:149 三处一致):adaptor 返回全长 T_lfr,然后 `fake_token_len = fun_asr_low_frame_rate_length(T_lfr)`(3× ceil(x/2)),`adaptor_out[:fake_token_len]` 切片。**无 conv/pool/stride**。前 N 帧(已通过 self-attention 双向关注全部 T_lfr)喂给 LLM。`get_audio_feature` 里就是这么做的。

### LLM = Qwen3ForCausalLM(sglang)
- `config.text_config`(Qwen3,28 层,hidden 1024,16 heads,8 kv heads,head_dim 128,vocab 151936,rope_theta 1e6,tie_word_embeddings)。
- `general_mm_embed_routine` 把 `get_audio_feature` 输出 splice 到 `<|object_ref_start|>` 占位符位置。
- `load_weights`:`model.audio_encoder.*`→`audio_encoder.*`、`model.audio_adaptor.*`→`audio_adaptor.*`、`model.language_model.*`→`language_model.model.*`(Qwen3ForCausalLM 把 Qwen3Model 包在 `self.model` 下),LLM 的 q/k/v→qkv_proj、gate/up→gate_up_proj 堆叠,跳过 `lm_head.weight`(tied)和 `rotary_emb.{inv_freq,cos_cached,sin_cached}`。

### get_audio_feature(被 general_mm_embed_routine 调用)
- 每个 `MultimodalDataItem.feature` 是 `[1, 560, T_padded]` LFR 特征;用 `feature_attention_mask` 取有效帧 → `[1, T, 560]` → permute 成 `[1, T, 560]`(encoder 期望 `[B, T, D]`)→ encoder → adaptor → `[:num_tokens]` 截断 → `[num_tokens, 1024]`,cat 跨 item。

---

## 待办

### 1. payload_types.py
qwen3_asr 无此文件(`SGLangARRequestData` 在 `sglang_omni/scheduling/sglang_backend/request_data.py`,request_builders 直接 import)。**可省,保持空文件或删除。**

### 2. 精度对比(重要,上线前必做)
目标:证明我们的 sglang_model.py forward 与官方 funasr 数值一致。

**Encoder 已 bit-exact**(0.000e+00 vs `SenseVoiceEncoderSmall`)。**Adaptor 有 ~1e-3 差异待定位**。对比方法:

```python
# 已验证可用的对比脚本骨架(需 sglang runtime 初始化,见上方样板):
import sys; sys.path.insert(0, '/tmp/funasr_src/funasr_extracted')
import torch
from funasr.models.sense_voice.model import SenseVoiceEncoderSmall, sequence_mask
from funasr.models.llm_asr.adaptor import Transformer as FunasrTransformerAdaptor
from sglang_omni.models.fun_asr.sglang_model import FunAsrNanoAudioEncoder, FunAsrNanoAdaptor
# ... 加载相同权重,相同随机输入,对比 forward 输出 max_abs_diff
```

**adaptor 差异定位方向**(优先级):
1. funasr `Transformer.forward` 用 `make_pad_mask(ilens)` 生成 mask 并传给 `EncoderLayer.forward(x, mask)` → `MultiHeadedAttention(query,key,value,mask)` 的 `forward_attention` 走 `masked_fill(mask, -inf)` 分支。即使 ilens==T(全 valid),mask 是全 True 的 `~make_pad_mask`,会走 `if mask is not None` 分支(scaled_fill 0.0),与我的无 mask `softmax` 路径在数值上**应**等价但 float32 累积顺序不同。→ 检查是否真是这个。
2. funasr `MultiHeadedAttention.forward_qkv` 的 reshape/transpose 顺序(q/k/v split → view(b,t,h,dk) → transpose(1,2))与我的实现对比。
3. `PositionwiseFeedForward` 的 `dropout(activation(w_1))` 顺序(eval 下 dropout=identity,应无影响)。

**最终精度验证(端到端)✅(HTTP serve 路径已验证)**:通过 `python -m sglang_omni.cli serve --model-path <ckpt> --port 8731` 启动 HTTP 服务器(`FunASRPipelineConfig` 经 `resolve_config_cls_for_model_path` 自动发现,无需 `--config`)。OpenAI 兼容 `POST /v1/audio/transcriptions` 端点跑通:8 条 `test_data/*.opus` 并发转写全部 200,`response_format` json/text/verbose_json 均正常,单请求 wall ~0.33s,8 并发 ~1s。转写与参考 `.txt` 高度一致(标点/个别字差异)。注意:**CLI 不支持 `--mem-fraction-static`**(单阶段 ASR pipeline 无 `mem_fraction_role_to_stage`,会报 "requires a pipeline with a supported SGLang AR mem_fraction_static target")——`mem_fraction_static=0.45` 已在 `stages.py` factory 内部写死,无需 CLI 传。

### 3. sglang model arch 注册确认 ✅(本轮已完成)
sglang 0.5.12 内置 model registry **无** `FunAsrNanoForConditionalGeneration`(qwen3_asr 有,是 sglang 自带)。已在 `sglang_model_runner.py:_register_omni_model` 的 `sglang_omni_models` 字典加入映射 → `sglang_omni.models.fun_asr.sglang_model:FunAsrNanoForConditionalGeneration`。`_register_omni_model` 在 `SGLModelRunner.__init__` 内执行(即 `create_sglang_infrastructure` 路径),GPU smoke test 已确认模型类被实例化。

---

## 环境备忘
- conda env `sglang`:sglang 0.5.12.post1,transformers 5.6.0,torch 2.11.0+cu130。
- funasr 1.3.14 wheel 已下载解压到 `/tmp/funasr_src/funasr_extracted/`(供对比,`sys.path.insert(0, ...)` 即可 import)。已装 `kaldiio`、`torch_complex` 满足 funasr import。
- sglang Qwen3ForCausalLM 构建需 `init_distributed_environment` + `initialize_model_parallel(1,1)` + `set_global_server_args_for_scheduler(ServerArgs(model_path=CKPT))`,且**必须先 import configuration_fun_asr** 注册 `fun_asr_nano` AutoConfig,否则 ServerArgs 加载 config 时报 `KeyError: 'fun_asr_nano'`。
- 完整 LLM 权重加载验证(qkv_proj==cat(q,k,v) 等)因 Qwen3ForCausalLM runtime 依赖未跑通,待 stages.py 路径验证。

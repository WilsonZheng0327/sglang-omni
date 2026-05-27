# SPDX-License-Identifier: Apache-2.0

import queue
from types import SimpleNamespace

import torch

from sglang_omni.models.higgs_tts import stages
from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.streaming_vocoder import (
    HiggsStreamingVocoderScheduler,
)
from sglang_omni.models.higgs_tts.utils import EOC_ID, apply_delay_pattern
from sglang_omni.proto import OmniRequest, StagePayload


def test_higgs_tts_engine_enables_cuda_graph_by_default(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_sglang_server_args(checkpoint_dir, context_length, **overrides):
        server_args = SimpleNamespace(
            disable_cuda_graph=overrides["disable_cuda_graph"],
            disable_overlap_schedule=False,
        )
        captured["checkpoint_dir"] = checkpoint_dir
        captured["context_length"] = context_length
        captured["overrides"] = overrides
        captured["server_args"] = server_args
        return server_args

    def fake_create_sglang_infrastructure(server_args, gpu_id):
        captured["gpu_id"] = gpu_id
        model = SimpleNamespace(reset_request=lambda _request_id: None)
        return (
            SimpleNamespace(model_runner=SimpleNamespace(model=model)),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
        )

    class FakeOutputProcessor:
        def __init__(self, **kwargs) -> None:
            captured["output_processor_kwargs"] = kwargs

    class FakeModelRunner:
        def __init__(self, model_worker, output_proc) -> None:
            captured["model_runner_args"] = (model_worker, output_proc)

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            captured["scheduler_kwargs"] = kwargs

    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages, "build_sglang_server_args", fake_build_sglang_server_args
    )
    monkeypatch.setattr(
        stages, "create_sglang_infrastructure", fake_create_sglang_infrastructure
    )
    monkeypatch.setattr(stages, "truncate_rope_to_bf16", lambda model: None)
    monkeypatch.setattr(stages, "SGLangOutputProcessor", FakeOutputProcessor)
    monkeypatch.setattr(stages, "HiggsTTSModelRunner", FakeModelRunner)

    def fake_make_adapters(model, **kwargs):
        captured["adapter_kwargs"] = kwargs
        return None, None

    monkeypatch.setattr(stages, "make_higgs_scheduler_adapters", fake_make_adapters)
    monkeypatch.setattr(stages, "OmniScheduler", FakeScheduler)

    stages.create_sglang_tts_engine_executor("boson-sglang/higgs-audio-v3-tts-4b-base")

    assert captured["checkpoint_dir"] == "boson-sglang/higgs-audio-v3-tts-4b-base"
    assert captured["context_length"] == 4096
    assert captured["gpu_id"] == 0
    assert captured["overrides"]["disable_cuda_graph"] is False
    assert captured["overrides"]["cuda_graph_max_bs"] == 32
    assert captured["server_args"].disable_overlap_schedule is True
    assert captured["adapter_kwargs"] == {"max_new_tokens_cap": 2048}


def test_higgs_model_runner_marks_sampler_finish() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = queue.Queue()
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _rid_to_row={"req": 0},
        _output_codes={"req": [torch.tensor([EOC_ID, 1, 2])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True])),
    )
    req = SimpleNamespace(
        is_chunked=0,
        finished_reason=None,
        finished=lambda: False,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="", params={"stream": True}),
        data={},
    )
    data = SimpleNamespace(
        req=req,
        output_codes=[],
        generation_done=False,
        stage_payload=payload,
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert len(data.output_codes) == 1
    out = runner._outbox.get_nowait()
    assert out.type == "stream"
    assert out.target == "vocoder"
    assert out.metadata["stream"] is True
    assert out.data.tolist() == [EOC_ID, 1, 2]


def test_higgs_model_runner_marks_sampler_finish_cg() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = queue.Queue()
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.tensor([0]),
        _cg_active_delay_count=torch.tensor([8], dtype=torch.int32),
        _cg_active_eoc_countdown=torch.tensor([0], dtype=torch.int32),
        _cg_active_generation_done=torch.tensor([True]),
        _cg_active_last_codes=torch.tensor([[1, 2, 3]]),
        _cg_was_done=torch.tensor([False]),
        _cg_codes_BN=torch.tensor([[EOC_ID, 1, 2]]),
        _cg_collect_staging=torch.zeros((1, 3 + 2), dtype=torch.long),
        _sampler_pool=SimpleNamespace(
            delay_count=torch.zeros(1, dtype=torch.int32),
            eoc_countdown=torch.zeros(1, dtype=torch.int32),
            generation_done=torch.zeros(1, dtype=torch.bool),
            last_codes=torch.zeros((1, 3), dtype=torch.long),
        ),
    )
    req = SimpleNamespace(is_chunked=0, finished_reason=None)
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="", params={"stream": False}),
        data={},
    )
    data = SimpleNamespace(
        req=req,
        output_codes=[],
        generation_done=False,
        stage_payload=payload,
    )
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )
    forward_batch = SimpleNamespace(batch_size=1)

    runner._collect_step_outputs_cg(
        result,
        forward_batch,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.generation_done is True
    assert req.finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert len(data.output_codes) == 1
    assert runner._outbox.empty()


def test_higgs_model_runner_collect_cg_mixed_batch() -> None:
    """A 4-row batch covering chunked / was-done / active rows verifies the
    batched single-D2H packing preserves per-row semantics, including the
    bool->long->bool round-trip for generation_done.
    """
    n, k = 4, 3
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = None
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _cg_row_indices=torch.arange(n),
        _cg_active_delay_count=torch.zeros(n, dtype=torch.int32),
        _cg_active_eoc_countdown=torch.zeros(n, dtype=torch.int32),
        # row1's True must NOT leak into the was-done (skipped) request.
        _cg_active_generation_done=torch.tensor([False, True, False, True]),
        _cg_active_last_codes=torch.zeros((n, k), dtype=torch.long),
        _cg_was_done=torch.tensor([False, True, False, False]),
        _cg_codes_BN=torch.tensor([[1, 1, 1], [7, 8, 9], [20, 1, 2], [EOC_ID, 3, 4]]),
        _cg_collect_staging=torch.zeros((n, k + 2), dtype=torch.long),
        _sampler_pool=SimpleNamespace(
            delay_count=torch.zeros(n, dtype=torch.int32),
            eoc_countdown=torch.zeros(n, dtype=torch.int32),
            generation_done=torch.zeros(n, dtype=torch.bool),
            last_codes=torch.zeros((n, k), dtype=torch.long),
        ),
    )
    # row0 chunked, row1 was-done, row2 active (not done), row3 active (EOC done).
    reqs = [
        SimpleNamespace(is_chunked=1, finished_reason=None),
        SimpleNamespace(is_chunked=0, finished_reason=None),
        SimpleNamespace(is_chunked=0, finished_reason=None),
        SimpleNamespace(is_chunked=0, finished_reason=None),
    ]
    datas = [
        SimpleNamespace(req=r, output_codes=[], generation_done=False) for r in reqs
    ]
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(n, 4))
    )
    forward_batch = SimpleNamespace(batch_size=n)

    runner._collect_step_outputs_cg(
        result,
        forward_batch,
        [SimpleNamespace(request_id=f"req{i}", data=d) for i, d in enumerate(datas)],
    )

    assert [len(d.output_codes) for d in datas] == [0, 0, 1, 1]
    # Direct bool-list equality locks the bool->long->bool round-trip; the
    # was-done row stays False despite _cg_active_generation_done[1] being True.
    assert [d.generation_done for d in datas] == [False, False, False, True]
    assert result.next_token_ids.tolist() == [0, 0, 20, EOC_ID]
    assert datas[2].output_codes[0].tolist() == [20, 1, 2]
    assert datas[3].output_codes[0].tolist() == [EOC_ID, 3, 4]
    assert reqs[3].finished_reason.to_json() == {"type": "stop", "matched": EOC_ID}
    assert all(reqs[i].finished_reason is None for i in (0, 1, 2))


def test_higgs_model_runner_skips_already_finished_eager_request() -> None:
    runner = object.__new__(HiggsTTSModelRunner)
    runner._outbox = queue.Queue()
    runner._vocoder_target = "vocoder"
    runner.model = SimpleNamespace(
        _rid_to_row={"req": 0},
        _output_codes={"req": [torch.tensor([EOC_ID, 1, 2])]},
        _sampler_pool=SimpleNamespace(generation_done=torch.tensor([True])),
    )
    req = SimpleNamespace(
        is_chunked=0,
        finished_reason=object(),
        finished=lambda: True,
    )
    data = SimpleNamespace(req=req, output_codes=[], generation_done=True)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=torch.zeros(1, 4))
    )

    runner._collect_step_outputs(
        result,
        [SimpleNamespace(request_id="req", data=data)],
    )

    assert data.output_codes == []
    assert result.next_token_ids.tolist() == [0]
    assert runner._outbox.empty()


class _FakeHiggsCodec:
    def __init__(self, samples_per_frame: int = 4) -> None:
        self.samples_per_frame = samples_per_frame

    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        frames = int(codes_TN.shape[0])
        return torch.arange(frames * self.samples_per_frame, dtype=torch.float32)


def _higgs_payload(request_id: str, *, stream: bool, delayed_rows):
    state = HiggsTtsState(
        output_codes_delayed=delayed_rows,
        num_codebooks=3,
        prompt_tokens=2,
        completion_tokens=len(delayed_rows),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": stream}),
        data=state.to_dict(),
    )


def test_higgs_streaming_vocoder_emits_chunks_and_slim_final() -> None:
    raw_codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15],
            [16, 17, 18],
        ],
        dtype=torch.long,
    )
    delayed = apply_delay_pattern(raw_codes)
    scheduler = HiggsStreamingVocoderScheduler(
        _FakeHiggsCodec(),
        stream_stride=3,
        stream_followup_stride=2,
        stream_holdback_tokens=0,
    )
    payload = _higgs_payload("req", stream=True, delayed_rows=delayed.tolist())

    for row in delayed:
        scheduler._on_chunk(
            "req",
            SimpleNamespace(data=row, metadata={"stream": True}),
        )
    scheduler._on_done("req")
    scheduler._on_new_request("req", payload)

    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())

    stream_messages = [msg for msg in messages if msg.type == "stream"]
    result_messages = [msg for msg in messages if msg.type == "result"]
    assert len(stream_messages) >= 2
    assert len(result_messages) == 1
    assert result_messages[0].data.data == {
        "modality": "audio",
        "sample_rate": 24000,
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": len(delayed),
            "total_tokens": 2 + len(delayed),
        },
    }
    assert all(msg.metadata["modality"] == "audio" for msg in stream_messages)
    assert all("audio_data" in msg.data for msg in stream_messages)


def test_higgs_vocoder_non_streaming_returns_full_audio() -> None:
    raw_codes = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    delayed = apply_delay_pattern(raw_codes)
    scheduler = HiggsStreamingVocoderScheduler(_FakeHiggsCodec())
    payload = _higgs_payload("req", stream=False, delayed_rows=delayed.tolist())

    scheduler._on_new_request("req", payload)

    out = scheduler.outbox.get_nowait()
    assert out.type == "result"
    assert out.data.data["modality"] == "audio"
    assert out.data.data["sample_rate"] == 24000
    assert out.data.data["audio_data"] == list(range(8))

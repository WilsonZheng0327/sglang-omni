# SPDX-License-Identifier: Apache-2.0
"""Regression test: ``ignore_eos`` threads through the request stack.

The MMMU decode-throughput-parity lane needs the upstream SGLang sampler to
keep emitting tokens until ``max_new_tokens`` regardless of EOS. The flag
flows: ``ChatCompletionRequest.ignore_eos`` →
``_build_chat_generate_request`` → ``GenerateRequest.sampling.ignore_eos``
→ ``SamplingParams.to_dict()`` (consumed by
``build_sglang_thinker_request``) → upstream
``sglang.srt.sampling.sampling_params.SamplingParams(ignore_eos=...)``.

This file is split into two groups so the final-hop tests run on
lightweight dev boxes:

- ``test_chat_completion_*``: exercises the OAI protocol → GenerateRequest
  → SamplingParams.to_dict() hops. Requires ``fastapi`` (because
  ``sglang_omni.serve.openai_api`` imports FastAPI at module top), so
  these tests are gated by a per-test ``importorskip``.
- ``test_build_sglang_thinker_request_*``: exercises the final hop into
  upstream SGLang's ``SamplingParams`` constructor. Stubs ``sys.modules``
  for ``sglang.srt.sampling.sampling_params`` and
  ``sglang.srt.managers.schedule_batch`` BEFORE importing
  ``request_builders``, so these tests run without the real SGLang
  runtime, torch, or xxhash installed.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _import_chat_protocol_modules():
    """Import the chat-completion plumbing or skip the calling test."""
    pytest.importorskip("fastapi")
    from sglang_omni.client.types import SamplingParams as ClientSamplingParams  # noqa: E402
    from sglang_omni.serve.openai_api import _build_chat_generate_request  # noqa: E402
    from sglang_omni.serve.protocol import (  # noqa: E402
        ChatCompletionRequest,
        ChatMessage,
    )

    return _build_chat_generate_request, ChatCompletionRequest, ChatMessage, ClientSamplingParams


def test_chat_completion_ignore_eos_default_false() -> None:
    build, ChatCompletionRequest, ChatMessage, _ = _import_chat_protocol_modules()
    req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="ping")])
    gen_req = build(req)
    assert gen_req.sampling.ignore_eos is False


def test_chat_completion_ignore_eos_threads_through() -> None:
    build, ChatCompletionRequest, ChatMessage, _ = _import_chat_protocol_modules()
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="ping")],
        ignore_eos=True,
    )
    gen_req = build(req)
    assert gen_req.sampling.ignore_eos is True


def test_client_sampling_params_to_dict_carries_ignore_eos() -> None:
    _, _, _, ClientSamplingParams = _import_chat_protocol_modules()
    sp = ClientSamplingParams(ignore_eos=True)
    assert sp.to_dict()["ignore_eos"] is True


def test_client_sampling_params_default_to_dict_false() -> None:
    _, _, _, ClientSamplingParams = _import_chat_protocol_modules()
    sp = ClientSamplingParams()
    assert sp.to_dict()["ignore_eos"] is False


# ---------------------------------------------------------------------------
# Final-hop tests: stub upstream SGLang modules BEFORE importing the
# request_builders module so the tests run on dev boxes that do not have
# the real sglang / torch / xxhash packages installed. The stub modules
# expose just enough surface for the function to execute the line we care
# about (the ``SamplingParams(...)`` instantiation) and bail out further
# down the chain.
# ---------------------------------------------------------------------------


def _install_module_stub(name: str, **attrs) -> types.ModuleType:
    """Install a stub module under sys.modules with the requested attrs."""
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def _install_stubs_and_get_builder():
    """Return (request_builders module, fake upstream SamplingParams).

    Installs minimal sys.modules stubs for every heavy dependency the
    target import chain pulls in. The goal is to load
    ``sglang_omni.models.qwen3_omni.request_builders`` without
    requiring sglang / torch / xxhash / safetensors / numpy / FastAPI
    installed. The function under test (``build_sglang_thinker_request``)
    only needs the SamplingParams + Req mocks to assert its kwargs.
    """
    fake_sampling_params = MagicMock(name="upstream_SamplingParams")
    fake_req_class = MagicMock(name="upstream_Req")
    fake_mm_inputs = MagicMock(name="upstream_MultimodalInputs")

    # Parent packages must exist before any submodule import; mark them
    # as packages by giving them __path__ so child imports work.
    for parent in (
        "sglang",
        "sglang.srt",
        "sglang.srt.sampling",
        "sglang.srt.managers",
        "sglang.srt.mem_cache",
    ):
        pkg = sys.modules.setdefault(parent, types.ModuleType(parent))
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = []

    # Stub the deeper SGLang module the scheduling backend imports.
    _install_module_stub(
        "sglang.srt.mem_cache.cache_init_params",
        CacheInitParams=MagicMock(name="CacheInitParams"),
    )

    _install_module_stub(
        "sglang.srt.sampling.sampling_params",
        SamplingParams=fake_sampling_params,
    )
    _install_module_stub(
        "sglang.srt.managers.schedule_batch",
        Req=fake_req_class,
        MultimodalInputs=fake_mm_inputs,
    )

    # torch + xxhash are imported at request_builders module top.
    if "torch" not in sys.modules:
        _install_module_stub(
            "torch", long="long", Tensor=MagicMock(name="torch.Tensor")
        )
    if "xxhash" not in sys.modules:
        _install_module_stub(
            "xxhash",
            xxh3_64=MagicMock(return_value=MagicMock(intdigest=lambda: 0)),
        )

    # safetensors is imported by talker_prefill.py, a transitive dep.
    if "safetensors" not in sys.modules:
        _install_module_stub("safetensors", safe_open=MagicMock())

    # Stub talker_prefill itself so the heavy chain (numpy, transformers, etc.)
    # never needs to import. We only need `TalkerPrefillBuilder` to be a
    # symbol; request_builders binds it at module load time.
    _install_module_stub(
        "sglang_omni.models.qwen3_omni.components.talker_prefill",
        TalkerPrefillBuilder=MagicMock(name="TalkerPrefillBuilder"),
    )

    # Stub the scheduling.sglang_backend package and scheduling.types so we
    # don't drag the upstream SGLang RadixCache chain through. Only the
    # `SGLangARRequestData` + `ARRequestData` + `OutgoingMessage` symbols
    # are needed at request_builders module load time.
    _install_module_stub(
        "sglang_omni.scheduling.sglang_backend",
        SGLangARRequestData=MagicMock(name="SGLangARRequestData"),
    )
    _install_module_stub(
        "sglang_omni.scheduling.types",
        ARRequestData=MagicMock(name="ARRequestData"),
    )
    _install_module_stub(
        "sglang_omni.scheduling.messages",
        OutgoingMessage=MagicMock(name="OutgoingMessage"),
    )

    # Force a fresh import so the stubs take effect.
    sys.modules.pop("sglang_omni.models.qwen3_omni.request_builders", None)
    from sglang_omni.models.qwen3_omni import request_builders  # noqa: E402

    return request_builders, fake_sampling_params


def _make_state_stub():
    """Build a minimal PipelineState-like object the builder can chew on."""
    state = MagicMock(name="PipelineState")
    # input_ids needs .clone(), .to(dtype=...), and .tolist(). MagicMock
    # auto-creates all of these; the values themselves are irrelevant since
    # the test stops once SamplingParams is called.
    state.prompt = {
        "input_ids": MagicMock(name="input_ids"),
        "attention_mask": None,
    }
    state.thinker_inputs = None
    return state


def test_build_sglang_thinker_request_passes_ignore_eos_to_upstream() -> None:
    """AC-10 final-hop: build_sglang_thinker_request forwards
    ``ignore_eos=True`` into the upstream SGLang ``SamplingParams(...)``
    at ``request_builders.py:347``. Runs without the real SGLang
    runtime via sys.modules stubs."""
    request_builders, fake_sampling_params = _install_stubs_and_get_builder()
    state = _make_state_stub()
    tokenizer = MagicMock()

    try:
        request_builders.build_sglang_thinker_request(
            state,
            params={"ignore_eos": True, "max_new_tokens": 256},
            tokenizer=tokenizer,
            vocab_size=152064,
        )
    except Exception:
        # Downstream Req construction may trip on the mock; we only care
        # about the SamplingParams call.
        pass

    fake_sampling_params.assert_called()
    kwargs = fake_sampling_params.call_args.kwargs
    assert kwargs.get("ignore_eos") is True


def test_build_sglang_thinker_request_defaults_ignore_eos_false() -> None:
    """When params dict omits ``ignore_eos``, upstream SGLang gets False."""
    request_builders, fake_sampling_params = _install_stubs_and_get_builder()
    state = _make_state_stub()
    tokenizer = MagicMock()

    try:
        request_builders.build_sglang_thinker_request(
            state,
            params={"max_new_tokens": 2048},
            tokenizer=tokenizer,
            vocab_size=152064,
        )
    except Exception:
        pass

    fake_sampling_params.assert_called()
    kwargs = fake_sampling_params.call_args.kwargs
    assert kwargs.get("ignore_eos") is False

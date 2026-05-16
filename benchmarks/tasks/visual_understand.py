# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark case: answer parsing and request execution."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
import re
import time
from typing import Callable, TypedDict

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import SendFn
from benchmarks.dataset.mmmu import MMMUSample, image_to_data_uri

logger = logging.getLogger(__name__)

MULTI_CHOICE_INSTRUCTION = (
    "\nAnswer the following multiple-choice question. "
    "The last line of your response should be of the "
    "following format: 'Answer: $LETTER' (without quotes) "
    "where LETTER is one of the options. "
    "Think step by step before answering."
)


class MMMURecord(TypedDict):
    sample_id: str
    subject: str
    question_type: str
    expected: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tok_per_s: float | None
    predicted: str
    raw_response: str
    is_correct: bool
    is_success: bool
    is_mc_fallback: bool
    error: str
    lane: str


def _check_is_number(s: str) -> bool:
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(s: str) -> list[float | str]:
    """Normalize a string for open-ended answer comparison."""
    s = s.strip()
    if _check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    return [s.lower()] if len(s) > 1 else [" " + s, s + " "]


def _extract_numbers(s: str) -> list[str]:
    """Extract all numbers (with commas, scientific notation, etc.) from *s*."""
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return (
        re.findall(pattern_commas, s)
        + re.findall(pattern_scientific, s)
        + re.findall(pattern_simple, s)
    )


def _parse_open_answer_tag(response: str) -> str | None:
    """Try to extract the answer from an explicit 'Answer: ...' line.

    Supports formats like ``Answer: 42``, ``Answer: MgS``,
    ``Answer: \\boxed{13.0}``.  Returns ``None`` when no match is found.
    """
    matches = re.findall(
        r"[Aa]nswer\s*:\s*\*?\*?\s*(.+)",
        response,
    )
    if not matches:
        return None
    raw = matches[-1].strip().rstrip(".")
    # Unwrap \boxed{...} if present
    boxed = re.search(r"\\boxed\{(.+?)\}", raw)
    if boxed:
        raw = boxed.group(1)
    # Strip surrounding ** (bold markdown)
    raw = raw.strip("*").strip()
    return raw if raw else None


def parse_open_response(response: str) -> list[float | str]:
    """Extract answer candidates from an open-ended model response.

    First tries to extract from an explicit ``Answer: ...`` line.
    Falls back to heuristic key-subresponse extraction.
    """
    # Fast path: explicit "Answer: ..."
    tag_answer = _parse_open_answer_tag(response)
    if tag_answer is not None:
        out: list = []
        out.extend(_normalize_str(tag_answer))
        for num in _extract_numbers(tag_answer):
            out.extend(_normalize_str(num))
        return list(dict.fromkeys(out))

    # Fallback: heuristic extraction
    def _get_key_subresponses(resp: str) -> list[str]:
        resp = resp.strip().strip(".").lower()
        subs = re.split(r"\.\s(?=[A-Z])|\n", resp)
        indicators = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        keys: list[str] = []
        for i, s in enumerate(subs):
            cands = indicators + ["="] if i == len(subs) - 1 else indicators
            shortest = None
            for ind in cands:
                if ind in s:
                    part = s.split(ind)[-1].strip()
                    if not shortest or len(part) < len(shortest):
                        shortest = part
            if shortest and shortest not in (":", ",", ".", "!", "?", ";", "'"):
                keys.append(shortest)
        return keys or [resp]

    key_resps = _get_key_subresponses(response)
    pred_list = key_resps.copy()
    for r in key_resps:
        pred_list.extend(_extract_numbers(r))
    out = []
    for x in pred_list:
        out.extend(_normalize_str(x))
    return list(dict.fromkeys(out))


def eval_open(gold: str | list[str], preds: list[float | str]) -> bool:
    """Check if any prediction matches the gold answer (fuzzy)."""
    if isinstance(gold, list):
        norm_answers: list = []
        for ans in gold:
            norm_answers.extend(_normalize_str(ans))
    else:
        norm_answers = _normalize_str(gold)
    for p in preds:
        if isinstance(p, str):
            for na in norm_answers:
                if isinstance(na, str) and na in p:
                    return True
        else:
            if p in norm_answers:
                return True
    return False


def parse_multi_choice_response(
    response: str,
    all_choices: list[str],
    index2ans: dict[str, str],
) -> tuple[str, bool]:
    """Extract a single answer letter from the model response.

    Priority: ``Answer: X`` → ``(A)`` bracket → ``·A·`` space-padded →
    option-text match → last-occurrence tie-break → random fallback.

    Returns ``(choice, is_fallback)``. ``is_fallback`` is ``True`` iff
    nothing could be parsed out of *response* and a random choice was
    returned — this counter is observational and doesn't change scoring
    behavior vs. the MMMU reference eval.
    """
    answer_matches = re.findall(r"[Aa]nswer\s*:\s*\*?\*?\s*\(?([A-Z])\)?", response)
    if answer_matches:
        candidate = answer_matches[-1]
        if candidate in all_choices:
            return candidate, False

    for char in (",", ".", "!", "?", ";", ":", "'"):
        response = response.strip(char)
    response = " " + response + " "

    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans and ans.lower() in response.lower():
                candidates.append(idx)
    if not candidates:
        return random.choice(all_choices), True
    if len(candidates) == 1:
        return candidates[0], False

    starts: list[int] = []
    for can in candidates:
        pos = response.rfind(f"({can})")
        if pos == -1:
            pos = response.rfind(f" {can} ")
        if pos == -1 and index2ans.get(can):
            pos = response.lower().rfind(index2ans[can].lower())
        starts.append(pos)
    return candidates[max(range(len(candidates)), key=starts.__getitem__)], False


def assert_sglang_payload_order_contract(payload: dict) -> None:
    """Validate ``[image_url..., text]`` part order on an SGLang payload.

    Mirrors ``Qwen3OmniPreprocessor._build_multimodal_messages``: image
    placeholders come BEFORE text. Raises ValueError on violation.
    """
    messages = payload.get("messages") or []
    if not messages:
        raise ValueError("payload has no messages")
    content = messages[0].get("content")
    if not isinstance(content, list):
        raise ValueError("sglang payload's first message must use a part-list content")
    if not content:
        raise ValueError("content list is empty")
    # All but the last part must be image_url; the last part must be text.
    for i, part in enumerate(content[:-1]):
        if part.get("type") != "image_url":
            raise ValueError(
                f"sglang payload content part {i} has type {part.get('type')!r}, "
                f"expected 'image_url' (text must come AFTER all image parts)"
            )
    last = content[-1]
    if last.get("type") != "text":
        raise ValueError(
            f"sglang payload's last content part has type {last.get('type')!r}, "
            f"expected 'text' (image parts come BEFORE text)"
        )


def build_mmmu_payload(
    sample: MMMUSample,
    model_name: str,
    *,
    backend: str,
    modalities: list[str],
    max_tokens: int,
    temperature: float,
    stream: bool,
    enable_audio: bool,
    seed: int | None = None,
    ignore_eos: bool = False,
) -> dict:
    """Render the chat-completions payload for one MMMU sample.

    ``backend="omni"`` → top-level ``images`` field + plain string prompt.
    ``backend="sglang"`` → OAI ``messages[].content`` part-list in order
    ``[image_url..., text]``, mirroring ``_build_multimodal_messages``
    (image placeholders BEFORE text).

    All other request fields (model/modalities/max_tokens/temperature/stream/
    seed/ignore_eos) are identical across backends.
    """
    image_uris = [image_to_data_uri(img) for img in sample.images]

    if backend == "omni":
        payload: dict = {
            "model": model_name,
            "messages": [{"role": "user", "content": sample.prompt}],
            "images": image_uris,
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    elif backend == "sglang":
        content_parts: list[dict] = []
        for uri in image_uris:
            content_parts.append({"type": "image_url", "image_url": {"url": uri}})
        content_parts.append({"type": "text", "text": sample.prompt})
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "modalities": modalities,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    else:
        raise ValueError(f"Unsupported backend: {backend!r}; expected 'omni' or 'sglang'")

    if enable_audio:
        payload["audio"] = {"format": "wav"}
    if seed is not None:
        payload["seed"] = seed
    if ignore_eos:
        payload["ignore_eos"] = True
    return payload


async def consume_sse_stream(
    response: aiohttp.ClientResponse,
    result: RequestResult,
    request_send_time: float,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> None:
    """Consume an SSE chat-completions stream and populate ``result`` in place.

    Frame classes: ``[DONE]`` sentinel (skip); role-only delta (skip);
    content delta (timestamp via offset from ``request_send_time``,
    concat into result.text, increment count); finish chunk (extract usage
    if present, not timed).

    Parser buffers raw bytes since TCP read boundaries don't align with
    SSE ``data: <json>\\n\\n`` frames. ``clock`` is overridable for tests.
    """
    buffer = bytearray()
    saw_done = False
    final_usage: dict | None = None

    async for chunk in response.content.iter_any():
        if not chunk:
            continue
        buffer.extend(chunk)
        while b"\n\n" in buffer:
            sep_idx = buffer.index(b"\n\n")
            raw_frame = bytes(buffer[:sep_idx])
            del buffer[: sep_idx + 2]

            for raw_line in raw_frame.split(b"\n"):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                body = line[len("data: "):]
                if body == "[DONE]":
                    saw_done = True
                    continue
                try:
                    frame = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning("SSE: skipping malformed JSON frame: %r", body[:200])
                    continue

                choices = frame.get("choices") or []
                delta = (choices[0].get("delta") if choices else {}) or {}
                finish_reason = choices[0].get("finish_reason") if choices else None
                content = delta.get("content")

                if content:
                    arrival = clock()
                    offset_ms = max(0.0, (arrival - request_send_time) * 1000.0)
                    if result.ttft_s is None:
                        result.ttft_s = round(arrival - request_send_time, 6)
                    result.content_chunk_offsets_ms.append(round(offset_ms, 4))
                    result.content_chunk_count += 1
                    result.text += content
                elif finish_reason is not None:
                    usage = frame.get("usage")
                    if isinstance(usage, dict):
                        final_usage = usage

        if saw_done:
            break

    # Parser contract negative: any bytes remaining in the buffer when the stream ends
    # mean the server emitted a partial frame without its terminating
    # ``\n\n``. That violates the SSE framing contract and would otherwise
    # cause the parser to silently drop a frame.
    trailing = bytes(buffer).strip()
    if trailing:
        raise RuntimeError(
            "SSE stream ended with an unterminated frame (no trailing CRLF/CRLF). "
            f"Trailing buffer (first 200 bytes): {trailing[:200]!r}"
        )

    if final_usage:
        prompt_tok = final_usage.get("prompt_tokens")
        comp_tok = final_usage.get("completion_tokens")
        if prompt_tok is not None:
            result.prompt_tokens = int(prompt_tok)
        if comp_tok is not None:
            result.completion_tokens = int(comp_tok)


def make_mmmu_send_fn(
    model_name: str,
    api_url: str,
    *,
    backend: str = "omni",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    stream: bool = False,
    seed: int | None = None,
    ignore_eos: bool = False,
    enable_audio: bool = False,
    audio_dir: str | None = None,
) -> SendFn:
    """Build a *send_fn* that POSTs an MMMUSample to /v1/chat/completions.

    ``stream=True`` consumes the SSE response and populates ttft_s +
    content_chunk_offsets_ms + content_chunk_count. Mutually exclusive
    with ``enable_audio=True`` (different response shape).

    Populates ``client_wall_time_s`` + ``timing_source="client_wall_time_s"``
    so aggregates land under ``tok_per_s_clientwall_agg``.
    """
    if stream and enable_audio:
        raise ValueError(
            "stream + enable_audio is out of scope: the audio response shape "
            "does not flow through the per-token SSE path"
        )
    if backend not in ("omni", "sglang"):
        raise ValueError(f"backend must be 'omni' or 'sglang', got {backend!r}")

    modalities = ["text", "audio"] if enable_audio else ["text"]
    if enable_audio:
        import soundfile as sf

    async def send_fn(
        session: aiohttp.ClientSession, sample: MMMUSample
    ) -> RequestResult:
        result = RequestResult(
            request_id=sample.sample_id,
            text="" if stream else sample.prompt[:60],
        )

        payload = build_mmmu_payload(
            sample,
            model_name,
            backend=backend,
            modalities=modalities,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            enable_audio=enable_audio,
            seed=seed,
            ignore_eos=ignore_eos,
        )

        start_time = time.perf_counter()
        try:
            if stream:
                result.text = ""
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    await consume_sse_stream(
                        response, result, request_send_time=start_time
                    )
                if result.content_chunk_count == 0 and not result.text:
                    result.error = "stream ended with no content chunks"
                    return result
                result.is_success = True
            else:
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    body = await response.json()

                message = body.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "")
                result.text = content or ""

                if enable_audio and audio_dir:
                    audio_obj = message.get("audio")
                    if audio_obj is None:
                        result.error = "No audio in response"
                        return result
                    audio_b64 = audio_obj.get("data", "")
                    if not audio_b64:
                        result.error = "Empty audio data in response"
                        return result
                    wav_path = os.path.join(audio_dir, f"{sample.sample_id}.wav")
                    with open(wav_path, "wb") as f:
                        f.write(base64.b64decode(audio_b64))
                    result.wav_path = wav_path

                    wav_info = sf.info(wav_path)
                    result.audio_duration_s = round(wav_info.duration, 4)

                result.is_success = True

                usage = body.get("usage", {})
                if usage:
                    result.prompt_tokens = usage.get("prompt_tokens", 0)
                    result.completion_tokens = usage.get("completion_tokens", 0)

            elapsed = time.perf_counter() - start_time
            result.client_wall_time_s = elapsed
            result.timing_source = "client_wall_time_s"
            if result.audio_duration_s > 0:
                result.rtf = elapsed / result.audio_duration_s
            if result.completion_tokens > 0 and elapsed > 0:
                result.tok_per_s = result.completion_tokens / elapsed
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        return result

    return send_fn


def build_mmmu_result_records(
    samples: list[MMMUSample],
    results: list[RequestResult],
    lane: str = "A",
) -> list[MMMURecord]:
    """Parse responses into persisted per-sample records.

    ``lane`` is stamped onto each record so the report generator can reject
    mixed-lane inputs at the record layer (not just file-level metadata).
    """
    assert len(samples) == len(
        results
    ), f"Sample/result count mismatch: {len(samples)} samples vs {len(results)} results"
    # Fix the random seed so MC fallback choices stay deterministic across CI runs.
    random.seed(42)

    per_sample: list[MMMURecord] = []

    for sample, result in zip(samples, results):
        record: MMMURecord = {
            "sample_id": sample.sample_id,
            "subject": sample.subject,
            "question_type": sample.question_type,
            "expected": sample.answer,
            "latency_s": round(result.latency_s, 4),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "tok_per_s": (round(result.tok_per_s, 1) if result.tok_per_s > 0 else None),
            "predicted": "",
            "raw_response": result.error,
            "is_correct": False,
            "is_success": False,
            "is_mc_fallback": False,
            "error": result.error,
            "lane": lane,
        }

        if result.is_success:
            gold = sample.answer
            is_fallback = False
            if (
                sample.question_type == "multiple-choice"
                and sample.all_choices
                and sample.index2ans
            ):
                predicted, is_fallback = parse_multi_choice_response(
                    result.text,
                    sample.all_choices,
                    sample.index2ans,
                )
                if is_fallback:
                    logger.debug(
                        f"MMMU multi-choice parse fallback for sample "
                        f"{sample.sample_id}"
                    )
                is_correct = gold is not None and predicted == gold
            else:
                parsed_list = parse_open_response(result.text)
                is_correct = gold is not None and eval_open(gold, parsed_list)
                predicted = ", ".join(map(str, parsed_list))

            record.update(
                predicted=predicted,
                raw_response=result.text,
                is_correct=is_correct,
                is_success=True,
                is_mc_fallback=is_fallback,
                error="",
            )

        per_sample.append(record)

    return per_sample

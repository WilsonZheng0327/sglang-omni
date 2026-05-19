# SPDX-License-Identifier: Apache-2.0
"""Shared exception types for sglang-omni."""

from __future__ import annotations

PIPELINE_BAD_REQUEST_MARKER = "[sglang_omni.pipeline_bad_request]"


class PipelineBadRequestError(ValueError):
    """User/request error raised inside pipeline execution.

    The stable marker survives process boundaries where exceptions are
    serialized into strings, while ``isinstance`` still works before that
    boundary.
    """

    def __str__(self) -> str:
        message = super().__str__()
        if message.startswith(PIPELINE_BAD_REQUEST_MARKER):
            return message
        return f"{PIPELINE_BAD_REQUEST_MARKER} {message}"


def is_pipeline_bad_request_error(exc: Exception) -> bool:
    """Return whether ``exc`` represents a pipeline bad-request failure."""

    return isinstance(
        exc, PipelineBadRequestError
    ) or PIPELINE_BAD_REQUEST_MARKER in str(exc)

# SPDX-License-Identifier: Apache-2.0
"""Reusable HTTP control routes for pipeline servers."""

from __future__ import annotations

import os
import time
from typing import Mapping

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from sglang_omni.profiler.profiler_control import ProfilerControlClient


def _default_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S")


def _default_template(profiler_dir: str, run_id: str) -> str:
    return os.path.join(profiler_dir, run_id, "trace")


class StartProfileRequest(BaseModel):
    run_id: str | None = None
    trace_path_template: str | None = None


class StopProfileRequest(BaseModel):
    run_id: str | None = None


def mount_control_routes(
    app,
    stage_control_endpoints: Mapping[str, str],
    *,
    profiler_dir: str | None = None,
) -> ProfilerControlClient:
    """Mount profiler/control routes and return the client they use.

    Callers that own an app outside :func:`launch_server` should close the
    returned client during shutdown.
    """

    profiler_ctl = ProfilerControlClient(dict(stage_control_endpoints))
    router = APIRouter()

    @router.post("/start_profile")
    async def start(req: StartProfileRequest):
        run_id = req.run_id or _default_run_id()
        if req.trace_path_template is not None:
            tpl = req.trace_path_template
        elif profiler_dir is not None:
            tpl = _default_template(profiler_dir, run_id)
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "trace_path_template is required when "
                    "SGLANG_TORCH_PROFILER_DIR is not set"
                ),
            )
        await profiler_ctl.broadcast_start(
            run_id=run_id,
            trace_path_template=tpl,
        )
        return {"run_id": run_id, "trace_path_template": tpl}

    @router.post("/stop_profile")
    async def stop(req: StopProfileRequest):
        run_id = req.run_id or "default"
        await profiler_ctl.broadcast_stop(run_id=run_id)
        return {"run_id": run_id}

    app.include_router(router)
    return profiler_ctl

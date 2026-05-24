# SPDX-License-Identifier: Apache-2.0
"""Process-local coordination for CUDA graph capture.

CUDA graph capture is sensitive to CUDA API calls from other Python threads in
the same process. The pipeline can run multiple stage schedulers and relay
drainers in one process, so encoder capture needs a shared guard with relay
CUDA copies.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import RLock
from typing import Iterator

# Serializes CUDA graph capture against relay CUDA copies in this process.
_CUDA_CAPTURE_LOCK = RLock()


@contextmanager
def cuda_capture_guard(enabled: bool = True) -> Iterator[None]:
    if enabled:
        with _CUDA_CAPTURE_LOCK:
            yield
    else:
        with nullcontext():
            yield

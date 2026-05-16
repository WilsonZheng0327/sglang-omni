#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single report-generation entry point for an MMMU sweep bundle.

Owns the mixed-lane guard: reads retained ``mmmu_results.json`` cells and
rejects payloads where per-sample records (or run_metadata) span more than
one lane. Caller-driven beyond that — no accuracy/latency tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


class MixedLaneError(ValueError):
    """Per-sample MMMU records span more than one lane."""


def assert_single_lane(records: Iterable[dict]) -> str:
    """Return the single lane these records carry; raise ``MixedLaneError`` on mix or missing field."""
    records = list(records)
    if not records:
        raise MixedLaneError(
            "no per-sample records supplied; cannot validate lane purity"
        )
    seen: set[str] = set()
    for idx, record in enumerate(records):
        if "lane" not in record:
            raise MixedLaneError(
                f"per-sample record at index {idx} is missing the 'lane' field "
                f"(sample_id={record.get('sample_id', '<unknown>')!r}); "
                f"cannot prove lane purity"
            )
        seen.add(record["lane"])
        if len(seen) > 1:
            raise MixedLaneError(
                f"mixed lane records detected: input contains both "
                f"{sorted(seen)!r}; report generator refuses mixed Lane A / "
                f"Lane B per-sample input. First conflict at record index "
                f"{idx} (sample_id={record.get('sample_id', '<unknown>')!r}, "
                f"lane={record['lane']!r})"
            )
    return next(iter(seen))


def load_cell(result_path: Path) -> tuple[str, list[dict]]:
    """Load a retained ``mmmu_results.json``, cross-check lane between metadata and per_sample, return ``(lane, per_sample)``."""
    data = json.loads(result_path.read_text())
    meta_lane = (data.get("run_metadata") or {}).get("lane")
    per_sample = data.get("per_sample") or []
    record_lane = assert_single_lane(per_sample) if per_sample else meta_lane
    if meta_lane and record_lane and meta_lane != record_lane:
        raise MixedLaneError(
            f"{result_path}: run_metadata.lane={meta_lane!r} disagrees with "
            f"per-sample record lane={record_lane!r}; refusing inconsistent input"
        )
    return (record_lane or meta_lane or ""), per_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cells",
        nargs="+",
        help="Paths to retained mmmu_results.json files to validate as one report input.",
    )
    args = parser.parse_args(argv)

    all_records: list[dict] = []
    for cell_path in args.cells:
        _, per_sample = load_cell(Path(cell_path))
        all_records.extend(per_sample)
    lane = assert_single_lane(all_records)
    print(f"[report] OK — {len(all_records)} records, lane={lane!r}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MixedLaneError as exc:
        print(f"[report] FAILED — {exc}", file=sys.stderr)
        sys.exit(2)

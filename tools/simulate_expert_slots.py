# SPDX-License-Identifier: Apache-2.0
"""Replay SEW-Offload MoE traces through a fixed-slot simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_ascend.moe_offload.slot_simulator import ExpertSizeTable, SlotSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--num-slots", type=int, required=True)
    parser.add_argument("--policy", default="lru", choices=("lru", "sticky_layer_lru"))
    parser.add_argument("--expert-bytes", type=int, default=14_680_064)
    parser.add_argument("--host-to-hbm-bandwidth-gbps", type=float, default=24.0)
    parser.add_argument("--output")
    return parser.parse_args()


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    records = load_trace(args.trace)
    simulator = SlotSimulator(
        size_table=ExpertSizeTable(default_expert_bytes=args.expert_bytes),
        host_to_hbm_bandwidth_gbps=args.host_to_hbm_bandwidth_gbps,
    )
    summary = simulator.replay(records, num_slots=args.num_slots, policy_name=args.policy).to_jsonable()
    summary["trace"] = str(args.trace)
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print("SIM_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

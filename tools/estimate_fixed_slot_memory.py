# SPDX-License-Identifier: Apache-2.0
"""Estimate fixed-slot memory ownership before releasing full expert weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES = 14_680_064
DEFAULT_OFFLOAD_BUDGET_GB = 13.5


def estimate_fixed_slot_memory(
    *,
    num_layers: int,
    num_experts_per_layer: int,
    expert_bytes: int,
    num_slots: int,
    original_weights_retained: bool = True,
    host_store_enabled: bool = True,
) -> dict[str, Any]:
    if num_layers <= 0:
        raise ValueError("num_layers must be greater than 0")
    if num_experts_per_layer <= 0:
        raise ValueError("num_experts_per_layer must be greater than 0")
    if expert_bytes <= 0:
        raise ValueError("expert_bytes must be greater than 0")
    if num_slots <= 0:
        raise ValueError("num_slots must be greater than 0")

    full_expert_bytes = int(num_layers) * int(num_experts_per_layer) * int(expert_bytes)
    slot_bank_bytes = int(num_layers) * int(num_slots) * int(expert_bytes)
    original_expert_weight_bytes = full_expert_bytes if original_weights_retained else 0
    host_store_bytes = full_expert_bytes if host_store_enabled else 0
    incremental_runtime_bytes = host_store_bytes + slot_bank_bytes
    total_managed_bytes = original_expert_weight_bytes + incremental_runtime_bytes

    return {
        "num_layers": int(num_layers),
        "num_experts_per_layer": int(num_experts_per_layer),
        "expert_bytes": int(expert_bytes),
        "num_slots": int(num_slots),
        "capacity_model": "per_layer_slot_bank",
        "original_weights_retained": bool(original_weights_retained),
        "host_store_enabled": bool(host_store_enabled),
        "original_expert_weight_bytes": original_expert_weight_bytes,
        "host_store_bytes": host_store_bytes,
        "slot_bank_bytes": slot_bank_bytes,
        "incremental_runtime_bytes": incremental_runtime_bytes,
        "total_managed_bytes": total_managed_bytes,
    }


def compare_slot_budget_models(
    *,
    num_layers: int,
    num_experts_per_layer: int,
    expert_bytes: int,
    num_slots: int,
    resident_layer_count: int = 0,
    original_weights_retained: bool = True,
    host_store_enabled: bool = True,
    offload_budget_gb: float = DEFAULT_OFFLOAD_BUDGET_GB,
) -> dict[str, Any]:
    """Contrast per-layer slot banks vs one global slot pool (design reference)."""
    if resident_layer_count < 0 or resident_layer_count > num_layers:
        raise ValueError("resident_layer_count must be in [0, num_layers]")

    offload_layers = int(num_layers) - int(resident_layer_count)
    per_layer = estimate_fixed_slot_memory(
        num_layers=num_layers,
        num_experts_per_layer=num_experts_per_layer,
        expert_bytes=expert_bytes,
        num_slots=num_slots,
        original_weights_retained=original_weights_retained,
        host_store_enabled=host_store_enabled,
    )
    global_slot_bank_bytes = int(num_slots) * int(expert_bytes)
    full_expert_bytes = int(num_layers) * int(num_experts_per_layer) * int(expert_bytes)
    resident_original_bytes = int(resident_layer_count) * int(num_experts_per_layer) * int(expert_bytes)
    if original_weights_retained:
        original_expert_weight_bytes = full_expert_bytes
    else:
        original_expert_weight_bytes = resident_original_bytes
    host_store_bytes = (
        int(offload_layers) * int(num_experts_per_layer) * int(expert_bytes) if host_store_enabled else 0
    )
    global_managed = original_expert_weight_bytes + host_store_bytes + global_slot_bank_bytes
    budget_bytes = int(float(offload_budget_gb) * (1024**3))

    return {
        "offload_budget_gb": float(offload_budget_gb),
        "offload_budget_bytes": budget_bytes,
        "resident_layer_count": int(resident_layer_count),
        "offload_layer_count": offload_layers,
        "per_layer_slot_bank": per_layer,
        "global_slot_bank": {
            "capacity_model": "global_slot_pool",
            "num_slots": int(num_slots),
            "slot_bank_bytes": global_slot_bank_bytes,
            "host_store_bytes": host_store_bytes,
            "original_expert_weight_bytes": original_expert_weight_bytes,
            "total_managed_bytes": global_managed,
        },
        "slot_bank_bytes_ratio_per_layer_over_global": (
            per_layer["slot_bank_bytes"] / global_slot_bank_bytes if global_slot_bank_bytes else None
        ),
        "per_layer_slot_bank_within_offload_budget": per_layer["slot_bank_bytes"] <= budget_bytes,
        "global_slot_bank_within_offload_budget": global_slot_bank_bytes <= budget_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts-per-layer", type=int, default=128)
    parser.add_argument("--expert-bytes", type=int, default=DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES)
    parser.add_argument("--num-slots", type=int, required=True)
    parser.add_argument("--original-weights-retained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host-store-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = estimate_fixed_slot_memory(
        num_layers=args.num_layers,
        num_experts_per_layer=args.num_experts_per_layer,
        expert_bytes=args.expert_bytes,
        num_slots=args.num_slots,
        original_weights_retained=args.original_weights_retained,
        host_store_enabled=args.host_store_enabled,
    )
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")
    print("FIXED_SLOT_MEMORY_ESTIMATE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

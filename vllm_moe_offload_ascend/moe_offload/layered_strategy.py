#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Any, Iterable

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.slot_simulator import (
    DEFAULT_HOST_TO_HBM_BANDWIDTH_GBPS,
    DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES,
    ExpertSizeTable,
)


@dataclass(frozen=True)
class LayeredStrategySummary:
    total_records: int
    full_weight_records: int
    slot_cache_records: int
    full_weight_layer_ids: tuple[int, ...]
    layers_requiring_full_weight: int
    slot_cache_hit_count: int
    slot_cache_miss_count: int
    slot_cache_eviction_count: int
    slot_cache_host_to_hbm_bytes: int
    slot_cache_estimated_load_ms: float
    num_slots: int
    fanout_threshold: int
    cache_scope: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "full_weight_records": self.full_weight_records,
            "slot_cache_records": self.slot_cache_records,
            "full_weight_layer_ids": list(self.full_weight_layer_ids),
            "layers_requiring_full_weight": self.layers_requiring_full_weight,
            "slot_cache_hit_count": self.slot_cache_hit_count,
            "slot_cache_miss_count": self.slot_cache_miss_count,
            "slot_cache_eviction_count": self.slot_cache_eviction_count,
            "slot_cache_host_to_hbm_bytes": self.slot_cache_host_to_hbm_bytes,
            "slot_cache_estimated_load_ms": self.slot_cache_estimated_load_ms,
            "num_slots": self.num_slots,
            "fanout_threshold": self.fanout_threshold,
            "cache_scope": self.cache_scope,
        }


class LayeredStrategyAnalyzer:
    """Offline prefill/full-weight plus decode/slot-cache strategy analysis."""

    def __init__(
        self,
        *,
        num_slots: int,
        fanout_threshold: int | None = None,
        cache_scope: str = "global",
        size_table: ExpertSizeTable | None = None,
        host_to_hbm_bandwidth_gbps: float = DEFAULT_HOST_TO_HBM_BANDWIDTH_GBPS,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than 0")
        self.num_slots = int(num_slots)
        self.fanout_threshold = int(fanout_threshold if fanout_threshold is not None else num_slots)
        if self.fanout_threshold <= 0:
            raise ValueError("fanout_threshold must be greater than 0")
        if cache_scope not in ("global", "per_layer"):
            raise ValueError("cache_scope must be either 'global' or 'per_layer'")
        self.cache_scope = cache_scope
        self.size_table = size_table or ExpertSizeTable(default_expert_bytes=DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES)
        self.host_to_hbm_bandwidth_gbps = float(host_to_hbm_bandwidth_gbps)

    def analyze(self, records: Iterable[dict[str, Any]]) -> LayeredStrategySummary:
        resident_by_scope: dict[int, set[ExpertKey]] = {}
        last_used_by_scope: dict[int, dict[ExpertKey, int]] = {}
        full_weight_layer_ids: set[int] = set()
        full_weight_records = 0
        slot_cache_records = 0
        slot_cache_hit_count = 0
        slot_cache_miss_count = 0
        slot_cache_eviction_count = 0
        slot_cache_host_to_hbm_bytes = 0
        total_records = 0

        for total_records, record in enumerate(records, start=1):
            layer_id = int(record["layer_id"])
            active_keys = _active_keys(record)
            if len(active_keys) > self.fanout_threshold:
                full_weight_records += 1
                full_weight_layer_ids.add(layer_id)
                continue

            slot_cache_records += 1
            scope_key = layer_id if self.cache_scope == "per_layer" else -1
            resident = resident_by_scope.setdefault(scope_key, set())
            last_used = last_used_by_scope.setdefault(scope_key, {})
            for key in active_keys:
                if key in resident:
                    slot_cache_hit_count += 1
                else:
                    slot_cache_miss_count += 1
                    if len(resident) >= self.num_slots:
                        victim = min(resident, key=lambda item: (last_used.get(item, -1), item.layer_id, item.expert_id))
                        resident.remove(victim)
                        slot_cache_eviction_count += 1
                    resident.add(key)
                    slot_cache_host_to_hbm_bytes += self.size_table.bytes_for(key)
                last_used[key] = total_records

        return LayeredStrategySummary(
            total_records=total_records,
            full_weight_records=full_weight_records,
            slot_cache_records=slot_cache_records,
            full_weight_layer_ids=tuple(sorted(full_weight_layer_ids)),
            layers_requiring_full_weight=len(full_weight_layer_ids),
            slot_cache_hit_count=slot_cache_hit_count,
            slot_cache_miss_count=slot_cache_miss_count,
            slot_cache_eviction_count=slot_cache_eviction_count,
            slot_cache_host_to_hbm_bytes=slot_cache_host_to_hbm_bytes,
            slot_cache_estimated_load_ms=self._estimate_load_ms(slot_cache_host_to_hbm_bytes),
            num_slots=self.num_slots,
            fanout_threshold=self.fanout_threshold,
            cache_scope=self.cache_scope,
        )

    def _estimate_load_ms(self, num_bytes: int) -> float:
        bytes_per_ms = self.host_to_hbm_bandwidth_gbps * 1_000_000_000 / 1000
        return num_bytes / bytes_per_ms if bytes_per_ms else 0.0


def _active_keys(record: dict[str, Any]) -> list[ExpertKey]:
    layer_id = int(record["layer_id"])
    return [ExpertKey(layer_id, int(expert_id)) for expert_id in record["active_experts"]]

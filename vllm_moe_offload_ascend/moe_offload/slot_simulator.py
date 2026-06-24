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
from vllm_moe_offload_ascend.moe_offload.policy import make_policy
from vllm_moe_offload_ascend.moe_offload.trace_collector import TraceRecord


DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES = 14_680_064
DEFAULT_HOST_TO_HBM_BANDWIDTH_GBPS = 24.0


@dataclass(frozen=True)
class ExpertSizeTable:
    default_expert_bytes: int = DEFAULT_QWEN3_30B_A3B_EXPERT_BYTES
    expert_bytes: dict[ExpertKey, int] | None = None

    def bytes_for(self, key: ExpertKey) -> int:
        if self.expert_bytes is None:
            return self.default_expert_bytes
        return self.expert_bytes.get(key, self.default_expert_bytes)


@dataclass(frozen=True)
class SlotSimulationSummary:
    total_records: int
    hit_count: int
    miss_count: int
    eviction_count: int
    host_to_hbm_bytes: int
    estimated_load_ms: float
    phase_opportunity_count: int
    num_slots: int
    policy: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "host_to_hbm_bytes": self.host_to_hbm_bytes,
            "estimated_load_ms": self.estimated_load_ms,
            "phase_opportunity_count": self.phase_opportunity_count,
            "num_slots": self.num_slots,
            "policy": self.policy,
        }


class SlotSimulator:
    def __init__(
        self,
        *,
        size_table: ExpertSizeTable | None = None,
        host_to_hbm_bandwidth_gbps: float = DEFAULT_HOST_TO_HBM_BANDWIDTH_GBPS,
    ) -> None:
        self.size_table = size_table or ExpertSizeTable()
        self.host_to_hbm_bandwidth_gbps = host_to_hbm_bandwidth_gbps

    def replay(
        self,
        records: Iterable[TraceRecord | dict[str, Any]],
        *,
        num_slots: int,
        policy_name: str,
    ) -> SlotSimulationSummary:
        if num_slots <= 0:
            raise ValueError("num_slots must be greater than 0")
        policy = make_policy(policy_name)
        resident: set[ExpertKey] = set()
        last_used: dict[ExpertKey, int] = {}
        hit_count = 0
        miss_count = 0
        eviction_count = 0
        host_to_hbm_bytes = 0
        phase_opportunity_count = 0
        total_records = 0

        for total_records, record in enumerate(records, start=1):
            active_keys = self._active_keys(record)
            record_hits = 0
            record_misses = 0
            for key in active_keys:
                if key in resident:
                    hit_count += 1
                    record_hits += 1
                else:
                    miss_count += 1
                    record_misses += 1
                    if len(resident) >= num_slots:
                        victim = policy.choose_victim(
                            sorted(resident),
                            last_used=last_used,
                            incoming=key,
                        )
                        resident.remove(victim)
                        eviction_count += 1
                    resident.add(key)
                    host_to_hbm_bytes += self.size_table.bytes_for(key)
                last_used[key] = total_records
            if record_hits > 0 and record_misses > 0:
                phase_opportunity_count += 1

        estimated_load_ms = self._estimate_load_ms(host_to_hbm_bytes)
        return SlotSimulationSummary(
            total_records=total_records,
            hit_count=hit_count,
            miss_count=miss_count,
            eviction_count=eviction_count,
            host_to_hbm_bytes=host_to_hbm_bytes,
            estimated_load_ms=estimated_load_ms,
            phase_opportunity_count=phase_opportunity_count,
            num_slots=num_slots,
            policy=policy_name,
        )

    @staticmethod
    def _active_keys(record: TraceRecord | dict[str, Any]) -> list[ExpertKey]:
        if isinstance(record, TraceRecord):
            layer_id = record.layer_id
            active_experts = record.active_experts
        else:
            layer_id = int(record["layer_id"])
            active_experts = tuple(int(expert) for expert in record["active_experts"])
        return [ExpertKey(int(layer_id), int(expert_id)) for expert_id in active_experts]

    def _estimate_load_ms(self, num_bytes: int) -> float:
        bytes_per_ms = self.host_to_hbm_bandwidth_gbps * 1_000_000_000 / 1000
        return num_bytes / bytes_per_ms if bytes_per_ms else 0.0

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
from enum import Enum
from itertools import count
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import torch

from vllm_ascend import envs
from vllm_moe_offload_ascend.moe_offload.config import MoeOffloadConfig
from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey
from vllm_moe_offload_ascend.moe_offload.host_store import HostExpertStore
from vllm_moe_offload_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState
from vllm_moe_offload_ascend.moe_offload.slot_mapping import ExpertSlotMapping, PreparedSlotWeights
from vllm_moe_offload_ascend.moe_offload.trace_collector import TraceCollector, TraceRecord
from vllm_moe_offload_ascend.moe_offload.expert_weight_release import release_layer_original_expert_weights
from vllm_moe_offload_ascend.moe_offload.tiered_residency import TieredResidencyPolicy
from vllm_moe_offload_ascend.moe_offload.transfer_engine import TransferEngine


@dataclass(frozen=True)
class MoeOffloadMemoryLedger:
    registered_layers: int
    host_experts: int
    original_expert_weight_bytes: int
    host_store_bytes: int
    slot_bank_bytes: int

    @property
    def original_expert_weights_retained(self) -> bool:
        return self.original_expert_weight_bytes > 0

    @property
    def total_managed_bytes(self) -> int:
        return self.original_expert_weight_bytes + self.host_store_bytes + self.slot_bank_bytes

    def to_jsonable(self) -> dict[str, int | bool]:
        return {
            "registered_layers": int(self.registered_layers),
            "host_experts": int(self.host_experts),
            "original_expert_weight_bytes": int(self.original_expert_weight_bytes),
            "host_store_bytes": int(self.host_store_bytes),
            "slot_bank_bytes": int(self.slot_bank_bytes),
            "original_expert_weights_retained": self.original_expert_weights_retained,
            "total_managed_bytes": int(self.total_managed_bytes),
        }


@dataclass(frozen=True)
class MoeExpertReleasePlan:
    ready: bool
    layers_ready: tuple[int, ...]
    blockers: tuple[str, ...]


class MoeOffloadDecisionPath(str, Enum):
    FULL_WEIGHT_PATH = "full_weight_path"
    SLOT_CACHE_PATH = "slot_cache_path"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class MoeOffloadPathDecision:
    path: MoeOffloadDecisionPath
    layer_id: int
    active_expert_count: int
    active_experts: tuple[int, ...]
    fanout_threshold: int
    full_weights_available: bool
    slot_cache_ready: bool
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "path": self.path.value,
            "layer_id": self.layer_id,
            "active_expert_count": self.active_expert_count,
            "active_experts": list(self.active_experts),
            "fanout_threshold": self.fanout_threshold,
            "full_weights_available": self.full_weights_available,
            "slot_cache_ready": self.slot_cache_ready,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MoeOffloadProfileEvent:
    name: str
    layer_id: int | None
    seconds: float
    memory_ledger: MoeOffloadMemoryLedger
    payload: dict[str, object] | None = None

    def to_jsonable(self) -> dict[str, object]:
        data = {
            "event": "moe_offload_profile",
            "name": self.name,
            "layer_id": self.layer_id,
            "seconds": self.seconds,
            "memory_ledger": self.memory_ledger.to_jsonable(),
        }
        if self.payload is not None:
            data["payload"] = self.payload
        return data


@dataclass(frozen=True)
class MoeOffloadTimelineEvent:
    name: str
    layer_id: int | None
    step_id: int
    start_ns: int
    end_ns: int
    payload: dict[str, object] | None = None

    @property
    def duration_ns(self) -> int:
        return max(0, int(self.end_ns) - int(self.start_ns))

    @property
    def seconds(self) -> float:
        return self.duration_ns / 1_000_000_000

    def to_jsonable(self) -> dict[str, object]:
        data = {
            "event": "moe_offload_timeline",
            "name": self.name,
            "layer_id": self.layer_id,
            "step_id": int(self.step_id),
            "start_ns": int(self.start_ns),
            "end_ns": int(self.end_ns),
            "duration_us": round(self.duration_ns / 1_000, 3),
            "seconds": round(self.seconds, 9),
        }
        if self.payload is not None:
            data["payload"] = self.payload
        return data


class MoeOffloadRuntime:
    def __init__(self, config: MoeOffloadConfig | None = None) -> None:
        self.config = config if config is not None else MoeOffloadConfig.from_env()
        self.trace_collector = TraceCollector(max_records=self.config.trace_max_records)
        self._step_counter = count()
        self._host_store = HostExpertStore()
        self._slot_banks: dict[int, ExpertSlotBank] = {}
        self._original_expert_weight_bytes_by_layer: dict[int, int] = {}
        self._released_original_weight_layers: set[int] = set()
        self._transfer_engine = TransferEngine()
        self._profile_events: list[MoeOffloadProfileEvent] = []

    def next_step_id(self) -> int:
        return int(next(self._step_counter))

    def trace_routing(
        self,
        *,
        layer_id: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_experts: int,
        mode: str = "unknown",
        step_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.should_trace:
            record = self.trace_collector.record(
                layer_id=layer_id,
                step_id=self.next_step_id() if step_id is None else int(step_id),
                topk_ids=topk_ids,
                num_experts=num_experts,
                mode=mode,
            )
            self._append_trace_record_jsonl(record)
        return topk_ids, topk_weights

    def export_trace(self, path: str | Path) -> int:
        return self.trace_collector.write_jsonl(path)

    @property
    def should_use_fixed_slots(self) -> bool:
        return self.config.enabled and not self.config.trace_only and self.config.num_slots > 0

    @property
    def should_use_layered_runtime(self) -> bool:
        return self.should_use_fixed_slots and self.config.layered_runtime

    def register_layer_for_fixed_slots(
        self,
        layer: torch.nn.Module,
        *,
        slot_device: torch.device | None = None,
    ) -> None:
        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            raise ValueError("layer.layer_id is required for fixed-slot registration")

        start = perf_counter()
        self._host_store.register_layer(layer)
        w13_weight = getattr(layer, "w13_weight")
        w2_weight = getattr(layer, "w2_weight")
        self._original_expert_weight_bytes_by_layer[layer_id] = _tensor_nbytes(w13_weight) + _tensor_nbytes(w2_weight)
        device = slot_device if slot_device is not None else w13_weight.device
        self._slot_banks[layer_id] = ExpertSlotBank(
            self.config.num_slots,
            tuple(int(dim) for dim in w13_weight.shape[1:]),
            tuple(int(dim) for dim in w2_weight.shape[1:]),
            dtype=w13_weight.dtype,
            device=device,
        )
        self._record_profile_event(
            "register_layer_for_fixed_slots",
            layer_id=layer_id,
            start=start,
        )

    def is_layer_registered(self, layer_id: int) -> bool:
        return int(layer_id) in self._slot_banks

    def is_resident_layer(self, layer_id: int) -> bool:
        return self.config.tiered_residency.is_resident_layer(int(layer_id))

    def should_use_fixed_slot_plan_for_layer(self, layer_id: int) -> bool:
        if not self.should_use_fixed_slots:
            return False
        return not self.is_resident_layer(int(layer_id))

    def memory_ledger(self) -> MoeOffloadMemoryLedger:
        original_bytes = sum(
            bytes_
            for layer_id, bytes_ in self._original_expert_weight_bytes_by_layer.items()
            if int(layer_id) not in self._released_original_weight_layers
        )
        return MoeOffloadMemoryLedger(
            registered_layers=len(self._slot_banks),
            host_experts=len(self._host_store),
            original_expert_weight_bytes=original_bytes,
            host_store_bytes=self._host_store.total_bytes,
            slot_bank_bytes=sum(slot_bank.total_bytes for slot_bank in self._slot_banks.values()),
        )

    def profiling_summary(self) -> dict[str, object]:
        total_seconds_by_event: dict[str, float] = {}
        for event in self._profile_events:
            total_seconds_by_event[event.name] = total_seconds_by_event.get(event.name, 0.0) + event.seconds
        return {
            "events": [event.to_jsonable() for event in self._profile_events],
            "total_seconds_by_event": total_seconds_by_event,
            "memory_ledger": self.memory_ledger().to_jsonable(),
        }

    def original_expert_weights_available_for_layer(self, layer_id: int) -> bool:
        return int(layer_id) not in self._released_original_weight_layers

    def decide_layered_path(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
        step_id: int | None = None,
    ) -> MoeOffloadPathDecision:
        normalized_layer_id = int(layer_id)
        unique_active_experts = _dedupe_preserve_order(active_experts)
        fanout_threshold = int(self.config.effective_fanout_threshold)
        full_weights_available = self.original_expert_weights_available_for_layer(normalized_layer_id)
        slot_cache_ready = self.should_use_fixed_slot_plan_for_layer(normalized_layer_id) and (
            normalized_layer_id in self._slot_banks
        )

        if not self.should_use_layered_runtime:
            path = MoeOffloadDecisionPath.SLOT_CACHE_PATH
            reason = "layered_runtime_disabled"
        elif len(unique_active_experts) > fanout_threshold:
            if full_weights_available:
                path = MoeOffloadDecisionPath.FULL_WEIGHT_PATH
                reason = "high_fanout_full_weights_available"
            else:
                path = MoeOffloadDecisionPath.FAIL_CLOSED
                reason = "high_fanout_full_weights_unavailable"
        elif slot_cache_ready:
            path = MoeOffloadDecisionPath.SLOT_CACHE_PATH
            reason = "low_fanout_slot_cache_ready"
        else:
            path = MoeOffloadDecisionPath.FAIL_CLOSED
            reason = "low_fanout_slot_cache_unavailable"

        decision = MoeOffloadPathDecision(
            path=path,
            layer_id=normalized_layer_id,
            active_expert_count=len(unique_active_experts),
            active_experts=unique_active_experts,
            fanout_threshold=fanout_threshold,
            full_weights_available=full_weights_available,
            slot_cache_ready=slot_cache_ready,
            reason=reason,
        )
        self._record_profile_event(
            "layered_path_decision",
            layer_id=normalized_layer_id,
            start=perf_counter(),
            payload={
                "step_id": self.next_step_id() if step_id is None else int(step_id),
                **decision.to_jsonable(),
            },
        )
        return decision

    def plan_original_weight_release(
        self,
        *,
        expected_layer_ids: tuple[int, ...],
        default_path_preserved: bool,
        host_store_is_complete: bool | None = None,
        allow_retained_original_weights: bool = False,
    ) -> MoeExpertReleasePlan:
        normalized_layer_ids = tuple(int(layer_id) for layer_id in expected_layer_ids)
        blockers: list[str] = []
        if not normalized_layer_ids:
            blockers.append("no_expected_layers")

        missing_layers = tuple(layer_id for layer_id in normalized_layer_ids if layer_id not in self._slot_banks)
        if missing_layers:
            blockers.append(f"layers_not_registered:{list(missing_layers)}")

        if not default_path_preserved:
            blockers.append("default_path_not_preserved")
        if host_store_is_complete is False:
            blockers.append("host_store_not_marked_complete")

        host_store_report = self._host_store.validate_complete_layers(normalized_layer_ids)
        blockers.extend(host_store_report.blockers)
        if self.memory_ledger().original_expert_weights_retained and not allow_retained_original_weights:
            blockers.append("original_expert_weights_still_retained")

        layers_ready = () if blockers else normalized_layer_ids
        return MoeExpertReleasePlan(
            ready=not blockers,
            layers_ready=layers_ready,
            blockers=tuple(blockers),
        )

    def release_original_expert_weights_if_ready(
        self,
        layer: torch.nn.Module,
        *,
        default_path_preserved: bool = True,
    ) -> MoeExpertReleasePlan:
        """Opt-in partial release for a single non-resident layer after host store is complete."""
        if not self.config.release_original_expert_weights:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("release_original_expert_weights_disabled",),
            )
        if not self.should_use_fixed_slots:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("fixed_slots_disabled",),
            )

        layer_id = int(getattr(layer, "layer_id", -1))
        if layer_id < 0:
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=("invalid_layer_id",),
            )
        if self.is_resident_layer(layer_id):
            return MoeExpertReleasePlan(
                ready=False,
                layers_ready=(),
                blockers=(f"resident_layer:{layer_id}",),
            )
        if layer_id in self._released_original_weight_layers:
            return MoeExpertReleasePlan(ready=True, layers_ready=(layer_id,), blockers=())

        plan = self.plan_original_weight_release(
            expected_layer_ids=(layer_id,),
            default_path_preserved=default_path_preserved,
            allow_retained_original_weights=True,
        )
        if not plan.ready:
            return plan

        start = perf_counter()
        release_layer_original_expert_weights(layer)
        self._released_original_weight_layers.add(layer_id)
        self._original_expert_weight_bytes_by_layer[layer_id] = 0
        self._record_profile_event(
            "release_original_expert_weights",
            layer_id=layer_id,
            start=start,
        )
        return MoeExpertReleasePlan(ready=True, layers_ready=(layer_id,), blockers=())

    def prepare_fixed_slot_plan(
        self,
        *,
        layer_id: int,
        step_id: int | None = None,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
    ) -> PreparedSlotWeights:
        if not self.should_use_fixed_slots:
            raise RuntimeError("fixed-slot plan requested while moe offload fixed slots are disabled")

        layer_id = int(layer_id)
        plan_step_id = self.next_step_id() if step_id is None else int(step_id)
        timeline_enabled = self._timeline_enabled()
        plan_start_ns = perf_counter_ns() if timeline_enabled else 0
        plan_payload: dict[str, object] = {
            "status": "ok",
            "active_expert_count": len(active_experts),
            "num_logical_experts": int(num_logical_experts),
        }
        prepared_weights: PreparedSlotWeights | None = None
        try:
            prepared_weights = self._prepare_fixed_slot_plan_impl(
                layer_id=layer_id,
                step_id=plan_step_id,
                active_experts=active_experts,
                num_logical_experts=num_logical_experts,
                device=device,
                timeline_enabled=timeline_enabled,
            )
            plan_payload["physical_expert_count"] = prepared_weights.physical_expert_count
            plan_payload["active_slot_ids"] = list(prepared_weights.mapping.active_slot_ids)
            return prepared_weights
        except Exception as exc:
            plan_payload["status"] = "error"
            plan_payload["error"] = str(exc)
            raise
        finally:
            if timeline_enabled:
                self._record_timeline_event(
                    "prepare_fixed_slot_plan",
                    layer_id=layer_id,
                    step_id=plan_step_id,
                    start_ns=plan_start_ns,
                    end_ns=perf_counter_ns(),
                    payload=plan_payload,
                )

    def _prepare_fixed_slot_plan_impl(
        self,
        *,
        layer_id: int,
        step_id: int,
        active_experts: tuple[int, ...],
        num_logical_experts: int,
        device: torch.device,
        timeline_enabled: bool,
    ) -> PreparedSlotWeights:
        if self.is_resident_layer(layer_id):
            raise RuntimeError(
                f"fixed-slot plan must not run on resident layer {layer_id}; use original NPU expert weights"
            )
        normalize_start_ns = perf_counter_ns() if timeline_enabled else 0
        unique_active_experts = _dedupe_preserve_order(active_experts)
        _validate_active_expert_ids(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
        )
        if timeline_enabled:
            self._record_timeline_event(
                "active_expert_normalize",
                layer_id=layer_id,
                step_id=step_id,
                start_ns=normalize_start_ns,
                end_ns=perf_counter_ns(),
                payload={
                    "active_expert_count": len(unique_active_experts),
                    "active_experts": list(unique_active_experts),
                },
            )
        if len(unique_active_experts) > self.config.num_slots:
            raise RuntimeError(
                f"active expert working set size {len(unique_active_experts)} exceeds num_slots={self.config.num_slots}"
            )

        slot_bank = self._slot_banks.get(layer_id)
        if slot_bank is None:
            raise RuntimeError(f"layer {layer_id} is not registered for fixed-slot execution")

        for expert_id in unique_active_experts:
            key = ExpertKey(layer_id, int(expert_id))
            lookup_start_ns = perf_counter_ns() if timeline_enabled else 0
            slot = slot_bank.lookup(key)
            if slot is not None and slot.state == SlotState.READY:
                slot.last_used_step = int(step_id)
                if timeline_enabled:
                    self._record_timeline_event(
                        "slot_cache_lookup",
                        layer_id=layer_id,
                        step_id=step_id,
                        start_ns=lookup_start_ns,
                        end_ns=perf_counter_ns(),
                        payload={
                            "expert_id": int(expert_id),
                            "cache_hit": True,
                            "slot_id": int(slot.slot_id),
                        },
                    )
                continue
            if timeline_enabled:
                self._record_timeline_event(
                    "slot_cache_lookup",
                    layer_id=layer_id,
                    step_id=step_id,
                    start_ns=lookup_start_ns,
                    end_ns=perf_counter_ns(),
                    payload={
                        "expert_id": int(expert_id),
                        "cache_hit": False,
                    },
                )

            allocate_start_ns = perf_counter_ns() if timeline_enabled else 0
            slot = slot_bank.allocate_for(key, step_id=step_id)
            if timeline_enabled:
                self._record_timeline_event(
                    "slot_allocate",
                    layer_id=layer_id,
                    step_id=step_id,
                    start_ns=allocate_start_ns,
                    end_ns=perf_counter_ns(),
                    payload={
                        "expert_id": int(expert_id),
                        "slot_id": int(slot.slot_id),
                        "slot_version": int(slot.version),
                    },
                )

            host_lookup_start_ns = perf_counter_ns() if timeline_enabled else 0
            bundle = self._host_store.get(layer_id, int(expert_id))
            if timeline_enabled:
                self._record_timeline_event(
                    "host_bundle_lookup",
                    layer_id=layer_id,
                    step_id=step_id,
                    start_ns=host_lookup_start_ns,
                    end_ns=perf_counter_ns(),
                    payload={
                        "expert_id": int(expert_id),
                        "slot_id": int(slot.slot_id),
                    },
                )

            transfer_start_ns = perf_counter_ns() if timeline_enabled else 0
            self._transfer_engine.load_sync(bundle, slot)
            if timeline_enabled:
                self._record_timeline_event(
                    "expert_h2d_load_sync",
                    layer_id=layer_id,
                    step_id=step_id,
                    start_ns=transfer_start_ns,
                    end_ns=perf_counter_ns(),
                    payload={
                        "expert_id": int(expert_id),
                        "slot_id": int(slot.slot_id),
                        "bytes": _bundle_nbytes(bundle),
                    },
                )

        mapping_start_ns = perf_counter_ns() if timeline_enabled else 0
        mapping = ExpertSlotMapping.from_slot_bank(
            layer_id=layer_id,
            active_experts=unique_active_experts,
            num_logical_experts=num_logical_experts,
            slot_bank=slot_bank,
            device=device,
        )
        if timeline_enabled:
            self._record_timeline_event(
                "slot_mapping_build",
                layer_id=layer_id,
                step_id=step_id,
                start_ns=mapping_start_ns,
                end_ns=perf_counter_ns(),
                payload={
                    "active_slot_ids": list(mapping.active_slot_ids),
                    "device": str(device),
                },
            )

        prepare_view_start_ns = perf_counter_ns() if timeline_enabled else 0
        prepared_weights = PreparedSlotWeights.from_slot_bank(slot_bank=slot_bank, mapping=mapping)
        if timeline_enabled:
            self._record_timeline_event(
                "prepared_slot_weights",
                layer_id=layer_id,
                step_id=step_id,
                start_ns=prepare_view_start_ns,
                end_ns=perf_counter_ns(),
                payload={
                    "physical_expert_count": int(prepared_weights.physical_expert_count),
                },
            )
        return prepared_weights

    def prepare_weights_for_execution(
        self,
        *,
        layer_id: int,
        active_experts: tuple[int, ...],
    ) -> None:
        del layer_id, active_experts
        if not self.should_use_fixed_slots:
            return None
        raise NotImplementedError(
            "fixed-slot execution requires num_logical_experts and backend wiring; "
            "use prepare_fixed_slot_plan() for the current safe planning path"
        )

    def _record_profile_event(
        self,
        name: str,
        *,
        layer_id: int | None,
        start: float,
        payload: dict[str, object] | None = None,
    ) -> None:
        event = MoeOffloadProfileEvent(
            name=name,
            layer_id=layer_id,
            seconds=perf_counter() - start,
            memory_ledger=self.memory_ledger(),
            payload=payload,
        )
        self._profile_events.append(event)
        self._append_profile_event_jsonl(event)

    @staticmethod
    def _timeline_enabled() -> bool:
        return bool(envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH)

    def _record_timeline_event(
        self,
        name: str,
        *,
        layer_id: int | None,
        step_id: int,
        start_ns: int,
        end_ns: int,
        payload: dict[str, object] | None = None,
    ) -> None:
        event = MoeOffloadTimelineEvent(
            name=name,
            layer_id=layer_id,
            step_id=step_id,
            start_ns=start_ns,
            end_ns=end_ns,
            payload=payload,
        )
        self._append_timeline_event_jsonl(event)

    @staticmethod
    def _append_profile_event_jsonl(event: MoeOffloadProfileEvent) -> None:
        profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
        if not profile_path:
            return
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_jsonable(), sort_keys=True) + "\n")

    @staticmethod
    def _append_timeline_event_jsonl(event: MoeOffloadTimelineEvent) -> None:
        profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
        if not profile_path:
            return
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_jsonable(), sort_keys=True) + "\n")

    @staticmethod
    def _append_trace_record_jsonl(record: TraceRecord) -> None:
        trace_path = envs.VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH
        if not trace_path:
            return
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_jsonable(), sort_keys=True) + "\n")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _bundle_nbytes(bundle) -> int:
    total = _tensor_nbytes(bundle.w13) + _tensor_nbytes(bundle.w2)
    if bundle.w13_scale is not None:
        total += _tensor_nbytes(bundle.w13_scale)
    if bundle.w2_scale is not None:
        total += _tensor_nbytes(bundle.w2_scale)
    return int(total)


_runtime: MoeOffloadRuntime | None = None


def get_moe_offload_runtime() -> MoeOffloadRuntime:
    global _runtime
    if _runtime is None:
        _runtime = MoeOffloadRuntime()
    return _runtime


def reset_moe_offload_runtime() -> None:
    global _runtime
    _runtime = None


def _dedupe_preserve_order(values: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen:
            seen.add(int_value)
            deduped.append(int_value)
    return tuple(deduped)


def _validate_active_expert_ids(
    *,
    layer_id: int,
    active_experts: tuple[int, ...],
    num_logical_experts: int,
) -> None:
    invalid_expert_ids = [
        int(expert_id)
        for expert_id in active_experts
        if int(expert_id) < 0 or int(expert_id) >= int(num_logical_experts)
    ]
    if invalid_expert_ids:
        raise ValueError(
            "fixed-slot active expert id out of range: "
            f"layer_id={int(layer_id)}, "
            f"num_logical_experts={int(num_logical_experts)}, "
            f"expert_ids={invalid_expert_ids}"
        )

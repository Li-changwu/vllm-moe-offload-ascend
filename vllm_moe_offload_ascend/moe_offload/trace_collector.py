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

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, init=False)
class TraceRecord:
    layer_id: int
    step_id: int
    mode: str
    source: str
    num_tokens: int
    top_k: int
    num_logical_experts: int
    fanout: int
    active_experts: tuple[int, ...]
    expert_token_counts: dict[int, int]
    group_list_type: int | None = None
    group_list_signature: str | None = None
    physical_expert_count: int | None = None

    def __init__(
        self,
        layer_id: int,
        step_id: int,
        mode: str,
        *args: Any,
        source: str = "logical_topk",
        num_tokens: int | None = None,
        top_k: int | None = None,
        num_logical_experts: int | None = None,
        fanout: int | None = None,
        active_experts: tuple[int, ...] | None = None,
        expert_token_counts: dict[int, int] | None = None,
        group_list_type: int | None = None,
        group_list_signature: str | None = None,
        physical_expert_count: int | None = None,
    ) -> None:
        if args:
            if len(args) != 5:
                raise TypeError(
                    "TraceRecord positional compatibility form is "
                    "(num_tokens, top_k, num_logical_experts, active_experts, expert_token_counts)"
                )
            num_tokens, top_k, num_logical_experts, active_experts, expert_token_counts = args
        if num_tokens is None or top_k is None or num_logical_experts is None:
            raise TypeError("num_tokens, top_k, and num_logical_experts are required")
        if active_experts is None or expert_token_counts is None:
            raise TypeError("active_experts and expert_token_counts are required")

        normalized_counts = {int(k): int(v) for k, v in expert_token_counts.items()}
        normalized_active = tuple(int(expert_id) for expert_id in active_experts)
        object.__setattr__(self, "layer_id", int(layer_id))
        object.__setattr__(self, "step_id", int(step_id))
        object.__setattr__(self, "mode", str(mode))
        object.__setattr__(self, "source", str(source))
        object.__setattr__(self, "num_tokens", int(num_tokens))
        object.__setattr__(self, "top_k", int(top_k))
        object.__setattr__(self, "num_logical_experts", int(num_logical_experts))
        object.__setattr__(self, "fanout", len(normalized_active) if fanout is None else int(fanout))
        object.__setattr__(self, "active_experts", normalized_active)
        object.__setattr__(self, "expert_token_counts", normalized_counts)
        object.__setattr__(self, "group_list_type", group_list_type)
        object.__setattr__(self, "group_list_signature", group_list_signature)
        object.__setattr__(self, "physical_expert_count", physical_expert_count)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "step_id": self.step_id,
            "mode": self.mode,
            "source": self.source,
            "num_tokens": self.num_tokens,
            "top_k": self.top_k,
            "num_logical_experts": self.num_logical_experts,
            "fanout": self.fanout,
            "active_experts": list(self.active_experts),
            "expert_token_counts": {str(k): v for k, v in self.expert_token_counts.items()},
            "group_list_type": self.group_list_type,
            "group_list_signature": self.group_list_signature,
            "physical_expert_count": self.physical_expert_count,
        }


class TraceCollector:
    def __init__(self, max_records: int = 4096) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be greater than 0")
        self._records: deque[TraceRecord] = deque(maxlen=max_records)
        self._latest_by_layer: dict[int, TraceRecord] = {}

    def record(
        self,
        *,
        layer_id: int,
        step_id: int,
        topk_ids: torch.Tensor,
        num_experts: int | None = None,
        num_logical_experts: int | None = None,
        mode: str = "unknown",
    ) -> TraceRecord:
        logical_experts = num_logical_experts if num_logical_experts is not None else num_experts
        if logical_experts is None:
            raise ValueError("num_logical_experts is required")
        return self.record_logical(
            layer_id=layer_id,
            step_id=step_id,
            topk_ids=topk_ids,
            num_logical_experts=logical_experts,
            mode=mode,
        )

    def record_logical(
        self,
        *,
        layer_id: int,
        step_id: int,
        topk_ids: torch.Tensor,
        num_logical_experts: int,
        mode: str = "unknown",
    ) -> TraceRecord:
        if topk_ids.ndim == 0:
            raise ValueError("topk_ids must have at least one dimension")

        counts = _counts_from_ids(topk_ids)

        record = TraceRecord(
            layer_id=int(layer_id),
            step_id=int(step_id),
            mode=mode,
            source="logical_topk",
            num_tokens=int(topk_ids.shape[0]),
            top_k=int(topk_ids.shape[1]) if topk_ids.ndim > 1 else 1,
            num_logical_experts=int(num_logical_experts),
            fanout=len(counts),
            active_experts=tuple(sorted(counts)),
            expert_token_counts=dict(sorted(counts.items())),
        )
        return self._append(record)

    def record_grouped(
        self,
        *,
        layer_id: int,
        step_id: int,
        group_list: torch.Tensor,
        group_list_type: int,
        physical_expert_count: int | None = None,
        mode: str = "unknown",
    ) -> TraceRecord:
        counts, signature = _counts_from_group_list(group_list, group_list_type)
        record = TraceRecord(
            layer_id=int(layer_id),
            step_id=int(step_id),
            mode=mode,
            source="grouped_dispatch",
            num_tokens=sum(counts.values()),
            top_k=1,
            num_logical_experts=0,
            fanout=len(counts),
            active_experts=tuple(sorted(counts)),
            expert_token_counts=dict(sorted(counts.items())),
            group_list_type=int(group_list_type),
            group_list_signature=signature,
            physical_expert_count=None if physical_expert_count is None else int(physical_expert_count),
        )
        return self._append(record)

    def _append(self, record: TraceRecord) -> TraceRecord:
        self._records.append(record)
        self._latest_by_layer[record.layer_id] = record
        return record

    def latest_for_layer(self, layer_id: int) -> TraceRecord | None:
        return self._latest_by_layer.get(int(layer_id))

    def records(self) -> list[TraceRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
        self._latest_by_layer.clear()

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [record.to_jsonable() for record in self._records]

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(record.to_jsonable(), sort_keys=True) + "\n" for record in self._records
        )

    def write_jsonl(self, path: str | Path) -> int:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_jsonl(), encoding="utf-8")
        return len(self._records)


def _tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    detached = tensor.detach()
    if detached.device.type != "cpu":
        detached = detached.to("cpu")
    return [int(value) for value in detached.reshape(-1).to(torch.int64).tolist()]


def _counts_from_ids(topk_ids: torch.Tensor) -> dict[int, int]:
    counts: dict[int, int] = {}
    for expert_id in _tensor_to_int_list(topk_ids):
        if expert_id < 0:
            continue
        counts[expert_id] = counts.get(expert_id, 0) + 1
    return counts


def _counts_from_group_list(group_list: torch.Tensor, group_list_type: int) -> tuple[dict[int, int], str]:
    values = _tensor_to_int_list(group_list)
    normalized_type = int(group_list_type)
    if normalized_type == 1:
        counts = {expert_id: count for expert_id, count in enumerate(values) if count > 0}
        return counts, "counts:" + ",".join(str(value) for value in values)
    if normalized_type == 0:
        counts: dict[int, int] = {}
        previous = 0
        for expert_id, cumulative in enumerate(values):
            count = int(cumulative) - previous
            previous = int(cumulative)
            if count > 0:
                counts[expert_id] = count
        return counts, "cumsum:" + ",".join(str(value) for value in values)
    return {}, f"unsupported:{normalized_type}"

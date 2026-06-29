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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import torch


class ComputeBucketDecisionPath(str, Enum):
    BUCKET = "bucket"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ComputeBucket:
    bucket_id: int
    signature: str
    sample_count: int = 0
    coverage_percent: float = 0.0
    active_expert_ids: tuple[int, ...] = ()
    compact_group_list: tuple[int, ...] = ()
    original_expert_count: int = 0

    @property
    def compact_expert_count(self) -> int:
        return len(self.active_expert_ids)

    def to_jsonable(self) -> dict[str, int | float | str | list[int]]:
        return {
            "bucket_id": self.bucket_id,
            "signature": self.signature,
            "sample_count": self.sample_count,
            "coverage_percent": self.coverage_percent,
            "active_expert_ids": list(self.active_expert_ids),
            "compact_group_list": list(self.compact_group_list),
            "original_expert_count": self.original_expert_count,
            "compact_expert_count": self.compact_expert_count,
        }


@dataclass(frozen=True)
class ComputeBucketDecision:
    path: ComputeBucketDecisionPath
    signature: str
    reason: str
    phase: str
    bucket_id: int | None = None
    bucket: ComputeBucket | None = None

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path.value,
            "signature": self.signature,
            "reason": self.reason,
            "phase": self.phase,
            "bucket_id": self.bucket_id,
        }
        if self.bucket is not None:
            payload["bucket"] = self.bucket.to_jsonable()
        return payload


class ComputeBucketClassifier:
    def __init__(
        self,
        *,
        phase: str,
        buckets: tuple[ComputeBucket, ...],
        coverage_percent: float = 0.0,
        fallback_percent: float = 100.0,
    ) -> None:
        self.phase = str(phase)
        self.buckets = buckets
        self.coverage_percent = float(coverage_percent)
        self.fallback_percent = float(fallback_percent)
        self._buckets_by_signature = {
            bucket.signature: bucket for bucket in buckets
        }

    @classmethod
    def from_plan(cls, plan: dict[str, Any]) -> "ComputeBucketClassifier":
        buckets = tuple(
            _bucket_from_plan_item(item)
            for item in plan.get("buckets", [])
            if item.get("signature")
        )
        return cls(
            phase=str(plan.get("phase", "unknown")),
            buckets=buckets,
            coverage_percent=_float(plan.get("coverage_percent")),
            fallback_percent=_float(plan.get("fallback_percent", 100.0)),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.buckets)

    def classify(
        self,
        *,
        group_list: torch.Tensor | None,
        group_list_type: int | None,
        phase: str = "unknown",
    ) -> ComputeBucketDecision:
        normalized_phase = str(phase)
        if (
            self.phase not in ("any", "mixed", "unknown")
            and normalized_phase != self.phase
        ):
            return ComputeBucketDecision(
                path=ComputeBucketDecisionPath.FALLBACK,
                signature="",
                reason="phase_mismatch",
                phase=normalized_phase,
            )

        signature = group_list_signature(group_list, group_list_type)
        bucket = self._buckets_by_signature.get(signature)
        if bucket is None:
            return ComputeBucketDecision(
                path=ComputeBucketDecisionPath.FALLBACK,
                signature=signature,
                reason="signature_not_planned",
                phase=normalized_phase,
            )
        return ComputeBucketDecision(
            path=ComputeBucketDecisionPath.BUCKET,
            signature=signature,
            reason="signature_matched",
            phase=normalized_phase,
            bucket_id=bucket.bucket_id,
            bucket=bucket,
        )


def load_compute_bucket_classifier(path: str | Path) -> ComputeBucketClassifier:
    source = Path(path)
    with source.open(encoding="utf-8") as f:
        payload = json.load(f)

    if "buckets" in payload:
        return ComputeBucketClassifier.from_plan(payload)

    for plan in payload.get("plans", []):
        if plan.get("target") != "P1-C":
            continue
        compute_bucket_plan = plan.get("compute_bucket_plan")
        if compute_bucket_plan:
            return ComputeBucketClassifier.from_plan(compute_bucket_plan)

    return ComputeBucketClassifier.from_plan({"phase": "unknown", "buckets": []})


def group_list_signature(
    group_list: torch.Tensor | None,
    group_list_type: int | None,
) -> str:
    if group_list is None or group_list_type is None:
        return "missing"
    values = [
        str(int(value))
        for value in group_list.detach().cpu().reshape(-1).tolist()
    ]
    prefix = {
        0: "cumsum",
        1: "counts",
    }.get(int(group_list_type), f"unsupported:{int(group_list_type)}")
    return f"{prefix}:{','.join(values)}"


def _bucket_from_plan_item(item: dict[str, Any]) -> ComputeBucket:
    signature = str(item.get("signature", ""))
    active_expert_ids, compact_group_list, original_expert_count = (
        _active_plan_from_signature(signature)
    )
    return ComputeBucket(
        bucket_id=_int(item.get("bucket_id")),
        signature=signature,
        sample_count=_int(item.get("sample_count")),
        coverage_percent=_float(item.get("coverage_percent")),
        active_expert_ids=active_expert_ids,
        compact_group_list=compact_group_list,
        original_expert_count=original_expert_count,
    )


def _active_plan_from_signature(
    signature: str,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    prefix, separator, payload = signature.partition(":")
    if separator != ":":
        return (), (), 0
    values = tuple(_int(value) for value in payload.split(",") if value.strip())
    if prefix == "counts":
        active_expert_ids = tuple(
            index for index, count in enumerate(values) if count > 0
        )
        compact_group_list = tuple(count for count in values if count > 0)
        return active_expert_ids, compact_group_list, len(values)
    if prefix == "cumsum":
        previous = 0
        active_expert_ids = []
        compact_group_list = []
        for index, cumulative_count in enumerate(values):
            if cumulative_count > previous:
                active_expert_ids.append(index)
                compact_group_list.append(cumulative_count)
            previous = cumulative_count
        return tuple(active_expert_ids), tuple(compact_group_list), len(values)
    return (), (), 0


def _float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ComputeBucket",
    "ComputeBucketClassifier",
    "ComputeBucketDecision",
    "ComputeBucketDecisionPath",
    "group_list_signature",
    "load_compute_bucket_classifier",
]

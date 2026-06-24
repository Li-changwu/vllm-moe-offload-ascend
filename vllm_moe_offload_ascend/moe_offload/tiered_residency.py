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

"""Tiered residency: which MoE layers keep full NPU expert weights vs slot-cache path."""

from __future__ import annotations

from dataclasses import dataclass


def parse_comma_separated_ints(raw: str) -> frozenset[int]:
    """Parse '0,1,2' or '' into a frozenset of ints."""
    text = (raw or "").strip()
    if not text:
        return frozenset()
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return frozenset(int(part) for part in parts)


@dataclass(frozen=True)
class TieredResidencyPolicy:
    """Decides per-layer execution mode for MVP-D.9.

    - Resident layers: keep original ``w13_weight`` / ``w2_weight`` on device;
      fixed-slot plan is skipped (no host→slot load on hot path).
    - Non-resident layers: host store + slot bank; optional partial release of
      original parameters after readiness guard passes.
    """

    resident_layer_ids: frozenset[int]
    release_original_expert_weights: bool

    def is_resident_layer(self, layer_id: int) -> bool:
        return int(layer_id) in self.resident_layer_ids

    def should_skip_fixed_slot_for_layer(self, layer_id: int) -> bool:
        return self.is_resident_layer(layer_id)
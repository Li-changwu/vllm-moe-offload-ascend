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

from abc import ABC, abstractmethod

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey


class ResidencyPolicy(ABC):
    @abstractmethod
    def choose_victim(
        self,
        candidates: list[ExpertKey],
        *,
        last_used: dict[ExpertKey, int],
        incoming: ExpertKey,
    ) -> ExpertKey:
        raise NotImplementedError


class LruPolicy(ResidencyPolicy):
    def choose_victim(
        self,
        candidates: list[ExpertKey],
        *,
        last_used: dict[ExpertKey, int],
        incoming: ExpertKey,
    ) -> ExpertKey:
        del incoming
        return min(candidates, key=lambda key: (last_used.get(key, -1), key.layer_id, key.expert_id))


class StickyLayerLruPolicy(LruPolicy):
    def choose_victim(
        self,
        candidates: list[ExpertKey],
        *,
        last_used: dict[ExpertKey, int],
        incoming: ExpertKey,
    ) -> ExpertKey:
        other_layer = [key for key in candidates if key.layer_id != incoming.layer_id]
        if other_layer:
            return super().choose_victim(other_layer, last_used=last_used, incoming=incoming)
        return super().choose_victim(candidates, last_used=last_used, incoming=incoming)


def make_policy(name: str) -> ResidencyPolicy:
    if name in ("lru", "deadline"):
        return LruPolicy()
    if name == "sticky_layer_lru":
        return StickyLayerLruPolicy()
    raise ValueError(f"unsupported MoE offload residency policy: {name}")

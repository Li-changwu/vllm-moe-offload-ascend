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
"""B1 top-k injection registry for the Option B three-way MoE seam.

In the decomposed path ``[moe_router] -> moe_offload_stage -> [moe_mlp]`` the
router op already computed ``(topk_weights, topk_ids)`` at the top level. To
avoid the apply-path recomputing them (and to feed the staged topk_ids into the
MLP), ``moe_mlp`` stashes them here keyed by ``layer_id`` right before invoking
the layer's forward path, and the apply select-site reads them back instead of
calling ``select_experts`` a second time. ``moe_mlp`` clears the entry in a
``finally`` so nothing leaks across layers/steps.

This is B1 (the design doc's chosen, least-invasive option): the apply-path keeps
its existing ``select_experts`` call; the only change is a guarded short-circuit
that activates ONLY when an injection exists for the current layer. When the seam
is disabled the registry is always empty, so ``has_injected_topk`` returns False
and the apply-path is byte-for-byte unchanged.

Single-threaded model-forward assumption: ``moe_mlp`` (which runs eager, and at
capture time records the kernel sequence) sets the entry and the apply-path reads
it synchronously on the same thread before ``moe_mlp`` clears it.
"""

from __future__ import annotations

import torch

# layer_id -> (topk_weights, topk_ids). Present ONLY for the duration of a single
# moe_mlp op call on the seam path.
_INJECTED: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def set_injected_topk(
    layer_id: int, topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> None:
    _INJECTED[int(layer_id)] = (topk_weights, topk_ids)


def has_injected_topk(layer_id: int) -> bool:
    return int(layer_id) in _INJECTED


def peek_injected_topk(
    layer_id: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Read without removing (the owning moe_mlp op clears in finally)."""
    return _INJECTED.get(int(layer_id))


def clear_injected_topk(layer_id: int) -> None:
    _INJECTED.pop(int(layer_id), None)

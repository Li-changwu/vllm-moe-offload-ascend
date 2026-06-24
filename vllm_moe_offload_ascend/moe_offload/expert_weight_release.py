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

"""Replace released MoE expert Parameters with zero-element placeholders on device."""

from __future__ import annotations

import torch


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def release_layer_original_expert_weights(layer: torch.nn.Module) -> int:
    """Drop original expert weight storage; layer must use slot-backed weights afterward."""
    w13 = getattr(layer, "w13_weight")
    w2 = getattr(layer, "w2_weight")
    freed = tensor_nbytes(w13) + tensor_nbytes(w2)
    device = w13.device
    dtype = w13.dtype
    layer.w13_weight = torch.nn.Parameter(
        torch.empty(0, device=device, dtype=dtype),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.empty(0, device=device, dtype=dtype),
        requires_grad=False,
    )
    return freed
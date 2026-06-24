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

import torch

from vllm_moe_offload_ascend.moe_offload.expert_key import ExpertKey


@dataclass(frozen=True)
class ExpertWeightBundle:
    layer_id: int
    expert_id: int
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None

    @property
    def key(self) -> ExpertKey:
        return ExpertKey(self.layer_id, self.expert_id)


@dataclass(frozen=True)
class HostExpertLayerSignature:
    layer_id: int
    num_experts: int
    w13_shape: tuple[int, ...]
    w13_dtype: torch.dtype
    w13_stride: tuple[int, ...]
    w2_shape: tuple[int, ...]
    w2_dtype: torch.dtype
    w2_stride: tuple[int, ...]


@dataclass(frozen=True)
class HostStoreCompletenessReport:
    complete: bool
    layers_checked: tuple[int, ...]
    blockers: tuple[str, ...]


class HostExpertStore:
    def __init__(self) -> None:
        self._weights: dict[ExpertKey, ExpertWeightBundle] = {}
        self._layer_signatures: dict[int, HostExpertLayerSignature] = {}

    def register_layer(self, layer: torch.nn.Module) -> None:
        layer_id = int(getattr(layer, "layer_id", -1))
        w13_weight = getattr(layer, "w13_weight")
        w2_weight = getattr(layer, "w2_weight")
        if w13_weight.shape[0] != w2_weight.shape[0]:
            raise ValueError("w13_weight and w2_weight must have the same number of experts")

        num_experts = int(w13_weight.shape[0])
        if num_experts <= 0:
            raise ValueError("host expert store requires at least one expert")
        self._weights = {key: bundle for key, bundle in self._weights.items() if key.layer_id != layer_id}
        self._layer_signatures[layer_id] = HostExpertLayerSignature(
            layer_id=layer_id,
            num_experts=num_experts,
            w13_shape=tuple(int(dim) for dim in w13_weight.shape[1:]),
            w13_dtype=w13_weight.dtype,
            w13_stride=_expert_stride(w13_weight),
            w2_shape=tuple(int(dim) for dim in w2_weight.shape[1:]),
            w2_dtype=w2_weight.dtype,
            w2_stride=_expert_stride(w2_weight),
        )
        for expert_id in range(num_experts):
            bundle = ExpertWeightBundle(
                layer_id=layer_id,
                expert_id=expert_id,
                w13=w13_weight[expert_id].detach().cpu().clone(),
                w2=w2_weight[expert_id].detach().cpu().clone(),
            )
            self._weights[bundle.key] = bundle

    def get(self, layer_id: int, expert_id: int) -> ExpertWeightBundle:
        return self._weights[ExpertKey(int(layer_id), int(expert_id))]

    def validate_complete_layers(self, expected_layer_ids: tuple[int, ...]) -> HostStoreCompletenessReport:
        normalized_layer_ids = tuple(int(layer_id) for layer_id in expected_layer_ids)
        blockers: list[str] = []

        missing_layers = [
            layer_id for layer_id in normalized_layer_ids if layer_id not in self._layer_signatures
        ]
        if missing_layers:
            blockers.append(f"host_store_missing_layers:{missing_layers}")

        for layer_id in normalized_layer_ids:
            signature = self._layer_signatures.get(layer_id)
            if signature is None:
                continue

            missing_expert_ids: list[int] = []
            for expert_id in range(signature.num_experts):
                key = ExpertKey(layer_id, expert_id)
                bundle = self._weights.get(key)
                if bundle is None:
                    missing_expert_ids.append(expert_id)
                    continue

                blockers.extend(_layout_mismatch_blockers(signature, bundle))

            if missing_expert_ids:
                blockers.append(f"host_store_missing_experts:layer={layer_id},experts={missing_expert_ids}")

        return HostStoreCompletenessReport(
            complete=not blockers,
            layers_checked=normalized_layer_ids,
            blockers=tuple(blockers),
        )

    @property
    def total_bytes(self) -> int:
        total = 0
        for bundle in self._weights.values():
            total += _tensor_nbytes(bundle.w13)
            total += _tensor_nbytes(bundle.w2)
            if bundle.w13_scale is not None:
                total += _tensor_nbytes(bundle.w13_scale)
            if bundle.w2_scale is not None:
                total += _tensor_nbytes(bundle.w2_scale)
        return total

    def __len__(self) -> int:
        return len(self._weights)


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _expert_stride(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor[0].stride())


def _layout_mismatch_blockers(
    signature: HostExpertLayerSignature,
    bundle: ExpertWeightBundle,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if tuple(int(dim) for dim in bundle.w13.shape) != signature.w13_shape:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13")
    if bundle.w13.dtype != signature.w13_dtype:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13_dtype")
    if tuple(int(value) for value in bundle.w13.stride()) != signature.w13_stride:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13_stride")
    if bundle.w13.device.type != "cpu":
        blockers.append(
            f"host_store_device_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w13={bundle.w13.device.type}"
        )
    if tuple(int(dim) for dim in bundle.w2.shape) != signature.w2_shape:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2")
    if bundle.w2.dtype != signature.w2_dtype:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2_dtype")
    if tuple(int(value) for value in bundle.w2.stride()) != signature.w2_stride:
        blockers.append(f"host_store_layout_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2_stride")
    if bundle.w2.device.type != "cpu":
        blockers.append(
            f"host_store_device_mismatch:layer={signature.layer_id},expert={bundle.expert_id},w2={bundle.w2.device.type}"
        )
    return tuple(blockers)

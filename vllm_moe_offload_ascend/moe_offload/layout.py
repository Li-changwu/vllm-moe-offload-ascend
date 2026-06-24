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

from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle


@dataclass(frozen=True)
class LayoutSignature:
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    device_type: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "LayoutSignature":
        return cls(
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=tensor.dtype,
            stride=tuple(int(value) for value in tensor.stride()),
            device_type=tensor.device.type,
        )


class LayoutValidator:
    @staticmethod
    def validate_bundle_matches_slot(bundle: ExpertWeightBundle, slot_bundle: ExpertWeightBundle) -> None:
        LayoutValidator.validate_copy_compatible(bundle, slot_bundle)
        LayoutValidator._validate_device("w13", bundle.w13, slot_bundle.w13)
        LayoutValidator._validate_device("w2", bundle.w2, slot_bundle.w2)

    @staticmethod
    def validate_copy_compatible(bundle: ExpertWeightBundle, slot_bundle: ExpertWeightBundle) -> None:
        LayoutValidator._validate_tensor("w13", bundle.w13, slot_bundle.w13)
        LayoutValidator._validate_tensor("w2", bundle.w2, slot_bundle.w2)

    @staticmethod
    def _validate_tensor(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
        source_signature = LayoutSignature.from_tensor(source)
        target_signature = LayoutSignature.from_tensor(target)
        if source_signature.shape != target_signature.shape:
            raise ValueError(f"{name} layout mismatch: shape {source_signature.shape} != {target_signature.shape}")
        if source_signature.dtype != target_signature.dtype:
            raise ValueError(f"{name} layout mismatch: dtype {source_signature.dtype} != {target_signature.dtype}")
        if source_signature.stride != target_signature.stride:
            raise ValueError(f"{name} layout mismatch: stride {source_signature.stride} != {target_signature.stride}")

    @staticmethod
    def validate_backend_ready(bundle: ExpertWeightBundle, *, expected_device_type: str) -> None:
        if bundle.w13.device.type != expected_device_type:
            raise ValueError(f"w13 backend device mismatch: {bundle.w13.device.type} != {expected_device_type}")
        if bundle.w2.device.type != expected_device_type:
            raise ValueError(f"w2 backend device mismatch: {bundle.w2.device.type} != {expected_device_type}")

    @staticmethod
    def _validate_device(name: str, source: torch.Tensor, target: torch.Tensor) -> None:
        if source.device.type != target.device.type:
            raise ValueError(f"{name} layout mismatch: device {source.device.type} != {target.device.type}")

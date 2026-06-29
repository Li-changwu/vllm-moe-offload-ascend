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

from vllm_moe_offload_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_moe_offload_ascend.moe_offload.layout import LayoutValidator
from vllm_moe_offload_ascend.moe_offload.slot_bank import ExpertSlot, SlotState


@dataclass(frozen=True)
class TransferReadyEvent:
    """Transfer-stream event plus the slots it makes consumable."""

    event: object | None
    slots: tuple[ExpertSlot, ...]

    def mark_ready(self) -> None:
        for slot in self.slots:
            slot.state = SlotState.READY


class TransferEngine:
    def __init__(self) -> None:
        self._h2d_stream = None

    def load_sync(self, bundle: ExpertWeightBundle, slot: ExpertSlot) -> None:
        LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        slot.w13.copy_(bundle.w13)
        slot.w2.copy_(bundle.w2)
        slot.state = SlotState.READY

    def load_async(
        self,
        bundle: ExpertWeightBundle,
        slot: ExpertSlot,
        *,
        wait_event=None,
        record_event: bool = True,
    ):
        """Queue a host-to-device expert load on a dedicated transfer stream."""
        import torch

        LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())
        stream = self._get_h2d_stream()
        if wait_event is not None:
            stream.wait_event(getattr(wait_event, "event", wait_event))
        slot.state = SlotState.LOADING
        with torch.npu.stream(stream):
            slot.w13.copy_(bundle.w13, non_blocking=True)
            slot.w2.copy_(bundle.w2, non_blocking=True)
            ready_event = None
            if record_event:
                ready_event = torch.npu.Event()
                ready_event.record(stream)
        if ready_event is None:
            stream.synchronize()
            slot.state = SlotState.READY
            return None
        return TransferReadyEvent(ready_event, (slot,))

    def load_many_async(
        self,
        loads: list[tuple[ExpertWeightBundle, ExpertSlot]],
        *,
        wait_event=None,
        record_event: bool = True,
        validate_layout: bool = True,
    ):
        """Queue several expert loads inside one transfer-stream context."""
        import torch

        if not loads:
            return None
        if validate_layout:
            for bundle, slot in loads:
                LayoutValidator.validate_copy_compatible(bundle, slot.as_bundle())

        stream = self._get_h2d_stream()
        if wait_event is not None:
            stream.wait_event(getattr(wait_event, "event", wait_event))
        with torch.npu.stream(stream):
            for bundle, slot in loads:
                slot.state = SlotState.LOADING
                slot.w13.copy_(bundle.w13, non_blocking=True)
                slot.w2.copy_(bundle.w2, non_blocking=True)
            ready_event = None
            if record_event:
                ready_event = torch.npu.Event()
                ready_event.record(stream)
        slots = tuple(slot for _, slot in loads)
        if ready_event is None:
            stream.synchronize()
            for slot in slots:
                slot.state = SlotState.READY
            return None
        return TransferReadyEvent(ready_event, slots)

    def synchronize(self) -> None:
        """Wait for all queued H2D copies on the transfer stream to finish."""
        if self._h2d_stream is not None:
            self._h2d_stream.synchronize()

    def _get_h2d_stream(self):
        if self._h2d_stream is None:
            import torch

            self._h2d_stream = torch.npu.Stream()
        return self._h2d_stream

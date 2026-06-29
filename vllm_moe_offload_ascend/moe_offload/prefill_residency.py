#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Profile-guided Prefill residency placement.

SEW Prefill has two very different paths:

* resident layers bypass the offload seam and run the native NPU fused MoE;
* offloaded layers enter the B2 routed-pair wave pipeline.

Given a fixed offload budget, the work-conserving choice is therefore to keep
the Prefill-expensive layers resident and spend the offload budget on cheaper
layers.  This module consumes Prefill profile JSONL emitted by
``b2_work_conserving_prefill`` and ``prefill_resident_native`` events and swaps
expensive default-offloaded layers with cheaper default-resident layers when
both have direct evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class PrefillLayerCost:
    layer_id: int
    calls: int = 0
    end_to_end_ms: float = 0.0
    loop_ms: float = 0.0
    scatter_ms: float = 0.0
    gmm_ms: float = 0.0
    stage_wait_ms: float = 0.0
    h2d_bytes: int = 0
    pairs: int = 0
    waves: int = 0
    active_experts: int = 0
    b2_calls: int = 0
    resident_native_calls: int = 0

    @property
    def score_ms(self) -> float:
        """Cost score used for residency decisions.

        Use total sampled time instead of per-call mean so a layer that appears
        more often in the profiled workload is protected naturally.
        """
        return float(self.end_to_end_ms)

    def add_event(self, record: dict[str, Any]) -> None:
        payload = record.get("payload") or {}
        control_ms = payload.get("control_ms") or {}
        waves = payload.get("waves") or ()
        event_name = str(record.get("name") or "")
        self.calls += 1
        if event_name == "b2_work_conserving_prefill":
            self.b2_calls += 1
            self.end_to_end_ms += _float(
                control_ms.get("end_to_end"),
                default=_float(record.get("seconds")) * 1000.0,
            )
            self.pairs += _int(payload.get("n_pairs"))
            self.waves += _int(payload.get("n_waves"))
            self.active_experts += _int(payload.get("n_active"))
            wave_summary = payload.get("wave_summary")
            if isinstance(wave_summary, dict):
                self.loop_ms += _float(control_ms.get("loop"))
                self.scatter_ms += _float(
                    wave_summary.get("layer_scatter_ms"),
                    default=_float(control_ms.get("scatter_total")),
                )
                self.gmm_ms += _float(wave_summary.get("gmm_ms"))
                self.stage_wait_ms += _float(wave_summary.get("stage_wait_ms"))
                self.h2d_bytes += _int(wave_summary.get("h2d_bytes"))
                return
            self.loop_ms += _float(control_ms.get("loop"))
            self.scatter_ms += _float(control_ms.get("scatter_total"))
        elif event_name == "prefill_resident_native":
            self.resident_native_calls += 1
            self.end_to_end_ms += _float(record.get("seconds")) * 1000.0
            self.pairs += _int(payload.get("n_tokens"))
        else:
            self.end_to_end_ms += _float(record.get("seconds")) * 1000.0
        for wave in waves if isinstance(waves, list) else ():
            if not isinstance(wave, dict):
                continue
            self.gmm_ms += _float(wave.get("gmm_ms"))
            self.stage_wait_ms += _float(wave.get("stage_wait_ms"))
            self.h2d_bytes += _int(wave.get("h2d_bytes"))

    def to_jsonable(self) -> dict[str, object]:
        mean = self.score_ms / self.calls if self.calls else 0.0
        return {
            "layer_id": int(self.layer_id),
            "calls": int(self.calls),
            "score_ms": round(self.score_ms, 3),
            "mean_score_ms": round(mean, 3),
            "loop_ms": round(self.loop_ms, 3),
            "scatter_ms": round(self.scatter_ms, 3),
            "gmm_ms": round(self.gmm_ms, 3),
            "stage_wait_ms": round(self.stage_wait_ms, 3),
            "h2d_bytes": int(self.h2d_bytes),
            "pairs": int(self.pairs),
            "waves": int(self.waves),
            "active_experts": int(self.active_experts),
            "b2_calls": int(self.b2_calls),
            "resident_native_calls": int(self.resident_native_calls),
        }


@dataclass(frozen=True)
class PrefillResidencyPlacement:
    resident_layer_ids: tuple[int, ...]
    offloaded_layer_ids: tuple[int, ...]
    swaps: tuple[dict[str, object], ...]
    profiled_layer_ids: tuple[int, ...]
    profiled_resident_layer_ids: tuple[int, ...]
    profiled_offloaded_layer_ids: tuple[int, ...]
    reason: str

    def to_jsonable(self) -> dict[str, object]:
        return {
            "resident_layer_ids": list(self.resident_layer_ids),
            "offloaded_layer_ids": list(self.offloaded_layer_ids),
            "swaps": list(self.swaps),
            "profiled_layer_ids": list(self.profiled_layer_ids),
            "profiled_resident_layer_ids": list(self.profiled_resident_layer_ids),
            "profiled_offloaded_layer_ids": list(self.profiled_offloaded_layer_ids),
            "reason": self.reason,
        }


def _float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, *, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def load_prefill_layer_costs(
    profile_path: str | Path,
    *,
    strict: bool = False,
    costs: dict[int, PrefillLayerCost] | None = None,
) -> dict[int, PrefillLayerCost]:
    """Load Prefill layer costs from a profile JSONL."""

    path = Path(profile_path)
    if not path.is_file():
        if strict:
            raise FileNotFoundError(str(path))
        return {}

    merged_costs: dict[int, PrefillLayerCost] = costs if costs is not None else {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                if strict:
                    raise ValueError(f"Invalid JSON on {path}:{lineno}") from None
                continue
            if not isinstance(record, dict):
                continue
            if record.get("name") not in (
                "b2_work_conserving_prefill",
                "prefill_resident_native",
            ):
                continue
            layer_id = _int(record.get("layer_id"), default=-1)
            if layer_id < 0:
                continue
            merged_costs.setdefault(layer_id, PrefillLayerCost(layer_id)).add_event(record)
    return merged_costs


def load_prefill_layer_costs_many(
    profile_paths: tuple[str | Path, ...],
    *,
    strict: bool = False,
) -> dict[int, PrefillLayerCost]:
    """Load and merge Prefill layer costs from multiple profile JSONLs."""

    costs: dict[int, PrefillLayerCost] = {}
    for profile_path in profile_paths:
        load_prefill_layer_costs(
            profile_path,
            strict=strict,
            costs=costs,
        )
    return costs


def plan_profile_guided_prefill_residency(
    *,
    num_layers: int,
    default_offloaded_layer_ids: tuple[int, ...],
    layer_costs: dict[int, PrefillLayerCost],
) -> PrefillResidencyPlacement:
    """Swap high-cost offloaded layers with lower-cost resident layers.

    The default group-based plan remains the fallback for unprofiled layers.
    We only swap when both sides have profile evidence and the default-offloaded
    layer is more expensive than the default-resident layer.
    """

    all_layers = tuple(range(int(num_layers)))
    offloaded = {int(layer_id) for layer_id in default_offloaded_layer_ids}
    profiled = {
        int(layer_id)
        for layer_id, cost in layer_costs.items()
        if 0 <= int(layer_id) < int(num_layers) and cost.calls > 0
    }
    if not profiled:
        resident = tuple(layer_id for layer_id in all_layers if layer_id not in offloaded)
        return PrefillResidencyPlacement(
            resident_layer_ids=resident,
            offloaded_layer_ids=tuple(sorted(offloaded)),
            swaps=(),
            profiled_layer_ids=(),
            profiled_resident_layer_ids=(),
            profiled_offloaded_layer_ids=(),
            reason="no_prefill_profile",
        )

    default_resident = set(all_layers) - offloaded
    profiled_offloaded = tuple(sorted(layer_id for layer_id in offloaded if layer_id in profiled))
    profiled_resident = tuple(sorted(layer_id for layer_id in default_resident if layer_id in profiled))
    offloaded_candidates = sorted(
        profiled_offloaded,
        key=lambda layer_id: layer_costs[layer_id].score_ms,
        reverse=True,
    )
    resident_candidates = sorted(
        profiled_resident,
        key=lambda layer_id: layer_costs[layer_id].score_ms,
    )
    if offloaded_candidates and not resident_candidates:
        resident = tuple(layer_id for layer_id in all_layers if layer_id not in offloaded)
        return PrefillResidencyPlacement(
            resident_layer_ids=resident,
            offloaded_layer_ids=tuple(sorted(offloaded)),
            swaps=(),
            profiled_layer_ids=tuple(sorted(profiled)),
            profiled_resident_layer_ids=profiled_resident,
            profiled_offloaded_layer_ids=profiled_offloaded,
            reason="missing_profiled_resident_layers",
        )
    if resident_candidates and not offloaded_candidates:
        resident = tuple(layer_id for layer_id in all_layers if layer_id not in offloaded)
        return PrefillResidencyPlacement(
            resident_layer_ids=resident,
            offloaded_layer_ids=tuple(sorted(offloaded)),
            swaps=(),
            profiled_layer_ids=tuple(sorted(profiled)),
            profiled_resident_layer_ids=profiled_resident,
            profiled_offloaded_layer_ids=profiled_offloaded,
            reason="missing_profiled_offloaded_layers",
        )

    swaps: list[dict[str, object]] = []
    for offloaded_layer, resident_layer in zip(
        offloaded_candidates,
        resident_candidates,
        strict=False,
    ):
        offloaded_score = layer_costs[offloaded_layer].score_ms
        resident_score = layer_costs[resident_layer].score_ms
        if offloaded_score <= resident_score:
            break
        offloaded.remove(offloaded_layer)
        offloaded.add(resident_layer)
        swaps.append(
            {
                "make_resident": int(offloaded_layer),
                "make_offloaded": int(resident_layer),
                "resident_gain_ms": round(offloaded_score - resident_score, 3),
                "old_offloaded_score_ms": round(offloaded_score, 3),
                "old_resident_score_ms": round(resident_score, 3),
            }
        )

    resident = tuple(layer_id for layer_id in all_layers if layer_id not in offloaded)
    reason = (
        "profile_guided_swaps"
        if swaps
        else "profile_available_no_beneficial_swap"
    )
    return PrefillResidencyPlacement(
        resident_layer_ids=resident,
        offloaded_layer_ids=tuple(sorted(offloaded)),
        swaps=tuple(swaps),
        profiled_layer_ids=tuple(sorted(profiled)),
        profiled_resident_layer_ids=profiled_resident,
        profiled_offloaded_layer_ids=profiled_offloaded,
        reason=reason,
    )

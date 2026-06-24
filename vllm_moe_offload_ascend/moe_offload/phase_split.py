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

"""MVP-D.11: Post-dispatch phase split semantic prototype.

This module provides the contracts, slicing, phase planning, partial MLP
execution, and scatter/gather logic for splitting the post-dispatch MoE MLP
compute into multiple phases (hit-first, then miss).  D.11 is a **semantic
prototype**: it proves that slicing + per-phase grouped matmul + gather
produces element-wise identical results to a single-phase run.  It does NOT
introduce async transfer, performance optimisation, or changes to router /
top-k / token count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from vllm_ascend import envs

if TYPE_CHECKING:
    import torch

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import (
        MoEMlpComputeInput,
        MoEWeights,
    )


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoEPhase:
    """A single phase of expert MLP execution.

    Each phase covers a contiguous (after extraction) range of tokens for a
    subset of the active experts.
    """

    phase_index: int
    # Logical expert ids covered by this phase (order as they appear in the
    # original sorted-hidden-states layout).
    expert_indices: tuple[int, ...]
    # Start / end (exclusive) offsets in the *original* sorted hidden states
    # that this phase will extract.  Because experts may be interleaved with
    # experts from other phases the extracted region is not necessarily
    # contiguous in the original buffer – the executor is responsible for
    # gathering the individual expert slices.
    token_slices: tuple[tuple[int, int], ...]
    # True when all experts in this phase are resident / slot-ready (hit).
    is_hit: bool

    @property
    def total_tokens(self) -> int:
        return sum(end - start for start, end in self.token_slices)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "phase_index": self.phase_index,
            "expert_indices": list(self.expert_indices),
            "token_slices": [[int(s), int(e)] for s, e in self.token_slices],
            "total_tokens": self.total_tokens,
            "is_hit": self.is_hit,
        }


@dataclass(frozen=True)
class MoEPhasePlan:
    """Complete phase-split plan for one MoE layer invocation."""

    phases: tuple[MoEPhase, ...]
    total_phases: int
    hit_phases: int
    miss_phases: int
    total_tokens: int
    # Optional human-readable reason for diagnostics.
    reason: str = ""

    def to_jsonable(self) -> dict[str, object]:
        return {
            "total_phases": self.total_phases,
            "hit_phases": self.hit_phases,
            "miss_phases": self.miss_phases,
            "total_tokens": self.total_tokens,
            "reason": self.reason,
            "phases": [p.to_jsonable() for p in self.phases],
        }


# ---------------------------------------------------------------------------
# Expert token slicing from group_list
# ---------------------------------------------------------------------------


def compute_expert_token_slices(
    group_list: "torch.Tensor",
    group_list_type: int,
) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for each expert in *group_list* order.

    *group_list_type*:
        - 1 (count):  ``group_list[i]`` = number of tokens for expert *i*.
        - 0 (cumsum): ``group_list[i]`` = cumulative end offset for expert *i*.
    """
    if group_list_type == 1:
        counts = [int(c) for c in group_list.cpu().tolist()]
    elif group_list_type == 0:
        cumsum = [int(c) for c in group_list.cpu().tolist()]
        counts = [cumsum[0]]
        for i in range(1, len(cumsum)):
            counts.append(cumsum[i] - cumsum[i - 1])
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}; D.11 supports 0 or 1 only.")

    slices: list[tuple[int, int]] = []
    offset = 0
    for count in counts:
        slices.append((offset, offset + count))
        offset += count
    return slices


# ---------------------------------------------------------------------------
# Phase planner
# ---------------------------------------------------------------------------


def _build_single_phase_plan(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    reason: str = "",
) -> MoEPhasePlan:
    """Fallback: every active expert in one phase."""
    total = sum(end - start for start, end in expert_slices)
    phase = MoEPhase(
        phase_index=0,
        expert_indices=active_expert_ids,
        token_slices=tuple(expert_slices),
        is_hit=True,
    )
    return MoEPhasePlan(
        phases=(phase,),
        total_phases=1,
        hit_phases=1,
        miss_phases=0,
        total_tokens=total,
        reason=reason,
    )


def plan_hit_miss_phases(
    expert_slices: list[tuple[int, int]],
    active_expert_ids: tuple[int, ...],
    slot_readiness: dict[int, bool] | None = None,
    max_phases: int = 2,
) -> MoEPhasePlan:
    """Split active experts into hit (ready) / miss phases.

    Parameters
    ----------
    expert_slices:
        Per-expert ``(start, end)`` offsets in the original sorted hidden
        states, aligned with *active_expert_ids*.
    active_expert_ids:
        Logical expert ids in the order they appear in ``group_list`` (which
        is also the order of *expert_slices*).
    slot_readiness:
        ``expert_id -> bool`` mapping.  Experts not in the map are treated as
        *ready* (hit).  Pass ``None`` to force single-phase.
    max_phases:
        Upper bound on the number of phases (default 2 = hit + miss).
    """
    if slot_readiness is None or max_phases <= 1:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="single_phase")

    if len(active_expert_ids) != len(expert_slices):
        raise ValueError(
            f"Mismatched lengths: active_expert_ids={len(active_expert_ids)}, "
            f"expert_slices={len(expert_slices)}"
        )

    hit_pairs: list[tuple[int, int]] = []  # (expert_id, slice_index)
    miss_pairs: list[tuple[int, int]] = []

    for slice_idx, expert_id in enumerate(active_expert_ids):
        if slot_readiness.get(int(expert_id), True):
            hit_pairs.append((int(expert_id), slice_idx))
        else:
            miss_pairs.append((int(expert_id), slice_idx))

    phases: list[MoEPhase] = []

    def _make_phase(idx: int, pairs: list[tuple[int, int]], is_hit: bool) -> MoEPhase | None:
        if not pairs:
            return None
        return MoEPhase(
            phase_index=idx,
            expert_indices=tuple(eid for eid, _ in pairs),
            token_slices=tuple(expert_slices[si] for _, si in pairs),
            is_hit=is_hit,
        )

    hit_phase = _make_phase(0, hit_pairs, True)
    if hit_phase is not None:
        phases.append(hit_phase)

    miss_phase = _make_phase(len(phases), miss_pairs, False)
    if miss_phase is not None:
        phases.append(miss_phase)

    if not phases:
        return _build_single_phase_plan(expert_slices, active_expert_ids, reason="empty_phases_fallback")

    total_tokens = sum(end - start for start, end in expert_slices)
    return MoEPhasePlan(
        phases=tuple(phases),
        total_phases=len(phases),
        hit_phases=sum(1 for p in phases if p.is_hit),
        miss_phases=sum(1 for p in phases if not p.is_hit),
        total_tokens=total_tokens,
        reason="hit_miss_split" if miss_pairs else "all_hit",
    )


# ---------------------------------------------------------------------------
# Partial MLP execution helpers
# ---------------------------------------------------------------------------


def _extract_phase_tokens(
    hidden_states: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Extract and concatenate tokens for a phase from sorted hidden states."""
    import torch

    if not token_slices:
        return torch.empty(0, hidden_states.size(1), dtype=hidden_states.dtype, device=hidden_states.device)

    chunks = [hidden_states[start:end] for start, end in token_slices]
    if len(chunks) == 1:
        return chunks[0].contiguous()
    return torch.cat(chunks, dim=0).contiguous()


def _build_phase_group_list(
    group_list: "torch.Tensor",
    group_list_type: int,
    expert_indices: tuple[int, ...],
) -> "torch.Tensor":
    """Build a new group_list tensor covering only *expert_indices*."""
    import torch

    if group_list_type == 1:
        selected = [group_list[i] for i in expert_indices]
        return torch.stack(selected) if selected else torch.empty(0, dtype=group_list.dtype, device=group_list.device)
    elif group_list_type == 0:
        # cumulative mode: need to reconstruct per-expert counts
        cumsum = group_list.cpu().tolist()
        prev = 0
        counts = []
        for i, end in enumerate(cumsum):
            counts.append(end - prev)
            prev = end
        selected = [counts[i] for i in expert_indices]
        result = torch.tensor(selected, dtype=group_list.dtype, device=group_list.device)
        # Convert back to cumulative
        return torch.cumsum(result, dim=0)
    else:
        raise ValueError(f"Unsupported group_list_type={group_list_type}")


def _slice_expert_weights(
    weights: "MoEWeights",
    expert_indices: tuple[int, ...],
) -> "MoEWeights":
    """Return a MoEWeights view sliced to *expert_indices*."""
    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEWeights as _MoEWeights

    def _index(w):
        if w is None:
            return None
        if isinstance(w, list):
            return [t[list(expert_indices)] for t in w]
        return w[list(expert_indices)]

    return _MoEWeights(
        w1=_index(weights.w1),
        w2=_index(weights.w2),
        w1_bias=_index(weights.w1_bias),
        w2_bias=_index(weights.w2_bias),
        w1_scale=_index(weights.w1_scale),
        w2_scale=_index(weights.w2_scale),
        w1_scale_bias=_index(weights.w1_scale_bias),
        w2_scale_bias=_index(weights.w2_scale_bias),
        w1_offset=_index(weights.w1_offset),
        w2_offset=_index(weights.w2_offset),
    )


# ---------------------------------------------------------------------------
# Scatter / gather
# ---------------------------------------------------------------------------


def _scatter_phase_output(
    full_output: "torch.Tensor",
    phase_output: "torch.Tensor",
    token_slices: tuple[tuple[int, int], ...],
) -> "torch.Tensor":
    """Write *phase_output* back into *full_output* at the given slices.

    Returns *full_output* (modified in-place).
    """
    offset = 0
    for start, end in token_slices:
        length = end - start
        if length > 0:
            full_output[start:end] = phase_output[offset : offset + length]
            offset += length
    return full_output


# ---------------------------------------------------------------------------
# Top-level phased MLP orchestrator
# ---------------------------------------------------------------------------


def execute_phased_mlp(
    *,
    mlp_compute_input: "MoEMlpComputeInput",
    phase_plan: MoEPhasePlan,
    _apply_mlp_fn=None,
) -> "torch.Tensor":
    """Execute MoE MLP in phases according to *phase_plan*.

    This replaces a single ``_apply_mlp(mlp_compute_input)`` call with one
    ``_apply_mlp`` call per phase, each operating on a contiguous subset of
    tokens / experts, then scatters the results back into a full output buffer.

    Parameters
    ----------
    mlp_compute_input:
        The full (single-phase) MLP compute input.
    phase_plan:
        Pre-computed phase split plan.
    _apply_mlp_fn:
        Callable ``(MoEMlpComputeInput) -> Tensor``.  Defaults to
        ``unified_apply_mlp``.

    Returns
    -------
    Tensor with the same shape / layout as a single-phase ``_apply_mlp`` call.
    """
    import torch

    if _apply_mlp_fn is None:
        from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp as _default

        _apply_mlp_fn = _default

    # Single phase → fast-path (no slicing overhead).
    if phase_plan.total_phases == 1:
        return _apply_mlp_fn(mlp_compute_input=mlp_compute_input)

    from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoEMlpComputeInput as _MoEMlpComputeInput

    hidden_states = mlp_compute_input.hidden_states
    group_list = mlp_compute_input.group_list
    group_list_type = mlp_compute_input.group_list_type
    hidden_size = hidden_states.size(-1)
    device = hidden_states.device
    dtype = hidden_states.dtype

    full_output = torch.empty(
        phase_plan.total_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )

    for phase in phase_plan.phases:
        if phase.total_tokens == 0:
            continue

        # Extract tokens
        phase_hidden = _extract_phase_tokens(hidden_states, phase.token_slices)

        # Build phase group_list
        phase_group_list = _build_phase_group_list(group_list, group_list_type, phase.expert_indices)

        # Slice weights
        phase_weights = _slice_expert_weights(mlp_compute_input.weights, phase.expert_indices)

        phase_input = _MoEMlpComputeInput(
            hidden_states=phase_hidden,
            group_list=phase_group_list,
            group_list_type=group_list_type,
            dynamic_scale=mlp_compute_input.dynamic_scale,
            topk_scales=mlp_compute_input.topk_scales,
            weights=phase_weights,
            quant=mlp_compute_input.quant,
            fusion=mlp_compute_input.fusion,
            activation=mlp_compute_input.activation,
            need_trans=mlp_compute_input.need_trans,
            dynamic_eplb=mlp_compute_input.dynamic_eplb,
        )

        phase_output = _apply_mlp_fn(mlp_compute_input=phase_input)

        # Scatter back
        _scatter_phase_output(full_output, phase_output, phase.token_slices)

    return full_output


# ---------------------------------------------------------------------------
# Observability helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSplitProfileEvent:
    name: str
    layer_id: int
    seconds: float
    phase_plan_jsonable: dict[str, object] | None = None
    fail_reason: str | None = None

    def to_jsonable(self) -> dict[str, object]:
        data: dict[str, object] = {
            "event": "phase_split",
            "name": self.name,
            "layer_id": self.layer_id,
            "seconds": round(self.seconds, 6),
        }
        if self.phase_plan_jsonable is not None:
            data["phase_plan"] = self.phase_plan_jsonable
        if self.fail_reason is not None:
            data["fail_reason"] = self.fail_reason
        return data


def _write_phase_split_profile_jsonl(event: PhaseSplitProfileEvent) -> None:
    profile_path = envs.VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH
    if not profile_path:
        return
    path = Path(profile_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.to_jsonable(), sort_keys=True) + "\n")

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
"""P0: Static startup advisor for (ascend-moe-offload-gb, num_slots).

Given device HBM, model config, and serving config, derives a reasonable
initial configuration without any profiling data. The intent is that most
users never need to set these parameters manually.

Typical usage (called once at engine init before apply_moe_offload_defaults):

    advisor = AutoConfigAdvisor.from_model_config(model_config)
    offload_gb, num_slots = advisor.suggest_config(
        device_hbm_gb=64.0,
        serving_config=ServingConfig(max_batch_size=32, max_seq_len=4096, top_k=2),
    )
    os.environ["VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS"] = str(num_slots)
    # then call apply_moe_offload_defaults(engine_args) with ascend_moe_offload_gb=offload_gb
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


_BYTES_PER_GIB = 1024 ** 3
# Conservative system reserve: driver overhead, NCCL buffers, misc.
_DEFAULT_SYSTEM_RESERVE_GIB = 2.0
# Target max B2 wave count per prefill layer (trade-off: 4 means 32 slots for 128 experts).
_DEFAULT_TARGET_B2_WAVES = 4


@dataclass(frozen=True)
class ServingConfig:
    """Serving deployment parameters used for NPU memory estimation."""

    max_batch_size: int = 32
    max_seq_len: int = 4096
    # MoE top-k per token
    top_k: int = 2
    # Target max B2 wave count for prefill (lower = larger num_slots, higher NPU cost).
    target_b2_waves: int = _DEFAULT_TARGET_B2_WAVES


@dataclass(frozen=True)
class AutoConfigAdvisor:
    """Advisor for (offload_gb, num_slots) given device + model + serving info."""

    # Model dimensions
    hidden_size: int
    moe_intermediate_size: int
    num_experts: int
    num_hidden_layers: int
    num_kv_heads: int
    head_dim: int
    # Bytes per parameter (2 for bf16/fp16, 1 for fp8/int8, 4 for fp32)
    dtype_bytes: int = 2
    # How many layers are MoE (vs dense attention-only). If unknown, assume all.
    num_moe_layers: int = -1  # -1 = use num_hidden_layers

    @classmethod
    def from_model_config(cls, model_config: dict[str, Any]) -> "AutoConfigAdvisor":
        """Construct from a model config.json dict."""
        def _get(key: str, *aliases: str, default: int = 0) -> int:
            for k in (key, *aliases):
                v = model_config.get(k)
                if v is not None:
                    return int(v)
            return default

        hidden_size = _get("hidden_size")
        num_layers = _get("num_hidden_layers")
        num_kv_heads = _get("num_key_value_heads", "num_kv_heads", default=_get("num_attention_heads"))
        head_dim = _get("head_dim", default=hidden_size // max(1, _get("num_attention_heads", default=1)))
        num_experts = _get("num_experts", "n_routed_experts", "moe_num_experts")
        moe_intermediate_size = _get("moe_intermediate_size", "intermediate_size")
        num_moe_layers = _get("num_moe_layers", default=-1)

        dtype_str = str(model_config.get("torch_dtype", "bfloat16")).lower().replace("torch.", "")
        dtype_bytes = 1 if "fp8" in dtype_str or "int8" in dtype_str else (4 if "float32" in dtype_str else 2)

        return cls(
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_hidden_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            num_moe_layers=num_moe_layers,
        )

    # ------------------------------------------------------------------
    # Memory estimation helpers
    # ------------------------------------------------------------------

    @property
    def _effective_moe_layers(self) -> int:
        return self.num_hidden_layers if self.num_moe_layers < 0 else self.num_moe_layers

    def expert_layer_gib(self) -> float:
        """Expert weights for ALL experts in one MoE layer (GiB)."""
        bytes_ = 3 * self.hidden_size * self.moe_intermediate_size * self.num_experts * self.dtype_bytes
        return bytes_ / _BYTES_PER_GIB

    def single_expert_gib(self) -> float:
        """Weights for one expert in one layer (GiB)."""
        bytes_ = 3 * self.hidden_size * self.moe_intermediate_size * self.dtype_bytes
        return bytes_ / _BYTES_PER_GIB

    def slot_bank_gib(self, num_slots: int, num_offloaded_layers: int) -> float:
        """Total NPU HBM consumed by slot banks (GiB)."""
        return self.single_expert_gib() * num_slots * num_offloaded_layers

    def estimate_kv_cache_gib(self, serving: ServingConfig) -> float:
        """Conservative KV cache estimate for max_batch * max_seq tokens (GiB)."""
        # k + v, all layers, all kv heads
        bytes_ = (
            2  # k + v
            * self.num_hidden_layers
            * self.num_kv_heads
            * self.head_dim
            * serving.max_batch_size
            * serving.max_seq_len
            * self.dtype_bytes
        )
        return bytes_ / _BYTES_PER_GIB

    def estimate_activation_gib(self, serving: ServingConfig) -> float:
        """Rough activation memory estimate during forward pass (GiB)."""
        # Approximate: hidden_states + intermediate buffers for one layer at a time,
        # scaled by batch * seq. This is a rough 4x factor over a single token buffer.
        bytes_ = 4 * self.hidden_size * serving.max_batch_size * serving.max_seq_len * self.dtype_bytes
        return bytes_ / _BYTES_PER_GIB

    # ------------------------------------------------------------------
    # Core suggestion logic
    # ------------------------------------------------------------------

    def suggest_num_slots(
        self,
        serving: ServingConfig,
        num_offloaded_layers: int,
        slot_budget_gib: float | None,
    ) -> int:
        """Suggest num_slots balancing B2 wave count, decode hit rate, and HBM budget.

        Three lower bounds, take max; then clamp to HBM budget.
        """
        n = self.num_experts

        # ① Cover decode working set: top_k experts per request × batch.
        #    Unique experts ≈ min(top_k * batch, n_experts) per step per layer.
        decode_working_set = min(serving.top_k * serving.max_batch_size, n)

        # ② Limit B2 wave count for prefill:
        #    waves = ceil(n / num_slots) ≤ target → num_slots ≥ ceil(n / target)
        b2_lower_bound = math.ceil(n / max(1, serving.target_b2_waves))

        # Take max of the two lower bounds.
        candidate = max(decode_working_set, b2_lower_bound, 1)

        # ③ Clamp by HBM budget: can't exceed what memory allows.
        if num_offloaded_layers > 0 and slot_budget_gib is not None:
            max_affordable = int(slot_budget_gib / (self.single_expert_gib() * num_offloaded_layers))
            candidate = min(candidate, max(1, max_affordable))

        # Cap at full residency (Regime A).
        return min(candidate, n)

    def suggest_config(
        self,
        device_hbm_gib: float,
        serving: ServingConfig,
        system_reserve_gib: float = _DEFAULT_SYSTEM_RESERVE_GIB,
    ) -> tuple[float, int]:
        """Return (recommended_offload_gib, recommended_num_slots).

        Strategy:
          free = device_hbm - kv_cache - activations - system_reserve
          offload as many MoE layers as possible within free, leaving a
          slot_fraction (30%) of free for slot banks.
        """
        kv_gib = self.estimate_kv_cache_gib(serving)
        act_gib = self.estimate_activation_gib(serving)
        free_gib = device_hbm_gib - kv_gib - act_gib - system_reserve_gib

        if free_gib <= 0:
            # No room to offload anything; return zeros (caller keeps default).
            return 0.0, 0

        total_expert_gib = self.expert_layer_gib() * self._effective_moe_layers

        # We can offload at most `free_gib` worth of expert weights.
        # Reserve 30% of `free_gib` headroom for slot banks.
        slot_budget_fraction = 0.30
        offload_budget_gib = free_gib * (1.0 - slot_budget_fraction)
        slot_budget_gib = free_gib * slot_budget_fraction

        offload_gib = min(offload_budget_gib, total_expert_gib)
        offload_gib = max(0.0, round(offload_gib, 2))

        if offload_gib <= 0:
            return 0.0, 0

        # Estimate how many layers get offloaded (mirrors derive_prefetch_defaults logic).
        layer_gib = self.expert_layer_gib()
        num_offloaded = min(
            self._effective_moe_layers,
            max(1, round(offload_gib / layer_gib)),
        )

        num_slots = self.suggest_num_slots(serving, num_offloaded, slot_budget_gib)
        return offload_gib, num_slots

    def explain(
        self,
        device_hbm_gib: float,
        serving: ServingConfig,
        system_reserve_gib: float = _DEFAULT_SYSTEM_RESERVE_GIB,
    ) -> dict[str, object]:
        """Return a human-readable breakdown of the suggestion (for logging/debugging)."""
        kv_gib = self.estimate_kv_cache_gib(serving)
        act_gib = self.estimate_activation_gib(serving)
        free_gib = device_hbm_gib - kv_gib - act_gib - system_reserve_gib
        offload_gib, num_slots = self.suggest_config(device_hbm_gib, serving, system_reserve_gib)
        layer_gib = self.expert_layer_gib()
        num_offloaded = max(1, round(offload_gib / layer_gib)) if offload_gib > 0 else 0
        b2_waves = math.ceil(self.num_experts / max(1, num_slots)) if num_slots > 0 else 0
        return {
            "device_hbm_gib": round(device_hbm_gib, 2),
            "kv_cache_estimate_gib": round(kv_gib, 2),
            "activation_estimate_gib": round(act_gib, 2),
            "system_reserve_gib": round(system_reserve_gib, 2),
            "free_gib": round(free_gib, 2),
            "recommended_offload_gib": round(offload_gib, 2),
            "recommended_num_slots": num_slots,
            "estimated_offloaded_layers": num_offloaded,
            "slot_bank_gib": round(self.slot_bank_gib(num_slots, num_offloaded), 2),
            "estimated_b2_waves_prefill": b2_waves,
            "regime": "A" if num_slots >= self.num_experts else ("B1" if b2_waves <= 1 else f"B2(waves={b2_waves})"),
        }
